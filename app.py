import re
import sys
import time
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from graph import build_graph, resume  # noqa: E402

PRIORITIES = ["Low", "Medium", "High", "Critical"]

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MAX_QUERY_LEN = 2000
MAX_SUMMARY_LEN = 200
MAX_DESC_LEN = 4000

# Session-scoped (in-memory, per browser session) rate limits -- not a
# substitute for server-side throttling, just a lightweight guard against
# casual abuse without adding external infra.
ASK_RATE_LIMIT = (10, 60)      # 10 questions / 60s
TICKET_RATE_LIMIT = (3, 600)   # 3 ticket submissions / 10 min


@st.cache_resource
def get_graph():
    return build_graph()


def _cfg():
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def _invoke(payload):
    st.session_state.result = get_graph().invoke(payload, config=_cfg())
    if st.session_state.result.get("email"):
        st.session_state.known_email = st.session_state.result["email"]


def _allow(log_key: str, limit: tuple[int, int]) -> bool:
    """Record this attempt and return whether it's within (max_count, window_seconds)."""
    max_count, window = limit
    now = time.time()
    log = [t for t in st.session_state.get(log_key, []) if now - t < window]
    if len(log) >= max_count:
        st.session_state[log_key] = log
        return False
    log.append(now)
    st.session_state[log_key] = log
    return True


st.set_page_config(page_title="OmniDesk", page_icon="🎫")
st.title("OmniDesk")
st.caption("Ask an HR or IT question, or check on a ticket you've already raised.")

query = st.text_input(
    "Your question",
    placeholder="e.g. Can I carry forward my comp-off to next quarter? / What's the status of my ticket?",
    max_chars=MAX_QUERY_LEN,
)

if st.button("Ask", type="primary") and query.strip():
    if not _allow("ask_log", ASK_RATE_LIMIT):
        st.error("Too many questions in a short time -- please wait a moment and try again.")
    else:
        st.session_state.thread_id = str(uuid.uuid4())
        payload = {"query": query}
        if st.session_state.get("known_email"):
            payload["email"] = st.session_state["known_email"]
        with st.spinner("Thinking..."):
            _invoke(payload)

result = st.session_state.get("result")

if result:
    if result.get("route"):
        st.markdown(f"**Route:** `{result['route']}`")
    if result.get("answer"):
        st.markdown(result["answer"])
        with st.expander("Retrieved context"):
            if result.get("hr_hits"):
                st.subheader("HR policies")
                for h in result["hr_hits"]:
                    st.markdown(f"- **{h.payload['topic']}** — {h.payload['question']}")
            if result.get("it_hits"):
                st.subheader("IT incidents")
                for h in result["it_hits"]:
                    st.markdown(f"- **{h.payload['title']}** (family {h.payload['family_id']})")

    interrupts = result.get("__interrupt__")
    kind = interrupts[0].value["type"] if interrupts else None

    # --- Ticket creation: step 1, the offer -----------------------------
    if kind == "collect_ticket_info":
        defaults = interrupts[0].value["defaults"]
        known_email = st.session_state.get("known_email")
        st.warning("I don't have enough information to answer that from the knowledge base.")
        st.subheader("Would you like me to file a support ticket?")
        with st.form("ticket_form"):
            if known_email:
                # Locks this session to the identity it already established,
                # so it can't silently refile as someone else mid-session.
                # A fresh session can still pick a different email -- real
                # verification (SSO/magic-link) would be needed to close that.
                st.text_input("Your email *", value=known_email, disabled=True)
                email = known_email
            else:
                email = st.text_input("Your email *", value=defaults["email"], max_chars=254)
            summary = st.text_input("Issue summary *", value=defaults["summary"], max_chars=MAX_SUMMARY_LEN)
            description = st.text_area("Issue description *", value=defaults["description"],
                                       height=150, max_chars=MAX_DESC_LEN)
            category = st.selectbox(
                "Category", ["HR", "IT", "HR/IT"],
                index=["HR", "IT", "HR/IT"].index(defaults["category"]),
            )
            priority = st.selectbox("Priority", PRIORITIES, index=PRIORITIES.index(defaults["priority"]))
            col1, col2 = st.columns(2)
            submitted = col1.form_submit_button("Continue", type="primary")
            declined = col2.form_submit_button("No thanks")

        if declined:
            with st.spinner("..."):
                _invoke(resume({"decline": True}))
            st.rerun()
        if submitted:
            if not email.strip() or not summary.strip() or not description.strip():
                st.error("Email, summary, and description are all required.")
            elif not EMAIL_RE.match(email.strip()):
                st.error("Please enter a valid email address.")
            elif not _allow("ticket_log", TICKET_RATE_LIMIT):
                st.error("Too many ticket submissions in a short time -- please wait a bit and try again.")
            else:
                with st.spinner("..."):
                    _invoke(resume({
                        "email": email.strip(), "summary": summary.strip(),
                        "description": description.strip(), "category": category, "priority": priority,
                    }))
                st.rerun()

    # --- Ticket creation: step 2, mandatory confirmation before the API call
    elif kind == "confirm_ticket":
        t = interrupts[0].value["ticket"]
        st.subheader("Confirm before I create this ticket")
        st.markdown(
            f"- **Reporter:** {t['email']}\n"
            f"- **Summary:** {t['summary']}\n"
            f"- **Category:** {t['category']}  |  **Priority:** {t['priority']}\n\n"
            f"**Description:**\n{t['description']}"
        )
        col1, col2 = st.columns(2)
        if col1.button("Yes, create the ticket", type="primary"):
            with st.spinner("Creating Jira ticket..."):
                _invoke(resume({"confirm": True}))
            st.rerun()
        if col2.button("Cancel"):
            with st.spinner("..."):
                _invoke(resume({"confirm": False}))
            st.rerun()

    # --- Status check: ask for email if not already known in this session --
    elif kind == "collect_status_email":
        st.subheader("What's the email you raised the ticket under?")
        known_email = st.session_state.get("known_email")
        with st.form("status_email_form"):
            if known_email:
                st.text_input("Your email *", value=known_email, disabled=True)
                status_email = known_email
            else:
                status_email = st.text_input("Your email *", max_chars=254)
            submitted = st.form_submit_button("Look up my tickets", type="primary")
        if submitted:
            if not status_email.strip():
                st.error("Please enter your email.")
            elif not EMAIL_RE.match(status_email.strip()):
                st.error("Please enter a valid email address.")
            elif not _allow("status_lookup_log", ASK_RATE_LIMIT):
                st.error("Too many lookups in a short time -- please wait a moment and try again.")
            else:
                with st.spinner("Looking up your tickets..."):
                    _invoke(resume(status_email.strip()))
                st.rerun()

    # --- Status check: let the employee pick among multiple open tickets ---
    elif kind == "choose_ticket":
        tickets = interrupts[0].value["tickets"]
        st.subheader("You have a few tickets on file — which one?")
        for t in tickets:
            st.markdown(f"- **{t['key']}** — {t['summary']} (status: `{t['status']}`)")
        chosen = st.selectbox("Ticket", [t["key"] for t in tickets])
        if st.button("Show details", type="primary"):
            with st.spinner("..."):
                _invoke(resume(chosen))
            st.rerun()

    # --- Resolved: show the ticket outcome (success / decline / cancel / error)
    elif result.get("ticket_result"):
        outcome = result["ticket_result"]
        if outcome.get("ok"):
            st.success(f"Ticket **{outcome['key']}** created. Track it here: {outcome['url']}")
        else:
            st.info(outcome["error"])
