import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from graph import build_graph  # noqa: E402


@st.cache_resource
def get_graph():
    return build_graph()


st.set_page_config(page_title="OmniDesk", page_icon="🎫")
st.title("OmniDesk")
st.caption("Ask an HR or IT question and get a grounded answer.")

query = st.text_input(
    "Your question",
    placeholder="e.g. Can I carry forward my comp-off to next quarter?",
)

if st.button("Ask", type="primary") and query.strip():
    with st.spinner("Thinking..."):
        result = get_graph().invoke({"query": query})

    st.markdown(f"**Route:** `{result['route']}`")
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
