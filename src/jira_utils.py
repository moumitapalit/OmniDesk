"""Jira REST API v3 client for support-ticket escalation.

Used when the RAG pipeline can't answer a question from the knowledge base
(see graph.INSUFFICIENT_INFO_PHRASE). Every write here (create_ticket) is
only ever called after the employee has explicitly confirmed the ticket
contents in the UI -- this module itself does not gate on that; callers own
the human-in-the-loop confirmation step.
"""

import re

import requests

from config import JIRA_API_TOKEN, JIRA_AUTH_EMAIL, JIRA_ISSUE_TYPE, JIRA_PROJECT_KEY, JIRA_URL

_AUTH = (JIRA_AUTH_EMAIL, JIRA_API_TOKEN)
_HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}


def _configured() -> bool:
    return bool(JIRA_URL and JIRA_AUTH_EMAIL and JIRA_API_TOKEN and JIRA_PROJECT_KEY)


def _email_label(email: str) -> str:
    """Jira labels can't contain '@', '.', or spaces -- sanitize to a
    stable, exact-match label so tickets can be looked up by reporter email."""
    return "employee-" + re.sub(r"[^a-zA-Z0-9_-]", "-", email.strip().lower())


def _text_doc(text: str) -> dict:
    """Wrap plain text in the Atlassian Document Format v3 expects."""
    return {
        "type": "doc",
        "version": 1,
        "content": [{"type": "paragraph", "content": [{"type": "text", "text": text}]}],
    }


def create_ticket(
    email: str,
    summary: str,
    description: str,
    category: str = "IT",
    priority: str = "Medium",
) -> dict:
    """Create a Jira issue for an employee who couldn't be helped by the KB.

    Returns {"ok": True, "key": ..., "url": ...} on success, or
    {"ok": False, "error": ...} on any failure (including missing config).
    """
    if not _configured():
        return {"ok": False, "error": "Jira is not configured (missing JIRA_* settings/token)."}
    if not email or not summary:
        return {"ok": False, "error": "Employee email and a summary are required."}

    payload = {
        "fields": {
            "project": {"key": JIRA_PROJECT_KEY},
            "summary": summary,
            "description": _text_doc(f"Reported by: {email}\n\n{description}"),
            "issuetype": {"name": JIRA_ISSUE_TYPE},
            "labels": [_email_label(email), category.lower()],
        }
    }
    if priority:
        payload["fields"]["priority"] = {"name": priority}

    try:
        resp = requests.post(
            f"{JIRA_URL}/rest/api/3/issue",
            json=payload, auth=_AUTH, headers=_HEADERS, timeout=15,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        detail = getattr(exc.response, "text", str(exc)) if exc.response is not None else str(exc)
        return {"ok": False, "error": f"Jira request failed: {detail}"}

    key = resp.json()["key"]
    return {"ok": True, "key": key, "url": f"{JIRA_URL}/browse/{key}"}


def get_tickets_by_email(email: str) -> dict:
    """Look up tickets previously raised by this employee (via the email label).

    Returns {"ok": True, "tickets": [{"key", "summary", "status"}, ...]} or
    {"ok": False, "error": ...}.
    """
    if not _configured():
        return {"ok": False, "error": "Jira is not configured (missing JIRA_* settings/token)."}
    if not email:
        return {"ok": False, "error": "An employee email is required."}

    jql = f'project = "{JIRA_PROJECT_KEY}" AND labels = "{_email_label(email)}" ORDER BY created DESC'
    try:
        resp = requests.post(
            f"{JIRA_URL}/rest/api/3/search/jql",
            json={"jql": jql, "fields": ["summary", "status", "priority", "updated"]},
            auth=_AUTH, headers=_HEADERS, timeout=15,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        detail = getattr(exc.response, "text", str(exc)) if exc.response is not None else str(exc)
        return {"ok": False, "error": f"Jira request failed: {detail}"}

    tickets = [
        {
            "key": issue["key"],
            "summary": issue["fields"]["summary"],
            "status": issue["fields"]["status"]["name"],
            "updated": issue["fields"].get("updated"),
        }
        for issue in resp.json().get("issues", [])
    ]
    return {"ok": True, "tickets": tickets}


def get_ticket_comments(issue_key: str) -> dict:
    """Fetch comments/updates on a ticket so an employee can see support-team activity."""
    if not _configured():
        return {"ok": False, "error": "Jira is not configured (missing JIRA_* settings/token)."}

    try:
        resp = requests.get(
            f"{JIRA_URL}/rest/api/3/issue/{issue_key}/comment",
            auth=_AUTH, headers=_HEADERS, timeout=15,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        detail = getattr(exc.response, "text", str(exc)) if exc.response is not None else str(exc)
        return {"ok": False, "error": f"Jira request failed: {detail}"}

    comments = []
    for c in resp.json().get("comments", []):
        body = c.get("body", {})
        text = " ".join(
            t.get("text", "")
            for block in body.get("content", [])
            for t in block.get("content", [])
        ) if isinstance(body, dict) else str(body)
        comments.append({"author": c["author"]["displayName"], "created": c["created"], "text": text})
    return {"ok": True, "comments": comments}
