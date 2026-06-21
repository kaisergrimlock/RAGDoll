from __future__ import annotations

SUPPORT_EVAL_PROMPT = """
In this task, you will evaluate whether each statement is supported by its corresponding citations. 
Note that the system responses may appear very fluent and well-formed, but contain slight inaccuracies that are not easy to discern at first glance. 
Pay close attention to the text. Read it carefully as you would when proofreading.

You will be provided with a statement and its corresponding citation. It may be helpful to ask yourself whether it is accurate to say "according to the citation" with a
statement following this phrase. Be sure to check all of the information in the statement. You will be given three options:

- "Full Support": All of the information in the statement is supported in the citation.
- "Partial Support": Only some of the information is supported in the citation, but other parts of the information are missing from the citation.
- "No Support": This citation does not support any part of the statement.

Please provide your response based on the information in the citation. If you are unsure, use your best judgment. 
Respond as either "Full Support", "Partial Support", or "No Support" with no additional information.

Statement: {statement}

Citation: {citation}

Response:
"""


def render_support_prompt(*, statement: str, citation: str) -> str:
    return SUPPORT_EVAL_PROMPT.format(statement=statement, citation=citation)


def parse_support_label(text: str) -> str | None:
    lowered = text.lower()
    if "full support" in lowered:
        return "FS"
    if "partial support" in lowered:
        return "PS"
    if "no support" in lowered:
        return "NS"
    return None
