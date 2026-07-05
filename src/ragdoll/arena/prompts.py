from __future__ import annotations

import re

ARENA_JUDGE_PROMPT = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants tasked to answer the user question displayed below. Choose the assistant that best answers the user question.

Your evaluation should consider factors such as correctness, helpfulness, completeness, accuracy, depth, and level of detail. Details are only useful if they answer the user question. If an answer contains non-relevant details, it should not be preferred over one that only uses relevant information.

Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible. Lastly, if both responses are citing the same sources of information and offer nearly identical information with minor differences, or if both responses are similarly good or similarly bad, output a tie.

Output your final verdict by strictly following this format:
"[[A]]" if Assistant A is better, "[[B]]" if Assistant B is better, and "[[Tie]]" for a tie.

[The Start of User's Question]
{query}
[The End of User's Question]

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
"""

_VERDICT_RE = re.compile(r"\s*\[\[(A|B|Tie)\]\]\s*")


def render_arena_prompt(*, query: str, answer_a: str, answer_b: str) -> str:
    return ARENA_JUDGE_PROMPT.format(query=query, answer_a=answer_a, answer_b=answer_b)


def parse_verdict(text: str) -> str | None:
    match = _VERDICT_RE.fullmatch(text)
    return match.group(1) if match else None
