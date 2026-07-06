from __future__ import annotations

import re

ARENA_JUDGE_PROMPT_NATIVE_RICH_HUMAN_VOTER = """Please act as a careful human Search Arena voter. Read the user's question and the two assistant answers, then choose the answer you would rather receive as the user.

Do not use a generic short-answer rubric. First infer what the user is trying to get done, then judge the answer by the dimensions that matter for that request. In search-result comparisons, a strong answer often wins because it gives more useful answer content, not because it is shorter or more polished.

Consider a broad set of possible dimensions:
- Intent match: answers the exact question, language, location, time frame, and requested scope.
- Answer density: contains many relevant, non-duplicative pieces of useful information.
- Coverage and recall: includes the important options, entities, subquestions, examples, caveats, and perspectives the user likely needs.
- Specificity: gives concrete names, dates, numbers, prices, addresses, source names, mechanisms, or distinctions when useful.
- Explanation quality: for "how", "why", effects, tradeoffs, or research questions, explains causes, mechanisms, and implications rather than only listing facts.
- Recommendation quality: for "best", "must try", "which", or planning questions, gives useful options, criteria, and practical details.
- Evidence and trust: claims are plausible, grounded, and not misleading; citations or source-derived details help when they support the answer.
- Calibration: handles uncertainty, currentness, locality, and assumptions honestly.
- Organization: makes a rich answer easy to use through grouping, ordering, and clear takeaways.
- Noise control: extra text is a problem only when it is irrelevant, repetitive, unsupported, or distracts from the user's need.

Prefer the answer with greater useful substance for this user's question. Do not penalize an answer merely for being long if the added material is relevant and useful. Do not reward brevity, citation count, or fluent wording by itself. If both answers would be similarly useful or similarly flawed, output a tie.

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

ARENA_JUDGE_PROMPT_W_TOPIC_RUBRIC_COVERAGE_COUNT = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants tasked to answer the user question displayed below. Choose the assistant that would receive the higher strict-vital human nugget score.

Your evaluation should consider factors such as correctness, helpfulness, completeness, accuracy, depth, and level of detail. Details are only useful if they answer the user question. If an answer contains non-relevant details, it should not be preferred over one that only uses relevant information.

Use the topic-specific rubric as a proxy for human nuggets. Criteria marked mandatory are vital. Optional or lower-priority criteria are secondary.

Before deciding, internally estimate:
1. Which mandatory criteria are clearly supported by Assistant A?
2. Which mandatory criteria are clearly supported by Assistant B?
3. Which answer supports more mandatory criteria with accurate, relevant information?
4. Does either answer contain important contradictions, unsupported claims, or off-topic material that should reduce confidence in its coverage?

Prefer the answer with higher supported mandatory-criterion coverage. Do not reward an answer for merely mentioning rubric words without actually answering the question. Do not reward length, fluency, citations, or optional details unless mandatory coverage is effectively tied. If both answers support the mandatory criteria similarly well, choose [[Both Good]]. If both answers miss most mandatory criteria or are similarly unreliable, choose [[Both Bad]].

Output your final verdict by strictly following this format:
"[[A]]" if Assistant A has higher supported mandatory coverage, "[[B]]" if Assistant B has higher supported mandatory coverage, "[[Both Good]]" if both responses satisfy the mandatory rubric criteria similarly well, and "[[Both Bad]]" if both responses miss most mandatory criteria or fail similarly badly.

[The Start of User's Question]
{query}
[The End of User's Question]

[The Start of Topic Rubric]
{rubric}
[The End of Topic Rubric]

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
"""

TIE_VERDICTS = frozenset({"Tie", "Both Good", "Both Bad"})
NATIVE_PROMPT_VARIANTS = {
    "default": ARENA_JUDGE_PROMPT_NATIVE_RICH_HUMAN_VOTER,
    "rich-human-voter": ARENA_JUDGE_PROMPT_NATIVE_RICH_HUMAN_VOTER,
}
ARENA_JUDGE_PROMPT_W_TOPIC_RUBRIC = ARENA_JUDGE_PROMPT_W_TOPIC_RUBRIC_COVERAGE_COUNT
TOPIC_RUBRIC_PROMPT_VARIANTS = {
    "default": ARENA_JUDGE_PROMPT_W_TOPIC_RUBRIC_COVERAGE_COUNT,
    "coverage-count": ARENA_JUDGE_PROMPT_W_TOPIC_RUBRIC_COVERAGE_COUNT,
}

_VERDICT_RE = re.compile(r"\s*\[\[(A|B|Tie|Both Good|Both Bad)\]\]\s*")


def render_arena_prompt(
    *,
    query: str,
    answer_a: str,
    answer_b: str,
    rubric: str | None = None,
    prompt_variant: str = "default",
) -> str:
    if rubric is not None:
        prompt = TOPIC_RUBRIC_PROMPT_VARIANTS.get(prompt_variant)
        if prompt is None:
            variants = ", ".join(sorted(TOPIC_RUBRIC_PROMPT_VARIANTS))
            raise ValueError(f"unknown topic rubric prompt variant {prompt_variant!r}; expected one of: {variants}")
        return prompt.format(
            query=query,
            answer_a=answer_a,
            answer_b=answer_b,
            rubric=rubric,
        )
    prompt = NATIVE_PROMPT_VARIANTS.get(prompt_variant)
    if prompt is None:
        variants = ", ".join(sorted(NATIVE_PROMPT_VARIANTS))
        raise ValueError(f"unknown native arena prompt variant {prompt_variant!r}; expected one of: {variants}")
    return prompt.format(query=query, answer_a=answer_a, answer_b=answer_b)


def parse_verdict(text: str) -> str | None:
    match = _VERDICT_RE.fullmatch(text)
    return match.group(1) if match else None
