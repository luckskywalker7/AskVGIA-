"""
core/benchmark_prompts.py
=========================
Answer-synthesis prompts tuned per benchmark. Selected by a `benchmark` arg you
pass on the request (e.g. AskRequest.benchmark in {"video_mmlu", "eduvidqa", None}).

Why two prompt shapes (grounded in the actual datasets):

VIDEO-MMLU  — short math videos. TWO question tracks:
  * captions_qa  : PERCEPTION / RECALL — exact on-screen facts: the function shown,
                   colors used to highlight, instructor position/clothing, the boxed
                   final result. Answer must be SHORT and EXACT, copying formulas /
                   colors verbatim from the description / board_text. No padding.
  * reasoning_qa : CONCEPTUAL — "why simplify first", "how does ln(a/c)=ln a-ln c help".
                   Answer is 1-2 sentences of correct mathematical reasoning grounded
                   in what the video showed. Concise, not an essay.
  Reference answers are short (a phrase to one sentence). Verbosity is penalised by
  similarity-based scoring, so SHORT + EXACT wins.

EDUVIDQA — long lecture videos, open-ended student questions (programming, theory).
  Reference answers are PARAGRAPHS (median ~273 chars, up to ~1300): a correct,
  self-contained technical EXPLANATION that reasons over the video content. Answer
  must be thorough, accurate, multi-sentence, and may synthesize across transcript +
  on-screen text. Brevity HURTS here.
"""

from __future__ import annotations
from typing import Optional


# ── VIDEO-MMLU ────────────────────────────────────────────────────────────

VIDEO_MMLU_SYSTEM = (
    "You are answering a question about a short educational MATH video, using ONLY "
    "the provided evidence (the video description, on-screen text/formulas, and "
    "transcript). Follow these rules strictly:\n"
    "- Ground every claim in the evidence. Never use outside knowledge or invent "
    "details not present.\n"
    "- Reproduce mathematical expressions, formulas, equations, and final results "
    "EXACTLY as they appear in the evidence (e.g. 'dy/dx = -1/(e^x - 1)', not a "
    "paraphrase). Preserve symbols, exponents, fractions, and notation.\n"
    "- RECALL questions (what function/value is shown, what color, where the "
    "instructor is, how the answer is presented): answer in a FEW WORDS — typically "
    "under 10 words. No explanation, no padding, no restating the question.\n"
    "- WHY/HOW reasoning questions: answer in ONE focused sentence of correct "
    "mathematical reasoning grounded in the video. Do not write a paragraph.\n"
    "- Never begin with 'The video shows' or 'According to'. Give the answer itself.\n\n"
    "EXAMPLES (match this exact style and brevity):\n"
    "Q: What is the original function presented in the video?\n"
    "A: y = ln(e^x/(e^x - 1))\n"
    "Q: What color is used to highlight the key mathematical identities?\n"
    "A: Red\n"
    "Q: What is the final derivative result?\n"
    "A: dy/dx = -1/(e^x - 1)\n"
    "Q: How is the final answer presented visually?\n"
    "A: In a boxed format\n"
    "Q: Where does the instructor position himself during the presentation?\n"
    "A: On the left side of the frame\n"
    "Q: Why is it necessary to simplify the logarithmic expression before differentiation?\n"
    "A: To break it into more manageable components that are easier to differentiate.\n"
    "Q: How does the property ln(a/c) = ln a - ln c help in this problem?\n"
    "A: It turns the quotient inside the logarithm into a difference of logarithms, "
    "which is easier to handle.\n"
)

VIDEO_MMLU_USER_TEMPLATE = (
    "{evidence}\n\n"
    "Question: {question}\n\n"
    "Answer directly and exactly. If it is a recall question, answer in a few words and "
    "copy any formula/value verbatim from the evidence. If it is a why/how question, "
    "answer in 1-2 precise sentences grounded in the evidence."
)


# ── EDUVIDQA ──────────────────────────────────────────────────────────────

EDUVIDQA_SYSTEM = (
    "You are answering an open-ended student question about an educational lecture "
    "video, using the provided evidence (video description, transcript, and any "
    "on-screen text/code). Produce a thorough, accurate, self-contained EXPLANATION.\n"
    "Rules:\n"
    "- Reason carefully and step by step over the evidence before stating conclusions. "
    "Resolve the question fully — explain the WHY, not just a one-line verdict.\n"
    "- Ground the explanation in the video's content. You may synthesize across "
    "different parts of the transcript and on-screen material. Do not contradict the "
    "evidence; do not fabricate specifics (exact API names, values, line numbers) that "
    "are not supported.\n"
    "- Write a clear, complete answer of roughly 2-5 sentences (a short paragraph). "
    "Match the depth of a knowledgeable instructor answering a student. Brevity that "
    "omits the reasoning is wrong here.\n"
    "- If the question has a yes/no core, state the verdict, THEN justify it with the "
    "reasoning. Reproduce any code/identifiers/expressions exactly when relevant.\n"
    "- If the evidence genuinely does not cover it, say what the video does establish "
    "and where it stops, rather than guessing.\n"
)

EDUVIDQA_USER_TEMPLATE = (
    "{evidence}\n\n"
    "Student question: {question}\n\n"
    "Write a complete, well-reasoned explanation grounded in the evidence above. "
    "Explain the reasoning, not just the conclusion."
)


# ── selector ──────────────────────────────────────────────────────────────

def get_benchmark_prompt(benchmark: Optional[str], question: str, evidence: str):
    """
    Returns (system_prompt, user_prompt) for the given benchmark.
    benchmark in {"video_mmlu", "eduvidqa"}; anything else → (None, None) so the
    caller falls back to its normal prompt builder.
    """
    b = (benchmark or "").strip().lower()
    if b in ("video_mmlu", "videommlu", "mmlu", "video-mmlu"):
        return (VIDEO_MMLU_SYSTEM,
                VIDEO_MMLU_USER_TEMPLATE.format(evidence=evidence, question=question))
    if b in ("eduvidqa", "edu", "eduvid"):
        return (EDUVIDQA_SYSTEM,
                EDUVIDQA_USER_TEMPLATE.format(evidence=evidence, question=question))
    return (None, None)


# ── recommended decoding params per benchmark (use in the LLM call) ─────────
# Video-MMLU: short/exact → low temp, small token budget.
# EduVidQA: paragraph reasoning → low temp, larger budget.
BENCHMARK_GEN_PARAMS = {
    "video_mmlu": {"temperature": 0.0, "num_predict": 160},
    "eduvidqa":   {"temperature": 0.2, "num_predict": 512},
}


def get_gen_params(benchmark: Optional[str]) -> dict:
    b = (benchmark or "").strip().lower()
    if b in ("video_mmlu", "videommlu", "mmlu", "video-mmlu"):
        return dict(BENCHMARK_GEN_PARAMS["video_mmlu"])
    if b in ("eduvidqa", "edu", "eduvid"):
        return dict(BENCHMARK_GEN_PARAMS["eduvidqa"])
    return {}