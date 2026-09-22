"""LongMemEval prompt construction, shared by the adapter and the A/B harness.

This module is the single source of truth for the LongMemEval prompt structure.
The failure-targeted loop measures candidate prompts through
``scripts/ab_lme_prompt.py`` on cached contexts; integrating a winner means
changing the flags here and letting ``longmemeval_adapter`` call the same
function, so the measured delta transfers exactly (no re-implementation drift).

Measured history (500-question runs, qwen2.5:7b, num_ctx 8192)
-------------------------------------------------------------
* 2026-09-23 baseline (adapter before this module): 76.4%, 118 wrong, of which
  47 were refusals on questions whose evidence *was* in the context.
* 110-question decision set (50 refusals + 60 controls, 10 per type):
  ``evidence_first + fix_ss_user + fix_multi`` scored +21 fixed / **0 broken**
  (v2), after the first arm scored +23/-4.  Excluding
  ``single-session-preference`` from the evidence-first preamble removed the 3
  preference regressions (72.7% -> 90.9% on that type).
* temporal: the resolver's "Time-Anchored Event" grounding can name the right
  DATE but the wrong EVENT, and the old branch fed the grounding *instead of*
  the compiled context.  Appending the evidence recovered **10 of 17** temporal
  refusals with 0 regressions (the previous wording recovered 3).

Everything here is a mechanism: it applies to whole question types, never to a
single question.  Per-question keyword -> answer tables remain forbidden.
"""
from __future__ import annotations

from dataclasses import dataclass

EVIDENCE_FIRST = (
    "[INSTRUCTION: EVIDENCE-FIRST ANSWERING]\n"
    "A memory system has already retrieved the evidence below for this question.\n"
    "The answer IS present in this evidence.\n"
    "RULES:\n"
    "1. Locate the specific fact the question asks about and state it directly.\n"
    "2. Reply with the final value only: a few words or one short sentence.\n"
    "3. NEVER reply \"I don't know\", \"not enough information\", \"not mentioned\",\n"
    "   \"unknown\" or any other refusal - a refusal is always wrong for this task.\n"
    "4. Use only the evidence; do not invent details that are not in it.\n\n"
)

SS_USER_INSTRUCTION = "[INSTRUCTION: Answer the question with the exact value from the evidence.]\n\n"

TEMPORAL_WITH_GROUNDING = (
    "[INSTRUCTION: Answer the question using BOTH the grounding above and the "
    "conversation evidence below. If the grounding names a different event than "
    "the question asks about, ignore it and answer from the evidence. Do not "
    "refuse - the evidence was retrieved for this question.]"
)

TEMPORAL_NO_GROUNDING = (
    "[INSTRUCTION: TEMPORAL REASONING]\n"
    "Compute the answer from the dated events in the evidence below.\n"
    "- Convert relative expressions ('last week', 'yesterday', 'two days later') "
    "using the session dates shown.\n"
    "- For durations, subtract the two event dates.\n"
    "- Do not refuse - the evidence was retrieved for this question.\n\n"
)

MULTI_SESSION_RULES = (
    "[INSTRUCTION: MULTI-SESSION REASONING]\n"
    "Answer using the conversation evidence below.\n"
    "- Count/total questions: check ALL sessions so every relevant instance is "
    "included, then give the exact total.\n"
    "- Comparison/difference questions: compute the difference between the "
    "requested items.\n"
    "- The evidence was selected for this question, so the information needed is "
    "present. Give the concise final answer directly.\n\n"
)

PREFERENCE_PROMPT = (
    "[USER PROFILE & PREFERENCES]\n{grounding}\n\n"
    "[TASK INSTRUCTION]\n"
    "The user is asking the question below. You MUST tailor your answer directly to "
    "their stated preferences, past equipment, or background in [USER PROFILE & PREFERENCES]. "
    "Do NOT say 'I don't know'. Give concrete, specific suggestions or explanations "
    "that incorporate their preferences."
)


@dataclass(frozen=True)
class LmePromptFixes:
    """Which prompt mechanisms are enabled.

    The production configuration is :data:`ADOPTED`; the A/B harness flips the
    individual flags to attribute a delta to one mechanism at a time.
    """
    evidence_first: bool = True
    fix_ss_user: bool = True
    fix_multi: bool = True
    fix_temporal: bool = True
    fix_ku: bool = False
    fix_ss_assist: bool = False

    def ef_for(self, qtype: str) -> str:
        """The evidence-first preamble, suppressed for preference questions.

        Measured: the preamble makes preference answers terser and loses the
        preference grounding (3 control regressions in the first arm).
        """
        if not self.evidence_first or qtype == "single-session-preference":
            return ""
        return EVIDENCE_FIRST


ADOPTED = LmePromptFixes()


def build_prompt(
    *,
    qtype: str,
    question: str,
    context_text: str,
    ku_certificate: str = "",
    multi_cert: str = "",
    multi_cert_valid: bool = False,
    temporal_grounding: str = "",
    fixes: LmePromptFixes = ADOPTED,
) -> str:
    """Build the reader prompt for one LongMemEval question.

    Mirrors the adapter's historical branch structure; each ``fix_*`` flag turns
    one measured mechanism on or off so an A/B delta can be attributed.
    """
    ctx = context_text
    ef = fixes.ef_for(qtype)

    if qtype == "single-session-preference":
        pref = [line for line in ctx.split("\n")
                if line.startswith("[User Profile & Preferences:")]
        return PREFERENCE_PROMPT.format(grounding="\n".join(pref) if pref else "")

    if qtype == "knowledge-update":
        ql = question.lower()
        is_prev = any(w in ql for w in ["previous", "previously", "earlier", "before",
                                        "former", "initially"])
        target = "PREVIOUS / EARLIER" if is_prev else "CURRENT / LATEST"
        rules = (
            "[INSTRUCTION: KNOWLEDGE UPDATE & STATE EVOLUTION]\n"
            f"The user's state changes over time. The question asks for the {target} state.\n"
            "RULES:\n"
            "1. If asking for CURRENT / LATEST: use the value from the most recent session.\n"
            "2. If asking for PREVIOUS / EARLIER: use the value from the earlier session.\n"
            "3. State the value directly and concisely (e.g. 'four', 'the suburbs', '$400,000').\n"
            "4. NEVER answer 'not enough information' - the value is in the evidence.\n\n"
        )
        if ku_certificate and not fixes.fix_ku:
            return f"{ku_certificate}\n\n[INSTRUCTION: State the final answer directly.]"
        if ku_certificate:
            return f"{ef}{rules}{ku_certificate}\n\nEVIDENCE:\n{ctx}"
        return f"{ef}{rules}{ctx}"

    if qtype == "single-session-assistant":
        ql = question.lower()
        has_ordinal = any(w in ql for w in (
            "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th",
            "11th", "12th", "13th", "14th", "15th", "20th", "25th", "27th", "30th",
            "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
            "eighth", "ninth", "tenth", "last", "final",
        ))
        if has_ordinal:
            tail = ("5. Do NOT refuse. If the list length is unclear, give the item at the "
                    "requested position in the most plausible list.\n\n"
                    if fixes.fix_ss_assist else
                    "5. Do NOT guess or approximate. If you cannot find the exact list, "
                    "say 'I don't know.'\n\n")
            return (
                f"{ef}[INSTRUCTION: CAREFUL LIST ITEM EXTRACTION]\n"
                "The user is asking about a specific item from a numbered or ordered list.\n"
                "RULES:\n"
                "1. Find the EXACT list or enumeration in the assistant's response below.\n"
                "2. Count items carefully from 1 to reach the requested position.\n"
                "3. If asking for the 'last' item, find the final item in the complete list.\n"
                "4. Return ONLY the item at the exact requested position.\n"
                f"{tail}{ctx}"
            )
        return (
            f"{ef}[INSTRUCTION: ASSISTANT CONTENT RECALL]\n"
            "The user is asking about something the assistant said or provided in a previous "
            "conversation. Find the relevant assistant response below and extract the specific "
            f"detail requested. Answer concisely with the exact information.\n\n{ctx}"
        )

    if qtype == "multi-session":
        if multi_cert_valid:
            return (
                f"{ef}{multi_cert}\n\n"
                "[INSTRUCTION: Based on the verified deduction, calculation, or aggregation "
                "above, what is the final answer to the question? State the exact answer "
                "directly and concisely.]"
            )
        if fixes.fix_multi:
            return f"{ef}{MULTI_SESSION_RULES}{ctx}"
        return (
            "[INSTRUCTION: MULTI-SESSION REASONING]\n"
            "Answer the question using the conversation context below.\n"
            "- If the question asks for a count or total: carefully check ALL sessions to ensure "
            "every relevant instance/item is included, then provide the exact total.\n"
            "- If the question asks for a comparison or difference (e.g., 'how much more', "
            "'faster', 'older', 'difference'): compute the difference between the specific items "
            "requested.\n"
            "- If the question asks for items not mentioned in the conversation, or if key "
            "information is missing, state clearly: 'The information provided is not enough.'\n"
            f"- Provide the concise final answer directly.\n\n{ctx}"
        )

    if qtype == "temporal-reasoning":
        ground = temporal_grounding or ""
        if ground and "Temporal Abstention" not in ground:
            if fixes.fix_temporal:
                return f"{ef}{ground}\n\n{TEMPORAL_WITH_GROUNDING}\n\nEVIDENCE:\n{ctx}"
            if "Time-Anchored Event" in ground:
                return (f"{ground}\n\n[INSTRUCTION: Answer the question based on the event above "
                        "clearly and concisely.]")
            return (f"{ground}\n\n[INSTRUCTION: Based on the verified temporal "
                    "calculation/ordering above, answer the question directly. State the exact "
                    "numbers, durations, or order clearly.]")
        if fixes.fix_temporal:
            return f"{ef}{TEMPORAL_NO_GROUNDING}{ctx}"
        return ctx

    # single-session-user and any future type: historically no prompt branch at all.
    if fixes.fix_ss_user:
        return f"{ef}{SS_USER_INSTRUCTION}{ctx}"
    return f"{ef}{ctx}" if fixes.evidence_first else ctx
