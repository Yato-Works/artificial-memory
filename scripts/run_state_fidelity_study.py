"""State Synthesis Fidelity Study (Phase C4).

Deconstructs why the current StateSynthesizer missed 15 questions compared to Gold State:
  Gold State:      22/32 correct (68.8%)
  Retrieved State:  7/32 correct (21.9%)
  Headroom:        15 questions (+46.9 pt gap)

Compares Gold State vs Generated State across all 32 Multi-Hop questions:
  - MATCH_EXACT: Entity, property, and value align with Gold State.
  - VALUE_PARTIAL_OR_MISSING: Entity & property detected, but value incomplete/missing.
  - PROPERTY_MISMATCH: Entity detected, but relational property misidentified.
  - NOT_GENERATED: StateSynthesizer produced 0 states.

Saves detailed diagnostic taxonomy to benchmark_results/protein/state_fidelity_study.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from artificial_memory.protein.state_synthesizer import StateSynthesizer
from artificial_memory.protein.state_trigger import StateTrigger
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter, LoCoMoQuestion

RESULTS_DIR = Path("benchmark_results/protein")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_gold_state(question: LoCoMoQuestion) -> dict:
    """Extract gold entity, property, and value for each multi-hop question."""
    q_lower = question.question.lower()
    gt = question.ground_truth.strip()
    ent = "Caroline" if "caroline" in q_lower else "Melanie"

    if "move" in q_lower:
        prop = "home country"
    elif "relationship" in q_lower:
        prop = "relationship status"
    elif "career" in q_lower:
        prop = "career path"
    elif "activities" in q_lower or "hobbies" in q_lower:
        prop = "activities"
    elif "bought" in q_lower or "items" in q_lower:
        prop = "items bought"
    elif "paint" in q_lower:
        prop = "paintings"
    elif "pottery" in q_lower:
        prop = "pottery types"
    elif "symbols" in q_lower:
        prop = "symbols"
    elif "children" in q_lower or "kids" in q_lower:
        prop = "children count"
    elif "beach" in q_lower:
        prop = "beach visits count"
    elif "destress" in q_lower:
        prop = "destress activities"
    elif "book" in q_lower:
        prop = "books read"
    elif "lgbtq" in q_lower or "events" in q_lower:
        prop = "events attended"
    elif "instruments" in q_lower:
        prop = "instruments played"
    elif "artists" in q_lower or "bands" in q_lower:
        prop = "musical artists"
    elif "support" in q_lower:
        prop = "support network"
    elif "changes" in q_lower:
        prop = "transition changes"
    elif "camp" in q_lower:
        prop = "camping locations"
    elif "like" in q_lower:
        prop = "kids preferences"
    elif "pets" in q_lower:
        prop = "pets names"
    elif "subject" in q_lower:
        prop = "shared subjects"
    elif "hike" in q_lower:
        prop = "hike activities"
    else:
        prop = "general"

    return {
        "entity": ent,
        "property": prop,
        "value": gt,
    }


def main():
    print("=" * 85)
    print("      AM APEX PHASE C4: STATE SYNTHESIS FIDELITY STUDY (32Q)")
    print("=" * 85)

    adapter = LoCoMoAdapter()
    turns, all_questions, ir_records = adapter.load_conversation(conv_idx=0)
    multihop_questions = [q for q in all_questions if q.category == 1]
    print(f"Analyzing {len(multihop_questions)} Category 1 (Multi-Hop) questions.")

    synthesizer = StateSynthesizer(max_states=2)
    trigger = StateTrigger()

    typology_counts = {
        "MATCH_EXACT": 0,
        "VALUE_PARTIAL_OR_MISSING": 0,
        "PROPERTY_MISMATCH": 0,
        "NOT_GENERATED": 0,
    }

    details = []

    print("\n" + f"{'Q#':<4} | {'Classification':<26} | {'Target Property':<20} | Question")
    print("-" * 85)

    for i, q in enumerate(multihop_questions):
        gold = generate_gold_state(q)
        trig_dec = trigger.evaluate(q.question, ir_records)
        generated_states = synthesizer.synthesize(q.question, ir_records)

        classification = "NOT_GENERATED"
        gen_ent = ""
        gen_prop = ""
        gen_val = ""

        if not generated_states:
            classification = "NOT_GENERATED"
        else:
            st = generated_states[0]
            gen_ent = st.entity
            gen_prop = st.target_property
            gen_val = st.value

            ent_match = gold["entity"].lower() in gen_ent.lower() or gen_ent.lower() in gold["entity"].lower()
            prop_match = gold["property"].lower() in gen_prop.lower() or gen_prop.lower() in gold["property"].lower()

            # Check value match
            gt_words = set(w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", gold["value"].lower()) if len(w) > 2)
            gen_words = set(w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", gen_val.lower()) if len(w) > 2)
            overlap = len(gt_words & gen_words)

            if ent_match and prop_match and overlap >= max(1, len(gt_words) // 2):
                classification = "MATCH_EXACT"
            elif ent_match and prop_match:
                classification = "VALUE_PARTIAL_OR_MISSING"
            elif ent_match:
                classification = "PROPERTY_MISMATCH"
            else:
                classification = "ENTITY_MISMATCH"

        typology_counts[classification] = typology_counts.get(classification, 0) + 1
        print(f"[{i+1:02d}] | {classification:<26} | {gold['property']:<20} | {q.question[:34]}")

        details.append({
            "question_id": q.question_id,
            "question": q.question,
            "gold_state": gold,
            "generated_state": {
                "entity": gen_ent,
                "property": gen_prop,
                "value": gen_val,
            } if generated_states else None,
            "classification": classification,
            "trigger_decision": {
                "should_synthesize": trig_dec.should_synthesize,
                "reason": trig_dec.trigger_reason,
            },
        })

    print("\n" + "=" * 85)
    print("      STATE SYNTHESIS FIDELITY BREAKDOWN (32 QUESTIONS)")
    print("=" * 85)
    for cat, cnt in typology_counts.items():
        pct = cnt / len(multihop_questions) * 100
        print(f"  * {cat:<26}: {cnt:2d}/32 ({pct:5.1f}%)")
    print("=" * 85)

    out_file = RESULTS_DIR / "state_fidelity_study.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_questions": len(multihop_questions),
                "typology_counts": typology_counts,
                "details": details,
            },
            f,
            indent=2,
        )
    print(f"Saved diagnostic taxonomy to {out_file}")


if __name__ == "__main__":
    main()
