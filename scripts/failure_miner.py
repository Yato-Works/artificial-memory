#!/usr/bin/env python3
"""Failure Miner for LoCoMo Benchmark.

Extracts failed questions from a run, classifies failure types,
generates reports, and creates a Failure Ledger (JSONL) for tracking
root causes and fixes without storing ground truth as training data.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from artificial_memory.research.benchmarks.failure_taxonomy_v2 import FailureClassifierV2, FailureCategory
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter


@dataclass
class FailureLedgerEntry:
    """Structured entry for the Failure Ledger (no ground truth stored)."""
    question_id: str
    category: str
    category_id: int
    failure_type: str
    retrieval: dict[str, Any]
    oracle_gap: bool
    evidence_state: str
    root_cause: dict[str, Any]
    affected_component: str
    commit: str | None = None
    before_fix: bool | None = None
    after_fix: bool | None = None
    notes: str = ""


def load_results(path: Path) -> dict:
    """Load results from a run file (conv_*_results.json or aggregated results.json)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_failures(data: dict) -> list[dict]:
    """Extract failed questions from results data."""
    failures = []
    for r in data.get("results", []):
        if not r.get("is_correct", True):
            failures.append(r)
    return failures


def classify_failure(
    classifier: FailureClassifierV2,
    question: dict,
    adapter: LoCoMoAdapter
) -> dict:
    """Classify a single failure using the failure taxonomy."""
    diag = classifier.classify(
        question=question.get("question", ""),
        ground_truth=question.get("ground_truth", ""),
        predicted_answer=question.get("predicted_answer", ""),
        context="",
        oracle_recall=question.get("oracle_recall", False),
        is_correct=question.get("is_correct", False),
        question_type=adapter.CATEGORY_NAMES.get(question.get("category", 0), "general"),
    )
    if diag:
        return {
            "category": diag.category.value,
            "confidence": diag.confidence,
            "reasoning": diag.reasoning,
        }
    return {"category": "UNKNOWN", "confidence": 0.0, "reasoning": ""}


def analyze_retrieval(question: dict, adapter: LoCoMoAdapter) -> dict:
    """Analyze retrieval state for a failed question."""
    oracle_recall = question.get("oracle_recall", False)
    evidence_ids = question.get("evidence_ids", []) if "evidence_ids" in question else []
    
    # Determine evidence state
    if not evidence_ids:
        evidence_state = "NO_EVIDENCE_REQUIRED"
    elif oracle_recall:
        evidence_state = "FULL_EVIDENCE_RETRIEVED"
    else:
        evidence_state = "PARTIAL_OR_MISSING_EVIDENCE"
    
    return {
        "oracle_recall": oracle_recall,
        "evidence_state": evidence_state,
        "num_evidence_ids": len(evidence_ids),
        "oracle_gap": not oracle_recall and question.get("is_correct") is False,
    }


def infer_root_cause(failure_type: str, retrieval: dict, category: str) -> dict:
    """Infer root cause from failure type and retrieval state."""
    oracle_recall = retrieval.get("oracle_recall", False)
    evidence_state = retrieval.get("evidence_state", "")
    
    # Map failure types to likely components and causes
    cause_map = {
        "RETRIEVAL_FAILURE": {
            "layer": "retriever",
            "type": "evidence_not_retrieved",
            "description": "Required evidence not found in retrieval",
        },
        "PARTIAL_EVIDENCE": {
            "layer": "retriever",
            "type": "incomplete_evidence_set",
            "description": "Some but not all required evidence retrieved",
        },
        "TEMPORAL": {
            "layer": "temporal_compiler",
            "type": "temporal_reasoning_failure",
            "description": "Temporal relation or date calculation error",
        },
        "MULTI_HOP": {
            "layer": "msc_compiler",
            "type": "multi_hop_reasoning_failure",
            "description": "Failed to chain multiple evidence pieces",
        },
        "ENTITY_LINK": {
            "layer": "entity_resolver",
            "type": "entity_linking_failure",
            "description": "Failed to link entity references across turns",
        },
        "CONTRADICTION": {
            "layer": "state_compiler",
            "type": "contradiction_handling_failure",
            "description": "Failed to resolve contradictory evidence",
        },
        "REASONING": {
            "layer": "llm_reasoning",
            "type": "inference_failure",
            "description": "Evidence retrieved but reasoning failed",
        },
        "GENERATION": {
            "layer": "llm_generation",
            "type": "answer_formulation_failure",
            "description": "Correct reasoning but answer formulation error",
        },
    }
    
    # Default to retrieval if oracle_recall is False
    if not oracle_recall and evidence_state != "FULL_EVIDENCE_RETRIEVED":
        if failure_type not in cause_map:
            return cause_map["RETRIEVAL_FAILURE"]
    
    return cause_map.get(failure_type, {
        "layer": "unknown",
        "type": "unclassified",
        "description": f"Unclassified failure: {failure_type}",
    })


def infer_affected_component(root_cause: dict) -> str:
    """Map root cause layer to affected component name."""
    layer_map = {
        "retriever": "SteroidEngine/Retriever",
        "temporal_compiler": "TemporalCompiler (CHRONOS)",
        "msc_compiler": "MSC Compiler (Protein)",
        "entity_resolver": "EntityResolver",
        "state_compiler": "StateCompiler/Overdrive",
        "llm_reasoning": "LLM Reasoning (Reader)",
        "llm_generation": "LLM Generation (Reader)",
        "unknown": "Unknown",
    }
    return layer_map.get(root_cause.get("layer", "unknown"), "Unknown")


def generate_ledger_entry(
    question: dict,
    failure_classification: dict,
    retrieval: dict,
    adapter: LoCoMoAdapter,
    commit: str | None = None
) -> FailureLedgerEntry:
    """Generate a Failure Ledger entry (no ground truth)."""
    category_id = question.get("category", 0)
    category_name = adapter.CATEGORY_NAMES.get(category_id, f"cat-{category_id}")
    
    root_cause = infer_root_cause(
        failure_classification.get("category", "UNKNOWN"),
        retrieval,
        category_name
    )
    
    return FailureLedgerEntry(
        question_id=question.get("question_id", "unknown"),
        category=category_name,
        category_id=category_id,
        failure_type=failure_classification.get("category", "UNKNOWN"),
        retrieval=retrieval,
        oracle_gap=retrieval.get("oracle_gap", False),
        evidence_state=retrieval.get("evidence_state", "UNKNOWN"),
        root_cause=root_cause,
        affected_component=infer_affected_component(root_cause),
        commit=commit,
        before_fix=None,
        after_fix=None,
        notes=f"Confidence: {failure_classification.get('confidence', 0):.2f}"
    )


def write_ledger(entries: list[FailureLedgerEntry], path: Path, append: bool = False) -> None:
    """Write Failure Ledger entries to JSONL file."""
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")


def generate_report(
    failures: list[dict],
    classifications: list[dict],
    retrievals: list[dict],
    adapter: LoCoMoAdapter,
    output_dir: Path
) -> None:
    """Generate human-readable failure analysis report."""
    report_path = output_dir / "failure_report.md"
    
    # Category breakdown
    cat_counts = defaultdict(int)
    type_counts = defaultdict(int)
    oracle_gap_counts = defaultdict(int)
    evidence_state_counts = defaultdict(int)
    component_counts = defaultdict(int)
    
    for f, c, r in zip(failures, classifications, retrievals):
        cat = adapter.CATEGORY_NAMES.get(f.get("category", 0), f"cat-{f.get('category', 0)}")
        cat_counts[cat] += 1
        type_counts[c.get("category", "UNKNOWN")] += 1
        if r.get("oracle_gap"):
            oracle_gap_counts[cat] += 1
        evidence_state_counts[r.get("evidence_state", "UNKNOWN")] += 1
        root_cause = infer_root_cause(c.get("category", "UNKNOWN"), r, cat)
        component_counts[infer_affected_component(root_cause)] += 1
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# LoCoMo Failure Analysis Report\n\n")
        f.write(f"Total failures analyzed: {len(failures)}\n\n")
        
        f.write("## By Category\n")
        for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
            f.write(f"- {cat}: {count}\n")
        f.write("\n")
        
        f.write("## By Failure Type\n")
        for ftype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            f.write(f"- {ftype}: {count}\n")
        f.write("\n")
        
        f.write("## Oracle Gap by Category\n")
        for cat, count in sorted(oracle_gap_counts.items(), key=lambda x: -x[1]):
            total = cat_counts[cat]
            f.write(f"- {cat}: {count}/{total} ({count/total*100:.1f}%)\n")
        f.write("\n")
        
        f.write("## Evidence State Distribution\n")
        for state, count in sorted(evidence_state_counts.items(), key=lambda x: -x[1]):
            f.write(f"- {state}: {count}\n")
        f.write("\n")
        
        f.write("## Affected Components\n")
        for comp, count in sorted(component_counts.items(), key=lambda x: -x[1]):
            f.write(f"- {comp}: {count}\n")
        f.write("\n")
        
        f.write("## Top Priority Targets\n")
        # Priority = failures where oracle_recall=False (retrieval issue) AND high count
        priority = [(cat, count) for cat, count in oracle_gap_counts.items()]
        priority.sort(key=lambda x: -x[1])
        for cat, count in priority[:5]:
            total = cat_counts[cat]
            f.write(f"- **{cat}**: {count}/{total} oracle gaps - fix retrieval first\n")
        f.write("\n")


def extract_failure_ids(failures: list[dict]) -> list[str]:
    """Extract question IDs from failures for --questions input."""
    return [f["question_id"] for f in failures]


def main():
    parser = argparse.ArgumentParser(description="LoCoMo Failure Miner")
    parser.add_argument("results", type=str, help="Path to results.json (aggregated or conv_*_results.json)")
    parser.add_argument("--output-dir", type=str, default="failure_analysis",
                        help="Output directory for reports and ledger")
    parser.add_argument("--classify", action="store_true", default=True,
                        help="Run failure classification (default: True)")
    parser.add_argument("--no-classify", action="store_false", dest="classify",
                        help="Skip failure classification")
    parser.add_argument("--oracle-gap", action="store_true", default=True,
                        help="Analyze oracle recall gaps (default: True)")
    parser.add_argument("--exclude-conv", type=str, default="3,7",
                        help="Comma-separated conversation indices to exclude from analysis (holdout)")
    parser.add_argument("--commit", type=str, default=None,
                        help="Git commit hash to associate with this analysis")
    parser.add_argument("--ledger-append", action="store_true",
                        help="Append to existing ledger instead of overwriting")
    
    args = parser.parse_args()
    
    results_path = Path(args.results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load holdout conversations
    holdout = {int(c) for c in args.exclude_conv.split(",") if c.strip()}
    
    # Load results
    data = load_results(results_path)
    all_failures = extract_failures(data)
    
    # Filter out holdout conversations
    failures = []
    for f in all_failures:
        qid = f.get("question_id", "")
        if qid.startswith("conv-"):
            try:
                conv_idx = int(qid.split("-")[1])
                if conv_idx not in holdout:
                    failures.append(f)
            except (IndexError, ValueError):
                failures.append(f)  # Keep if can't parse
        else:
            failures.append(f)
    
    print(f"Total failures: {len(all_failures)}")
    print(f"After holdout exclusion (Conv {sorted(holdout)}): {len(failures)}")
    
    if not failures:
        print("No failures to analyze!")
        return
    
    # Initialize classifier and adapter
    classifier = FailureClassifierV2()
    adapter = LoCoMoAdapter()
    
    # Classify and analyze each failure
    classifications = []
    retrievals = []
    ledger_entries = []
    
    for f in failures:
        # Classify
        if args.classify:
            cls = classify_failure(classifier, f, adapter)
        else:
            cls = {"category": "UNCLASSIFIED", "confidence": 0.0, "reasoning": ""}
        classifications.append(cls)
        
        # Analyze retrieval
        ret = analyze_retrieval(f, adapter)
        retrievals.append(ret)
        
        # Generate ledger entry
        entry = generate_ledger_entry(f, cls, ret, adapter, commit=args.commit)
        ledger_entries.append(entry)
    
    # Save failure IDs for re-running
    failure_ids = extract_failure_ids(failures)
    ids_path = output_dir / "failure_ids.json"
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(failure_ids, f, ensure_ascii=False, indent=2)
    print(f"Failure IDs saved to: {ids_path}")
    
    # Save failure questions for --questions input
    failure_questions = [
        {"question_id": f["question_id"], "category": f.get("category")}
        for f in failures
    ]
    questions_path = output_dir / "failure_questions.json"
    with open(questions_path, "w", encoding="utf-8") as f:
        json.dump(failure_questions, f, ensure_ascii=False, indent=2)
    print(f"Failure questions saved to: {questions_path}")
    
    # Write ledger
    ledger_path = output_dir / "failure_ledger.jsonl"
    write_ledger(ledger_entries, ledger_path, append=args.ledger_append)
    print(f"Failure Ledger saved to: {ledger_path}")
    
    # Generate report
    generate_report(failures, classifications, retrievals, adapter, output_dir)
    print(f"Failure Report saved to: {output_dir / 'failure_report.md'}")
    
    # Print summary
    print("\n=== FAILURE SUMMARY ===")
    cat_counts = defaultdict(int)
    for f in failures:
        cat = adapter.CATEGORY_NAMES.get(f.get("category", 0), f"cat-{f.get('category', 0)}")
        cat_counts[cat] += 1
    
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        oracle_gaps = sum(1 for f, r in zip(failures, retrievals) 
                          if adapter.CATEGORY_NAMES.get(f.get("category", 0), "") == cat 
                          and r.get("oracle_gap"))
        print(f"  {cat}: {count} failures, {oracle_gaps} oracle gaps")


if __name__ == "__main__":
    main()