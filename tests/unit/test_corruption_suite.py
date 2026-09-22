"""Unit tests for Memory Corruption & Mutation Stress Suite (Phase 6-A)."""

import pytest

from artificial_memory.research.benchmarks.corruption_suite import MemoryCorruptionSuite


def test_corruption_suite_all_cases() -> None:
    """Verify all 5 corruption and mutation stress cases pass."""
    suite = MemoryCorruptionSuite()
    results = suite.run_all()

    assert len(results) == 5
    for r in results:
        assert r.passed, f"Failed {r.case_name}: {r.details}"
