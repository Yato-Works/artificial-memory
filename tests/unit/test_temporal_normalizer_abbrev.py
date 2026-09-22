"""Guards for the deterministic temporal normalizer (abbreviation support).

LoCoMo chat turns frequently abbreviate weekdays ("Last Fri", "last Tues").
Rule 10 of TemporalNormalizer must resolve those to a parenthesized absolute
expression so the answer model can copy calendar facts without arithmetic.
"""

from artificial_memory.context.temporal_normalizer import TemporalNormalizer


def test_last_day_abbreviation_gets_absolute_parenthetical() -> None:
    nz = TemporalNormalizer()
    out = nz.normalize("Last Fri I finally took my kids to a pottery workshop.", "1:51 pm on 15 July, 2023")
    assert "(the Friday before 15 July 2023)" in out
    assert "last Friday (the Friday before 15 July 2023)" in out


def test_last_day_full_name_behavior_unchanged() -> None:
    nz = TemporalNormalizer()
    out = nz.normalize("last Tuesday we went hiking", "1:51 pm on 15 July, 2023")
    assert "last Tuesday (the Tuesday before 15 July 2023)" in out


def test_abbreviations_resolve_to_correct_full_day() -> None:
    nz = TemporalNormalizer()
    assert "the Saturday before" in nz.normalize("last Sat", "2023-05-25")
    assert "the Wednesday before" in nz.normalize("last weds", "2023-05-25")
    assert "the Thursday before" in nz.normalize("last Thurs", "2023-05-25")
