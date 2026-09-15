# Contributing to Artificial Memory

Thank you for contributing to a **Cognitive Memory Runtime**! This document
explains how to set up, develop, test, and submit changes.

## Development Setup

```bash
# Clone & install (Python 3.11+)
git clone <repository-url>
cd artificial_memory
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/macOS
pip install -e ".[dev]"

# Optional extras
pip install -e ".[llm]"       # Ollama / OpenAI providers
pip install -e ".[mcp]"       # MCP server interface
pip install -e ".[nli]"       # DeBERTa-v3-base semantic verification (optional)
```

## Project Structure

```
src/artificial_memory/
├── compiler/      # Deterministic conversation -> MemoryIR pipeline (10 stages)
├── core/          # Domain models + Memory IR / Context IR (strict Pydantic)
├── memory/        # Evolution, belief engine, contradictions, healing
├── recall/        # Adaptive recall engines
├── context/       # Context IR builder & budget allocation
├── temporal/      # Time-travel & belief history
├── runtime/       # Public facade (all interfaces go through this)
├── storage/       # SQLite / PostgreSQL stores
├── mcp/           # MCP interface (priority: MCP -> SDK -> REST)
└── ...
```

## Ground Rules

1. **All new capabilities go through the Runtime Facade**
   (`src/artificial_memory/runtime/facade.py`). Do not expose internals via
   CLI/REST/MCP directly.
2. **Determinism matters.** Compiler stages must be pure functions. No
   randomness without an injectable, seedable source.
3. **Memory is never deleted** — status/resolution transitions only.
4. **Evidence ≠ Belief.** Beliefs are mutable interpretations; evidence
   (memories) is immutable and versioned.
5. **Every meaningful memory change persists an EvolutionEvent.**
6. **Version your algorithms.** Compression/recall changes must update
   `policy_version` / `algorithm_version` on `MemoryVersion`.

## Testing

```bash
pytest tests -x -q              # full suite
pytest tests/test_core.py -q    # fast subset
ruff check src tests            # lint
mypy src                        # type check (strict)
```

Tests must pass on Windows and Linux. Use `tempfile.TemporaryDirectory()`
context managers (not bare `NamedTemporaryFile`) to avoid Windows cleanup
flakiness.

## Pull Requests

1. Fork / branch from `main` (`feat/<topic>` or `fix/<topic>`).
2. Add tests for any behavior change (unit + integration where relevant).
3. Run the full test suite, ruff, and mypy.
4. Update documentation (README / docstrings) for public API changes.
5. Keep PRs focused; one logical change per PR.

## Commit Messages

Use conventional prefixes: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`,
`chore:`.

## Code of Conduct

By participating you agree to the
[Contributor Covenant](CODE_OF_CONDUCT.md).
