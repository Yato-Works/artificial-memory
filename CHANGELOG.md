# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `vector` extra (faiss-cpu, sentence-transformers); numpy moved to core dependencies
- Lazy exports in `artificial_memory.memory` (heavy faiss/torch imports no longer
  load on light module imports; import time 19s -> 0.3s)
- Shared SentenceTransformer model cache (`memory.embeddings`), reused across engines
- Deterministic README benchmark harness (`scripts/run_readme_benchmark.py`)
- Ollama LLM QA benchmark, 3 conditions (no-memory / full-context / AM recall)
  (`scripts/run_llm_benchmark.py`)
- MCP smoke test covering all 7 tools over real stdio (`scripts/mcp_smoke_test.py`)
- Real-agent cross-session E2E: remember -> process exit -> recall / trace /
  timeline / explain / expand / inspect on a fresh process (`scripts/real_agent_e2e.py`)
- Agent tool-selection test: a local LLM autonomously chooses MCP tools
  (`scripts/agent_tool_selection_test.py`)
- Agent integration guide (`docs/mcp-agent-testing.md`) and E2E evidence report
  (`docs/real-agent-e2e.md`)
- Reproducible benchmark evidence in `benchmark_results/`

### Changed
- `psycopg[binary]` and `mcp>=1.0,<2.0` pinned (FastMCP moved out of mcp 2.x)
- CI installs vector/mcp extras; mypy step is continue-on-error while strict-mode
  adoption finishes (~27 remaining findings)
- Planning documents moved to `docs/planning/`

### Fixed
- MCP tool responses rendered nested pydantic `MemoryIR` as repr strings; replaced
  with a recursive JSON-safe serializer so agents can read `memory_id`s
- `memory_inspect` crashed inside the MCP event loop (`asyncio.run` on a running
  loop); the tool now awaits `trace()` directly
- PostgreSQL tests hung for minutes when no server was reachable; added a
  3-second connectivity pre-check and graceful skip
- SentenceTransformer was reloaded per engine instance; models are now shared via
  an LRU cache

### Earlier pre-release hardening
- Pre-commit configuration with ruff, mypy, isort, black
- GitHub issue templates (bug report, feature request) and PR template
- Enhanced pyproject.toml with metadata (classifiers, URLs, keywords)
- VectorSearchEngine protocol in core.interfaces
- ContextIRCompiler.optimize_ir method in RuleBasedIRCompiler
- Fixed all ruff linting issues (153+ errors resolved)
- Fixed majority of mypy type errors (940+ errors reduced to ~27)
- Improved RuntimeConfig with better secret handling warnings
- Moved ruff configuration to modern [tool.ruff.lint] section
- Research components now imported at runtime to avoid circular imports
- AdaptiveRecallEngine naming conflict with instance variable
- VectorSearchEngine protocol signature mismatch with implementations
- create_federated_memory_engine argument types
- RuleBasedIRCompiler untyped call in facade
- Research component constructor arguments

## [0.1.0] - 2024-01-XX

### Added
- Initial release of Artificial Memory / Context Runtime
- Core memory system with 6 resolution levels (RAW to DEEP_LONG_TERM)
- Progressive memory compression (Forget = Resolution Down)
- Adaptive-resolution recall with provenance tracking
- Topic-based organization with automatic classification
- Memory IR / Context IR formal definitions
- Vector search with FAISS and pgvector backends
- Temporal queries and time travel capabilities
- Memory integrity checking and self-healing
- Belief management with contradiction detection
- Federated multi-agent memory exchange
- Enterprise governance (tenancy, audit, retention, trust)
- Kubernetes operator with Helm chart
- MCP interface for AI agents
- CLI and REST API
- Experiment framework with ablation studies and red-team suite
- WebSocket support for real-time updates

### Infrastructure
- Docker and docker-compose support
- GitHub Actions CI/CD pipeline
- Pre-commit hooks configuration
- Comprehensive test suite (113 tests)
- Type checking with mypy (strict mode)
- Linting with ruff

---

## Release Process

1. Update version in `pyproject.toml`
2. Update this CHANGELOG.md
3. Create a git tag: `git tag v<version>`
4. Push tag: `git push origin v<version>`
5. GitHub Actions will build and publish release

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute.

## License

MIT License - see [LICENSE](LICENSE) for details.