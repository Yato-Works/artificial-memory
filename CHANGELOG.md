# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-09-15

### Added
- Initial release of Artificial Memory / Context Runtime
- Core memory system with 6 resolution levels (RAW to DEEP_LONG_TERM)
- Progressive memory compression (Forget = Resolution Down)
- Adaptive-resolution recall with provenance tracking
- Topic-based organization with automatic classification
- Memory IR / Context IR formal definitions
- Vector search with FAISS and pgvector backends (`vector` extra)
- Shared SentenceTransformer model cache and lazy module exports (import time reduced from 19s to 0.3s)
- Temporal queries, belief management, and contradiction detection
- Memory integrity checking and self-healing
- Federated multi-agent memory exchange and enterprise governance
- MCP interface for AI agents with 7 verified tools and real-agent cross-session E2E support
- CLI, WebSocket, and REST API
- Reproducible benchmark harness (deterministic recall, Ollama LLM QA, MCP smoke test)
- Docker, docker-compose, and Kubernetes Operator with Helm chart

### Changed
- Pinned `psycopg[binary]` and `mcp>=1.0,<2.0` for stability
- CI enhanced with matrix testing, vector/mcp extras, and non-blocking mypy strict adoption

### Fixed
- Fixed MCP tool response nested Pydantic `MemoryIR` JSON serialization
- Fixed `memory_inspect` event loop deadlock in MCP tools
- Fixed PostgreSQL test timeout with fast connectivity check and graceful skip
- Resolved all Ruff linting issues and majority of type annotations

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