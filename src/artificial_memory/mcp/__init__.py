"""MCP (Model Context Protocol) server for Artificial Memory.

Priority interface per V1 decision: MCP -> Python SDK -> REST.
All MCP tools pass through the ``ArtificialMemoryRuntime`` facade so no
interface-specific logic leaks into the cognitive core.

Requires the optional ``mcp`` dependency::

    pip install -e ".[mcp]"
    python -m artificial_memory.mcp
"""
