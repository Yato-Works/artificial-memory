# Testing Artificial Memory with Real MCP Agents

Two levels of MCP verification exist in this repo:

1. **Automated smoke test** (`scripts/mcp_smoke_test.py`) — spawns the real
   MCP server as a subprocess and exercises all 7 tools end-to-end via the
   official MCP stdio client. CI-friendly:

   ```bash
   python scripts/mcp_smoke_test.py
   # -> MCP SMOKE TEST: PASS (all tools responded)
   ```

2. **Real-agent testing** — register the server in an actual MCP client
   (Cline, Hermes, Claude Desktop, ...) and drive it conversationally.
   This is what the automated test cannot replace: the agent's tool-selection
   behavior, multi-turn memory usage, and latency feel.

## Register with Cline (VS Code)

1. Build/install the package once:

   ```bash
   pip install -e ".[mcp,vector]"
   ```

2. Open the Cline MCP settings file:

   `%APPDATA%\Code\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json`

3. Add the server entry:

   ```json
   {
     "mcpServers": {
       "artificial-memory": {
         "command": "python",
         "args": ["-m", "artificial_memory.mcp"],
         "env": {},
         "disabled": false,
         "autoApprove": [
           "memory_recall",
           "memory_explain",
           "memory_timeline",
           "memory_trace",
           "memory_inspect"
         ]
       }
     }
   }
   ```

   `autoApprove` for read-only tools avoids permission prompts; keep
   `memory_remember` / `memory_expand` manually approved so writes stay intentional.

   By default the server stores memories in `memory.db` under the working
   directory where Cline launched it. To pin a specific location, point the
   command at a small bootstrap script that constructs `RuntimeConfig` with
   explicit paths (see `scripts/mcp_smoke_test.py` for a working example).

## Manual E2E checklist (any agent)

Run these as a *fresh conversation* per step to test real persistence:

| Step | Say / do | Expected |
|---|---|---|
| 1 | "Remember that we chose PostgreSQL for production because of multi-region support." | Agent calls `memory_remember`, reports success |
| 2 | "Remember that release 0.2.0 ships in November with the memory debugger." | Second memory stored |
| 3 | New session: "Which database did we pick for production, and why?" | Agent calls `memory_recall`, answers PostgreSQL + multi-region with confidence |
| 4 | "Show me where that memory came from." | Agent calls `memory_trace`, cites source conversation/timestamp |
| 5 | "What do we know about the 0.2.0 release timeline?" | `memory_recall` or `memory_timeline` returns the November entry |
| 6 | "Why would you recall these memories for that question?" | `memory_explain` returns mode/resolution/reasoning |

Pass criteria: steps 3–6 succeed **across sessions**, and provenance in step 4
traces back to the exact conversation from step 1.

## Notes

- The MCP server is stdio-based (`python -m artificial_memory.mcp`); HTTP
  transports are out of scope for V1.
- Requires `mcp>=1.0,<2.0` (the FastMCP API moved to a separate package in
  mcp 2.x — see `pyproject.toml`).
- If a tool returns an error, re-run the smoke test first; it isolates server
  issues from agent-integration issues.
