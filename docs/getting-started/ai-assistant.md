# Petrophysical AI Assistant & MCP Server

The `quick_pp` backend includes a built-in **Petrophysical AI Assistant** and **Model Context Protocol (MCP) Server** (`quick_pp/app/backend/mcp_server.py` and `quick_pp/app/backend/services/chat.py`) for interactive subsurface evaluation.

## Features

- **Zero-API-Key Built-in Agent (`native_mcp`)**: Runs deterministic and conformal petrophysical workflows directly against project wells without requiring external API keys.
- **Multi-Provider LLM Support**: Switch between `native_mcp`, `openai`, `gemini`, `anthropic`, and local `ollama` directly from the workbench AI Assistant drawer.
- **In-Process Petrophysical MCP Tools**:
  - `inspect_well`: Inspect available curves, depth intervals, formation tops, fluid contacts, core plug counts, and zonal statistics.
  - `run_well_qaqc`: Execute outer-limit checks, washout/badhole flagging, coal detection, and NPHI/RHOB hydrocarbon corrections.
  - `run_multi_mineral`: Run `MultiMineral` or `SandSiltClay` lithology and porosity solvers.
  - `run_shf_evaluation`: Run FZI rock typing, Leverett-J / Skelt-Harrison / Cuddy Saturation Height Function (`SHF`), and Free Water Level (`FWL`) optimization.
  - `evaluate_with_conformal_bounds`: Generate calibrated P10–P90 conformal prediction intervals (`PHIT_P10`/`P90`, `SW_P10`/`P90`, `PERM_P10`/`P90`).
  - `control_workbench_ui`: Navigate workbench tabs, filter zones/depths, or scroll the log track canvas to a target depth.

## API Endpoints

- `GET /quick_pp/assistant/providers`: List supported LLM providers and models
- `POST /quick_pp/assistant/chat`: Execute an agentic chat turn with live tool calls and computed curves
- `GET /quick_pp/assistant/mcp/tools`: List registered Petrophysical MCP tools and JSON schemas
- `POST /quick_pp/assistant/mcp/call`: Directly invoke a Petrophysical MCP tool
- `POST /quick_pp/assistant/mcp`: Standard MCP JSON-RPC 2.0 Streamable HTTP endpoint

