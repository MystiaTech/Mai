# Available Tools & MCP Integration

This document lists all available tools and MCP (Model Context Protocol) servers that Mai development can leverage.

## Hugging Face Hub Integration

**Status**: Authenticated as `mystiatech`

### Tools Available

#### Model Discovery
- `mcp__claude_ai_Hugging_Face__model_search` — Search ML models by task, author, library, trending
- `mcp__claude_ai_Hugging_Face__hub_repo_details` — Get detailed info on any model, dataset, or space

**Use Cases:**
- Phase 1: Discover quantized models for local inference (Mistral, Llama, etc.)
- Phase 12: Find audio/voice models for visualization
- Phase 13: Find avatar/animation models (VRoid compatible options)
- Phase 14: Research Android-compatible model formats

#### Dataset Discovery
- `mcp__claude_ai_Hugging_Face__dataset_search` — Find datasets by task, author, tags, trending
- Search filters: language, size, task categories

**Use Cases:**
- Phase 4: Training data research for memory compression
- Phase 5: Conversation quality datasets
- Phase 12: Audio visualization datasets

#### Research Papers
- `mcp__claude_ai_Hugging_Face__paper_search` — Search ML research papers with abstracts

**Use Cases:**
- Phase 2: Safety and sandboxing research papers
- Phase 4: Memory system and RAG papers
- Phase 5: Conversational AI and reasoning papers
- Phase 7: Self-improvement and code generation papers

#### Spaces & Interactive Models
- `mcp__claude_ai_Hugging_Face__space_search` — Discover Hugging Face Spaces (demos)
- `mcp__claude_ai_Hugging_Face__dynamic_space` — Run interactive tasks (Image Gen, OCR, TTS, etc.)

**Use Cases:**
- Phase 12: Voice/audio visualization demos
- Phase 13: Avatar generation or manipulation
- Phase 14: Android UI pattern research

#### Documentation
- `mcp__claude_ai_Hugging_Face__hf_doc_search` — Search HF docs and guides
- `mcp__claude_ai_Hugging_Face__hf_doc_fetch` — Fetch full documentation pages

**Use Cases:**
- Phase 1: LMStudio/Ollama integration documentation
- Phase 5: Transformers library best practices
- Phase 14: Mobile inference frameworks (ONNX Runtime, TensorFlow Lite)

#### Account Info
- `mcp__claude_ai_Hugging_Face__hf_whoami` — Get authenticated user info

## Web Research

### Tools Available
- `WebSearch` — Search the web for current information (2026 context)
- `WebFetch` — Fetch and analyze specific URLs

**Use Cases:**
- Research current best practices in AI safety (Phase 2)
- Find Android development patterns (Phase 14)
- Discover voice visualization libraries (Phase 12)
- Research avatar systems (Phase 13)
- Find Discord bot best practices (Phase 10)

## Code & Repository Tools

### Tools Available
- `Bash` — Execute terminal commands (git, npm, python, etc.)
- `Glob` — Fast file pattern matching
- `Grep` — Ripgrep-based content search
- `Read` — Read file contents
- `Edit` — Edit files with string replacement
- `Write` — Create new files

**Use Cases:**
- All phases: Create and manage project structure
- All phases: Execute tests and build commands
- All phases: Manage git commits and history

## Claude Code (GSD) Workflow

### Orchestrators Available
- `/gsd:new-project` — Initialize project
- `/gsd:plan-phase N` — Create detailed phase plans
- `/gsd:execute-phase N` — Execute phase with atomic commits
- `/gsd:discuss-phase N` — Gather phase context
- `/gsd:verify-work` — User acceptance testing

### Specialized Agents
- `gsd-project-researcher` — Domain research (stack, features, architecture, pitfalls)
- `gsd-phase-researcher` — Phase-specific research
- `gsd-codebase-mapper` — Analyze and document existing code
- `gsd-planner` — Create executable phase plans
- `gsd-executor` — Execute plans with state management
- `gsd-verifier` — Verify deliverables match requirements
- `gsd-debugger` — Systematic debugging with checkpoints

## How to Use MCPs in Development

### In Phase Planning
When creating `/gsd:plan-phase N`:
- Researchers can use Hugging Face tools to discover libraries and models
- Use WebSearch for current best practices
- Query papers for architectural patterns

### In Phase Execution
When running `/gsd:execute-phase N`:
- Download models from Hugging Face
- Use WebFetch for documentation
- Run Spaces for prototyping UI patterns

### Example Usage by Phase

**Phase 1: Model Interface**
```
- mcp__claude_ai_Hugging_Face__model_search
  Query: "quantized models for local inference"
  → Find Mistral, Llama, TinyLlama options
  
- mcp__claude_ai_Hugging_Face__hf_doc_fetch
  → Get Hugging Face Transformers documentation
  
- WebSearch
  → Latest LMStudio/Ollama integration patterns
```

**Phase 2: Safety System**
```
- mcp__claude_ai_Hugging_Face__paper_search
  Query: "code sandboxing, safety verification"
  → Find relevant research papers
  
- WebSearch
  → Docker security best practices
```

**Phase 5: Conversation Engine**
```
- mcp__claude_ai_Hugging_Face__dataset_search
  Query: "conversation quality, multi-turn dialogue"
  
- mcp__claude_ai_Hugging_Face__paper_search
  Query: "conversational AI, context management"
```

**Phase 12: Voice Visualization**
```
- mcp__claude_ai_Hugging_Face__space_search
  Query: "audio visualization, waveform display"
  → Find working demos
  
- mcp__claude_ai_Hugging_Face__model_search
  Query: "speech recognition, audio models"
```

**Phase 13: Desktop Avatar**
```
- mcp__claude_ai_Hugging_Face__space_search
  Query: "avatar generation, VRoid, character animation"
  
- WebSearch
  → VRoid SDK documentation
  → Avatar animation libraries
```

**Phase 14: Android App**
```
- mcp__claude_ai_Hugging_Face__model_search
  Query: "mobile inference, quantized models, ONNX"
  
- WebSearch
  → Kotlin ML Kit documentation
  → TensorFlow Lite best practices
```

## Configuration

Add to `.planning/config.json` to enable MCP usage:

```json
{
  "mcp": {
    "huggingface": {
      "enabled": true,
      "authenticated_user": "mystiatech",
      "default_result_limit": 10
    },
    "web_search": {
      "enabled": true,
      "domain_restrictions": []
    },
    "code_tools": {
      "enabled": true
    }
  }
}
```

## Research Output Format

When researchers use MCPs, they produce:
- `.planning/research/STACK.md` — Technologies and libraries
- `.planning/research/FEATURES.md` — Capabilities and patterns
- `.planning/research/ARCHITECTURE.md` — System design patterns
- `.planning/research/PITFALLS.md` — Common mistakes and solutions

These inform phase planning and implementation.

---

**Updated: 2026-01-26**
**Next Review: When new MCP servers become available**
