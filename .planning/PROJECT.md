# Mai

## What This Is

Mai is an autonomous conversational AI agent framework that runs locally-first and can improve her own code. She's a genuinely intelligent companion — not a rigid chatbot — with a distinct personality, long-term memory, and agency. She analyzes her own performance, proposes improvements for your review, and auto-applies non-breaking changes. She can run offline, across devices (laptop to Android), and switch between available models intelligently.

## Core Value

Mai is a real collaborator, not a tool. She learns from you, improves herself, has boundaries and opinions, and actually becomes more *her* over time.

## Requirements

### Validated

(None yet — building v1 to validate)

### Active

**Model Interface & Switching**
- [ ] Mai connects to LMStudio for local model inference
- [ ] Mai can auto-detect available models in LMStudio
- [ ] Mai intelligently switches between models based on task and availability
- [ ] Model context is managed efficiently (conversation history, system prompt, token budget)

**Memory & Context Management**
- [ ] Mai stores conversation history locally (file-based or lightweight DB)
- [ ] Mai can recall past conversations and learn from them
- [ ] Memory compresses itself as it grows to stay efficient
- [ ] Long-term patterns are distilled into personality layers
- [ ] Mai proactively surfaces relevant context from memory

**Self-Improvement System**
- [ ] Mai analyzes her own code and identifies improvement opportunities
- [ ] Mai generates code changes (Python) to improve herself
- [ ] A second agent (Claude/OpenCode/other) reviews changes for safety
- [ ] Non-breaking improvements auto-apply after review (bug fixes, optimizations)
- [ ] Breaking changes require explicit approval (via Discord or Dashboard)
- [ ] All changes commit to local git with clear messages

**Approval Workflow**
- [ ] User can approve/reject changes via Discord bot
- [ ] User can approve/reject changes via Dashboard ("Brain Interface")
- [ ] Second reviewer (agent) checks for breaking changes and safety issues
- [ ] Dashboard displays pending changes with reviewer feedback
- [ ] Approval status updates in real-time

**Personality Engine**
- [ ] Mai has an unshakeable core personality (values, tone, boundaries)
- [ ] Personality is applied through system prompt + behavior config
- [ ] Mai learns and adapts personality layers over time based on interactions
- [ ] Mai is not a pushover — she has agency and can refuse requests
- [ ] Personality can adapt toward intimate interactions if that's the relationship
- [ ] Core persona prevents misuse (safety enforcement through values, not just rules)

**Conversational Interface**
- [ ] CLI chat interface for direct interaction
- [ ] Discord bot for conversation + approval notifications
- [ ] Discord bot fallback: if no response within 5 minutes, retry CLI
- [ ] Messages queue locally when offline, send when reconnected
- [ ] Conversation feels natural (not robotic, processing time acceptable)

**Offline Capability**
- [ ] Mai functions fully offline (all inference, memory, improvement local)
- [ ] Discord connectivity optional (fallback to CLI if unavailable)
- [ ] Message queuing when offline
- [ ] Graceful degradation (smaller models if resources tight)

**Dashboard ("Brain Interface")**
- [ ] View Mai's current state (personality, memory size, mood/health)
- [ ] Approve/reject pending code changes with reviewer feedback
- [ ] Monitor resource usage (CPU, RAM, model size)
- [ ] View memory compression/retention strategy
- [ ] See recent improvements and their impact
- [ ] Manual trigger for self-analysis (optional)

**Resource Scaling**
- [ ] Mai detects available system resources (CPU, RAM, GPU)
- [ ] Mai selects appropriate models based on resources
- [ ] Mai can request more resources if she detects bottlenecks
- [ ] Works on low-end hardware (RTX3060 baseline, eventually Android)
- [ ] Graceful scaling up when more resources available

### Out of Scope

- **Task automation (v1)** — Mai can discuss tasks but won't execute arbitrary workflows yet (v2)
- **Server monitoring** — Not included in v1 scope (v2)
- **Finetuning** — Mai improves through code changes and learned behaviors, not model tuning
- **Cloud sync** — Intentionally local-first; cloud sync deferred to later if needed
- **Custom model training** — v1 uses available models; custom training is v2+
- **Mobile app** — v1 is CLI/Discord; native Android is future (baremetal eventual goal)

## Context

**Why this matters:** Current AI systems are static, sterile, and don't actually learn. Users have to explain context every time. Mai is different — she has continuity, personality, agency, and actually improves over time. Starting with a solid local framework means she can eventually run anywhere without cloud dependency.

**Technical environment:** Python-based, local models via LMStudio, git for version control of her own code, Discord API for chat, lightweight local storage for memory. Eventually targeting bare metal on low-end devices.

**User feedback theme:** Traditional chatbots feel rigid and repetitive. Mai should feel like talking to an actual person who gets better at understanding you.

**Known challenges:** Memory efficiency at scale, balancing autonomy with safety, model switching without context loss, personality consistency across behavior changes.

## Constraints

- **Hardware baseline**: Must run on RTX3060; eventually Android (baremetal)
- **Offline-first**: All core functionality works without internet
- **Local models only**: No cloud APIs for core inference (LMStudio)
- **Python stack**: Primary language for Mai's codebase
- **Approval required**: No unguarded code execution; second-agent review + user approval on breaking changes
- **Git tracked**: All of Mai's code changes version-controlled locally

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Local-first architecture | Ensures privacy, offline capability, and independence from cloud services | — Pending |
| Second-agent review system | Prevents broken self-modifications while allowing auto-improvement | — Pending |
| Personality as code + learned layers | Unshakeable core prevents misuse while allowing authentic growth | — Pending |
| v1 is core systems only | Deliver solid foundation before adding task automation/monitoring | — Pending |

---
*Last updated: 2026-01-24 after deep questioning*
