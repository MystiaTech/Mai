# Mai Project Roadmap

## Overview

Mai's development is organized into three major milestones, each delivering distinct capabilities while building toward the full vision of an autonomous, self-improving AI agent.

---

## v1.0 Core - Foundation Systems
**Goal:** Establish core AI agent infrastructure with local model support, safety guardrails, and conversational foundation.

### Phase 1: Model Interface & Switching
- Connect to LMStudio for local model inference
- Auto-detect available models in LMStudio
- Intelligently switch between models based on task and availability
- Manage model context efficiently (conversation history, system prompt, token budget)

**Plans:** 3 plans in 2 waves
- [x] 01-01-PLAN.md — LM Studio connectivity and resource monitoring foundation
- [x] 01-02-PLAN.md — Conversation context management and memory system  
- [x] 01-03-PLAN.md — Intelligent model switching integration

### Phase 2: Safety & Sandboxing
- Implement sandbox execution environment for generated code
- Multi-level security assessment (LOW/MEDIUM/HIGH/BLOCKED)
- Audit logging with tamper detection
- Resource-limited container execution

**Plans:** 4 plans in 3 waves
- [x] 02-01-PLAN.md — Security assessment infrastructure (Bandit + Semgrep)
- [x] 02-02-PLAN.md — Docker sandbox execution environment
- [x] 02-03-PLAN.md — Tamper-proof audit logging system
- [x] 02-04-PLAN.md — Safety system integration and testing

### Phase 3: Resource Management
- Detect available system resources (CPU, RAM, GPU)
- Select appropriate models based on resources
- Request more resources when bottlenecks detected
- Graceful scaling from low-end hardware to high-end systems

**Plans:** 4 plans in 2 waves
- [x] 03-01-PLAN.md — Enhanced GPU detection with pynvml support
- [x] 03-02-PLAN.md — Hardware tier detection and management system
- [x] 03-03-PLAN.md — Proactive scaling with hybrid monitoring
- [x] 03-04-PLAN.md — Personality-driven resource communication

### Phase 4: Memory & Context Management
- Store conversation history locally (file-based or lightweight DB)
- Recall past conversations and learn from them
- Compress memory as it grows to stay efficient
- Distill long-term patterns into personality layers
- Proactively surface relevant context from memory

**Plans:** 4 plans in 3 waves
- [x] 04-01-PLAN.md — Storage foundation with SQLite and sqlite-vec
- [x] 04-02-PLAN.md — Semantic search and context-aware retrieval
- [x] 04-03-PLAN.md — Progressive compression and JSON archival
- [x] 04-04-PLAN.md — Personality learning and adaptive layers

### Phase 5: Conversation Engine
- Multi-turn context preservation
- Reasoning transparency and clarifying questions
- Complex request handling with task breakdown
- Natural timing and human-like response patterns

**Milestone v1.0 Complete:** Mai has a working local foundation with models, safety, memory, and natural conversation.

---

## v1.1 Interfaces & Intelligence
**Goal:** Add interaction interfaces and self-improvement capabilities to enable Mai to improve her own code.

### Phase 6: CLI Interface
- Command-line interface for direct terminal interaction
- Session history persistence
- Resource usage and processing state indicators
- Approval integration for code changes

### Phase 7: Self-Improvement System
- Analyze own code to identify improvement opportunities
- Generate code changes (Python) to improve herself
- AST validation for syntax/import errors
- Second-agent review for safety and breaking changes
- Auto-apply non-breaking improvements after review

### Phase 8: Approval Workflow
- User approval via CLI and Dashboard
- Second reviewer (agent) checks for breaking changes
- Dashboard displays pending changes with reviewer feedback
- Real-time approval status updates

### Phase 9: Personality System
- Unshakeable core personality (values, tone, boundaries)
- Personality applied through system prompt + behavior config
- Learn and adapt personality layers based on interactions
- Agency and refusal capabilities for value violations
- Values-based guardrails to prevent misuse

### Phase 10: Discord Interface
- Discord bot for conversation and approval notifications
- Direct message and channel support with context preservation
- Approval reactions (thumbs up/down for changes)
- Fallback to CLI when Discord unavailable
- Retry mechanism if no response within 5 minutes

**Milestone v1.1 Complete:** Mai can improve herself safely with human oversight and communicate through Discord.

---

## v1.2 Presence & Mobile
**Goal:** Add visual presence, voice capabilities, and native mobile support for rich cross-device experience.

### Phase 11: Offline Operations
- Full offline functionality (all inference, memory, improvement local)
- Discord connectivity optional with graceful degradation
- Message queuing when offline, send when reconnected
- Smaller models available for tight resource scenarios

### Phase 12: Voice Visualization
- Real-time visualization of audio input during voice conversations
- Low-latency waveform/frequency display
- Visual feedback for speech detection and processing
- Works on both desktop and Android

### Phase 13: Desktop Avatar
- Visual representation using static image or VRoid model
- Avatar expressions respond to conversation context (mood/state)
- Efficient rendering on RTX3060 and mobile devices
- Customizable appearance (multiple models or user-provided image)

### Phase 14: Android App
- Native Android app with local model inference
- Standalone operation (works without desktop instance)
- Voice input/output with low-latency processing
- Avatar and visualizer integrated in mobile UI
- Efficient resource management for battery and CPU

### Phase 15: Device Synchronization
- Sync conversation history and memory with desktop
- Synchronized state without server dependency
- Conflict resolution for divergent changes
- Efficient delta-based sync protocol

**Milestone v1.1 Complete:** Mai has visual presence and works seamlessly across desktop and Android devices.

---

## Phase Dependencies & Execution Path

```
v1.0 Core (Phases 1-5)
  ↓
v1.1 Interfaces (Phases 6-10)
  ├─ Parallel: Phase 6 (CLI), Phase 7-8 (Self-Improvement), Phase 9 (Personality)
  └─ Then: Phase 10 (Discord)
  ↓
v1.2 Presence (Phases 11-15)
  ├─ Parallel: Phase 11 (Offline), Phase 12 (Voice Viz)
  ├─ Then: Phase 13 (Avatar)
  ├─ Then: Phase 14 (Android)
  └─ Finally: Phase 15 (Sync)
```

---

## Success Criteria by Milestone

### v1.0 Core ✓
- [x] Local models working via LMStudio
- [x] Sandbox for safe code execution
- [x] Memory persists and retrieves correctly
- [x] Natural conversation flow maintained
- [ ] **Next:** Move to v1.1

### v1.1 Interfaces
- [ ] CLI interface fully functional
- [ ] Self-improvement system generates valid changes
- [ ] Second-agent review prevents unsafe changes
- [ ] Discord bot responds to commands and approvals
- [ ] Personality system maintains core values
- [ ] **Next:** Move to v1.2

### v1.2 Presence
- [ ] Full offline operation validated
- [ ] Voice visualization renders in real-time
- [ ] Avatar responds appropriately to conversation
- [ ] Android app syncs with desktop
- [ ] All features work on mobile
- [ ] **Release:** v1.0 complete

---

## Constraints & Considerations

- **Hardware baseline**: Must run on RTX3060 (desktop) and modern Android devices (2022+)
- **Offline-first**: All core functionality works without internet
- **Local models only**: No cloud APIs for core inference
- **Safety critical**: Second-agent review on all changes
- **Git tracked**: All modifications version-controlled
- **Python venv**: All dependencies in `.venv`

---

## Key Metrics

- **Total Requirements**: 99 (mapped across 15 phases)
- **Core Infrastructure**: Phases 1-5
- **Interface & Intelligence**: Phases 6-10
- **Visual & Mobile**: Phases 11-15
- **Coverage**: 100% of requirements

---

*Roadmap created: 2026-01-26*
*Based on fresh planning with Android, visualizer, and avatar features*
