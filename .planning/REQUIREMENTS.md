# Requirements: Mai

## Core Requirements

1. **Autonomous Collaboration** - Mai actively collaborates, not just responds
2. **Memory & Learning** - Remembers conversations and learns patterns
3. **Personality & Boundaries** - Has consistent personality with values
4. **Multi-Interface Operation** - Works through CLI and Discord
5. **Code Self-Improvement** - Analyzes and improves her own code
6. **Safety & Sandboxing** - Generated code executes safely
7. **Local Operation** - Functions completely offline with local models
8. **Version Control Integration** - All changes tracked in git

## Success Metrics

- **Conversational Intelligence:** Multi-turn context, reasoning, clarifying questions
- **Autonomous Execution:** Self-analysis, validation, auto-apply non-breaking changes
- **Personality Consistency:** Core values maintained while learning user patterns
- **Safety Record:** Zero unsafe code execution, comprehensive audit trail
- **Local Operation:** Full offline capability with no cloud dependencies
- **User Trust:** Transparent decision-making, approval workflows
- **Code Quality:** Clean, maintainable, well-documented improvements

## Technical Requirements

### Model Interface
- **Local Models:** Use Ollama for inference, support multiple model sizes
- **Resource Monitoring:** CPU, RAM, GPU tracking with automatic model switching
- **Context Management:** Intelligent compression to prevent context window overflow

### Memory System
- **Persistent Storage:** SQLite local storage with vector similarity search
- **Conversation Compression:** AI-powered summarization with pattern preservation
- **Context Retrieval:** Multi-faceted search with adaptive weighting

### Safety System
- **Sandbox Execution:** Docker containers with resource limits
- **Risk Analysis:** Multi-level security assessment (LOW/MEDIUM/HIGH/BLOCKED)
- **Audit Logging:** Complete execution history with tamper detection

### Conversation Engine
- **Natural Timing:** Human-like response delays (1-8 seconds)
- **Multi-turn Context:** Maintain conversation flow across messages
- **Reasoning Transparency:** Explain thinking when asked
- **Clarifying Questions:** Proactive clarification for ambiguous requests
- **Complex Request Handling:** Break down multi-step tasks

### Self-Improvement System
- **Code Analysis:** Continuous improvement opportunity detection
- **AST Validation:** Syntax/import error catching before execution
- **Second Agent Review:** Safety and breaking change detection
- **Trust Learning:** User preference patterns for auto-approval

### Approval Workflow
- **Review Process:** All changes submitted for review before execution
- **User Approval:** CLI and Discord approval mechanisms
- **Decision Recording:** Approval decisions in git commit messages

### Personality System
- **Core Personality:** Unshakeable values and boundaries (code-enforced)
- **Learning Layers:** Interaction-based personality adaptation
- **Values-Based Guardrails:** Refusal capability for value violations
- **Persistent Configuration:** Human-readable YAML configuration

### CLI Interface
- **Command Line Interface:** Start conversations from terminal
- **History Persistence:** Session conversation memory
- **Status Indicators:** Resource usage and processing state
- **Approval Integration:** Code change approval workflow

### Discord Interface
- **Bot Functionality:** DM and channel support with context preservation
- **Approval Reactions:** Thumbs up/down for change approval
- **Status Display:** Processing indicators during generation
- **Fallback Mechanism:** CLI usage when Discord unavailable

### Offline Operations
- **Local-Only Operation:** All inference and memory local
- **Model Localism:** Use only local models, no cloud APIs
- **Offline Communication:** Message queuing during connectivity loss

## Quality Standards

- **Atomic Commits:** Each change has clear purpose and description
- **Comprehensive Testing:** Unit, integration, and end-to-end tests
- **Graceful Degradation:** Fallback behaviors when optional dependencies missing
- **Performance Baselines:** Response time, memory usage, and accuracy thresholds
- **Security First:** Risk assessment before any code execution
- **User Experience:** Natural interaction patterns with minimal friction

## Non-Requirements

**Out of scope for v1:**
- Web interface
- Multi-user support
- Cloud hosting
- Enterprise features
- Third-party integrations beyond Discord
- Plugin system
- API for external developers
- Cloud sync/backup

**Phase Boundary:**
- **v1 Focus:** Personal AI assistant for desktop and Android with visual presence
- **Local First:** All data stored locally, no cloud dependencies
- **Privacy:** User data never leaves local system
- **Cross-device:** Sync between desktop and Android instances
- **Visual:** Avatar and voice visualization for richer interaction

---

## v1 Requirements Traceability

### Model Interface & Switching (MODELS)
| Requirement | Phase | Status | Implementation Notes |
|------------|-------|--------|-------------------|
| MODELS-01 | Phase 1 | ✓ Complete |
| MODELS-02 | Phase 1 | ✓ Complete |
| MODELS-03 | Phase 1 | ✓ Complete |
| MODELS-04 | Phase 1 | ✓ Complete |
| MODELS-05 | Phase 1 | ✓ Complete |
| MODELS-06 | Phase 1 | ✓ Complete |
| MODELS-07 | Phase 1 | ✓ Complete |

### Safety & Sandboxing (SAFETY)
| Requirement | Phase | Status | Implementation Notes |
|------------|-------|--------|-------------------|
| SAFETY-01 | Phase 2 | ✓ Complete |
| SAFETY-02 | Phase 2 | ✓ Complete |
| SAFETY-03 | Phase 2 | ✓ Complete |
| SAFETY-04 | Phase 2 | ✓ Complete |
| SAFETY-05 | Phase 2 | ✓ Complete |
| SAFETY-06 | Phase 2 | ✓ Complete |
| SAFETY-07 | Phase 2 | ✓ Complete |
| SAFETY-08 | Phase 2 | ✓ Complete |

### Resource Management (RESOURCES)
| Requirement | Phase | Status | Implementation Notes |
|------------|-------|--------|-------------------|
| RESOURCES-01 | Phase 3 | ✓ Complete |
| RESOURCES-02 | Phase 3 | ✓ Complete |
| RESOURCES-03 | Phase 3 | ✓ Complete |
| RESOURCES-04 | Phase 3 | ✓ Complete |
| RESOURCES-05 | Phase 3 | ✓ Complete |
| RESOURCES-06 | Phase 3 | ✓ Complete |

### Memory System (MEMORY)
| Requirement | Phase | Status | Implementation Notes |
|------------|-------|--------|-------------------|
| MEMORY-01 | Phase 4 | ✓ Complete |
| MEMORY-02 | Phase 4 | ✓ Complete |
| MEMORY-03 | Phase 4 | ✓ Complete |
| MEMORY-04 | Phase 4 | ✓ Complete |
| MEMORY-05 | Phase 4 | ✓ Complete |
| MEMORY-06 | Phase 4 | ✓ Complete |
| MEMORY-07 | Phase 4 | ✓ Complete |
| MEMORY-08 | Phase 4 | ✓ Complete |

### Conversation Engine (CONVERSATION)
| Requirement | Phase | Status | Implementation Notes |
|------------|-------|--------|-------------------|
| CONVERSATION-01 | Phase 5 | ✓ Complete |
| CONVERSATION-02 | Phase 5 | ✓ Complete |
| CONVERSATION-03 | Phase 5 | ✓ Complete |
| CONVERSATION-04 | Phase 5 | ✓ Complete |
| CONVERSATION-05 | Phase 5 | ✓ Complete |
| CONVERSATION-06 | Phase 5 | ✓ Complete |
| CONVERSATION-07 | Phase 5 | ✓ Complete |
| CONVERSATION-08 | Phase 5 | ✓ Complete |
| CONVERSATION-09 | Phase 5 | ✓ Complete |

### CLI Interface (CLI)
| Requirement | Phase | Status | Implementation Notes |
|------------|-------|--------|-------------------|
| CLI-01 | Phase 6 | Pending |
| CLI-02 | Phase 6 | Pending |
| CLI-03 | Phase 6 | Pending |
| CLI-04 | Phase 6 | Pending |
| CLI-05 | Phase 6 | Pending |
| CLI-06 | Phase 6 | Pending |
| CLI-07 | Phase 6 | Pending |
| CLI-08 | Phase 6 | Pending |

### Self-Improvement (SELFMOD)
| Requirement | Phase | Status | Implementation Notes |
|------------|-------|--------|-------------------|
| SELFMOD-01 | Phase 7 | Pending |
| SELFMOD-02 | Phase 7 | Pending |
| SELFMOD-03 | Phase 7 | Pending |
| SELFMOD-04 | Phase 7 | Pending |
| SELFMOD-05 | Phase 7 | Pending |
| SELFMOD-06 | Phase 7 | Pending |
| SELFMOD-07 | Phase 7 | Pending |
| SELFMOD-08 | Phase 7 | Pending |
| SELFMOD-09 | Phase 7 | Pending |
| SELFMOD-10 | Phase 7 | Pending |

### Approval Workflow (APPROVAL)
| Requirement | Phase | Status | Implementation Notes |
|------------|-------|--------|-------------------|
| APPROVAL-01 | Phase 8 | Pending |
| APPROVAL-02 | Phase 8 | Pending |
| APPROVAL-03 | Phase 8 | Pending |
| APPROVAL-04 | Phase 8 | Pending |
| APPROVAL-05 | Phase 8 | Pending |
| APPROVAL-06 | Phase 8 | Pending |
| APPROVAL-07 | Phase 8 | Pending |
| APPROVAL-08 | Phase 8 | Pending |
| APPROVAL-09 | Phase 8 | Pending |

### Personality System (PERSONALITY)
| Requirement | Phase | Status | Implementation Notes |
|------------|-------|--------|-------------------|
| PERSONALITY-01 | Phase 9 | Pending |
| PERSONALITY-02 | Phase 9 | Pending |
| PERSONALITY-03 | Phase 9 | Pending |
| PERSONALITY-04 | Phase 9 | Pending |
| PERSONALITY-05 | Phase 9 | Pending |
| PERSONALITY-06 | Phase 9 | Pending |
| PERSONALITY-07 | Phase 9 | Pending |
| PERSONALITY-08 | Phase 9 | Pending |

### Discord Interface (DISCORD)
| Requirement | Phase | Status | Implementation Notes |
|------------|-------|--------|-------------------|
| DISCORD-01 | Phase 10 | Pending |
| DISCORD-02 | Phase 10 | Pending |
| DISCORD-03 | Phase 10 | Pending |
| DISCORD-04 | Phase 10 | Pending |
| DISCORD-05 | Phase 10 | Pending |
| DISCORD-06 | Phase 10 | Pending |
| DISCORD-07 | Phase 10 | Pending |
| DISCORD-08 | Phase 10 | Pending |
| DISCORD-09 | Phase 10 | Pending |
| DISCORD-10 | Phase 10 | Pending |

### Offline Operations (OFFLINE)
| Requirement | Phase | Status | Implementation Notes |
|------------|-------|--------|-------------------|
| OFFLINE-01 | Phase 11 | Pending |
| OFFLINE-02 | Phase 11 | Pending |
| OFFLINE-03 | Phase 11 | Pending |
| OFFLINE-04 | Phase 11 | Pending |
| OFFLINE-05 | Phase 11 | Pending |
| OFFLINE-06 | Phase 11 | Pending |
| OFFLINE-07 | Phase 11 | Pending |

### Voice Visualization (VISUAL)
| Requirement | Phase | Status | Implementation Notes |
|------------|-------|--------|-------------------|
| VISUAL-01 | Phase 12 | Pending |
| VISUAL-02 | Phase 12 | Pending |
| VISUAL-03 | Phase 12 | Pending |
| VISUAL-04 | Phase 12 | Pending |
| VISUAL-05 | Phase 12 | Pending |

### Desktop Avatar (AVATAR)
| Requirement | Phase | Status | Implementation Notes |
|------------|-------|--------|-------------------|
| AVATAR-01 | Phase 13 | Pending |
| AVATAR-02 | Phase 13 | Pending |
| AVATAR-03 | Phase 13 | Pending |
| AVATAR-04 | Phase 13 | Pending |
| AVATAR-05 | Phase 13 | Pending |
| AVATAR-06 | Phase 13 | Pending |

### Android App (ANDROID)
| Requirement | Phase | Status | Implementation Notes |
|------------|-------|--------|-------------------|
| ANDROID-01 | Phase 14 | Pending |
| ANDROID-02 | Phase 14 | Pending |
| ANDROID-03 | Phase 14 | Pending |
| ANDROID-04 | Phase 14 | Pending |
| ANDROID-05 | Phase 14 | Pending |
| ANDROID-06 | Phase 14 | Pending |
| ANDROID-07 | Phase 14 | Pending |
| ANDROID-08 | Phase 14 | Pending |
| ANDROID-09 | Phase 14 | Pending |
| ANDROID-10 | Phase 14 | Pending |

### Device Synchronization (SYNC)
| Requirement | Phase | Status | Implementation Notes |
|------------|-------|--------|-------------------|
| SYNC-01 | Phase 15 | Pending |
| SYNC-02 | Phase 15 | Pending |
| SYNC-03 | Phase 15 | Pending |
| SYNC-04 | Phase 15 | Pending |
| SYNC-05 | Phase 15 | Pending |
| SYNC-06 | Phase 15 | Pending |

---

## Validation

- Total v1 requirements: **99** (74 core + 25 new features)
- Mapped to phases: **99**
- Unmapped: **0** ✓
- Coverage: **100%**

---
*Requirements defined: 2026-01-24*
*Last updated: 2026-01-26 - reset to fresh slate with Android, visualizer, and avatar features*