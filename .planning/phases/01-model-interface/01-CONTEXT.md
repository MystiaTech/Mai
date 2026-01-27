# Phase 01: Model Interface & Switching - Context

**Gathered:** 2026-01-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Connect to LMStudio for local model inference, auto-detect available models, intelligently switch between models based on task and availability, and manage model context efficiently (conversation history, system prompt, token budget).

</domain>

<decisions>
## Implementation Decisions

### Model Selection Strategy
- Primary factor: Available resources (CPU, RAM, GPU)
- Preference: Most efficient model that fits constraints
- Categorize models by both capability tier AND resource needs
- Fallback: Try minimal model even if slow when no model fits constraints

### Context Management Policy
- Trigger compression at 70% of context window
- Use hybrid approach: summarize very old messages, keep some middle ones intact, preserve all recent messages
- Priority during compression: Always preserve user instructions and explicit requests
- Adapts to different model context sizes based on percentage

### Switching Behavior
- Silent switching: No user notifications when changing models
- Dynamic switching: Can switch mid-task if current model struggles
- Smart context transfer: Send context relevant to why switching occurred
- Queue new tasks: Prepare new model in background, use for next message

### Failure Handling
- Auto-start LM Studio if not running
- Try next best model automatically if model fails to load
- Switch and retry immediately if model gives no response or errors
- Graceful degradation: Switch to minimal resource usage mode when exhausted

### Claude's Discretion
- Exact model capability tier definitions
- Context compression algorithms and thresholds within hybrid approach
- What constitutes "struggling" for dynamic switching
- Graceful degradation specifics (which features to disable)

</decisions>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches for local model management.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-model-interface*
*Context gathered: 2026-01-27*