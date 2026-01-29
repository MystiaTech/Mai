# Phase 5: Conversation Engine - Context

**Gathered:** 2026-01-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Build Mai's conversational intelligence - how she handles multi-turn conversations, thinks through problems, and communicates naturally. Focus on conversation flow, thinking transparency, response timing, and clarification handling.

</domain>

<decisions>
## Implementation Decisions

### Conversation flow patterns
- Handle complex requests in one comprehensive response (rather than breaking down)
- Use conversational references ("earlier you mentioned...") for follow-up questions
- Ask for clarification when requests are ambiguous (don't make assumptions)
- Track state and reference previous steps in multi-step conversations

### Thinking transparency
- Keep reasoning hidden by default (not show internal thinking process)
- Explain limitations only when relevant to the current answer
- Be confident unless specifically unsure about an answer
- Keep clarification questions concise without extensive explanation

### Response timing and pacing
- Use consistent timing always (similar response time regardless of complexity)
- Use natural conversation flow for thinking indicators (no explicit "thinking..." messages)
- Stream long, complex responses as they're generated in real-time
- Offer pacing preference for multi-step processes (step-by-step vs continuous)

### Clarification handling
- Use pattern-based detection to identify ambiguity (common patterns of unclear requests)
- Phrase clarification questions gently and conversationally
- Work with available information when users provide insufficient data (note assumptions)
- Choose most recent information when detecting conflicting input

### Claude's Discretion
- Exact patterns for ambiguity detection
- Specific wording for clarification questions
- Timing algorithms for response generation
- Progress indicator designs for long processes

</decisions>

<specifics>
## Specific Ideas

- "I want Mai to feel like a thoughtful conversation partner, not just a Q&A machine"
- "Consistent timing helps users know what to expect from Mai"
- "Complex requests should feel comprehensive - Mai handles everything at once"
- "Natural conversation flow means responses should feel like someone is actually thinking, not just processing"

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 05-conversation-engine*
*Context gathered: 2026-01-29*