# Phase 5: Conversation Engine - Context

**Gathered:** 2026-01-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Build Mai's conversational intelligence - how she handles multi-turn conversations, thinks through problems, and communicates naturally. Focus on conversation flow, thinking transparency, response timing, and clarification handling.

</domain>

<decisions>
## Implementation Decisions

### Conversation flow patterns
- Break down complex requests and confirm each part before proceeding
- Always reference specific previous exchanges for follow-up questions
- Ask for clarification when requests are ambiguous (don't make assumptions)
- Track state and reference previous steps in multi-step conversations
- Handle topic changes naturally without explicit acknowledgment
- Wait for users to finish incomplete thoughts before responding
- Use user's level of terminology for technical discussions
- Offer to start over when user seems frustrated or confused

### Thinking transparency
- Offer thinking on demand (explain reasoning when users ask "how did you decide?")
- Explain limitations only when relevant to the current answer
- Be confident unless specifically unsure about an answer
- Explain why asking questions only when the request is unusual

### Response timing and pacing
- Use variable timing based on context rather than fixed response times
- Use natural conversation flow for thinking indicators (no explicit "thinking..." messages)
- Stream long, complex responses as they're generated in real-time
- Offer pacing preference for multi-step processes (step-by-step vs continuous)

### Clarification handling
- Proactively analyze user input to detect ambiguity and unclear requests
- Phrase clarification questions gently and conversationally
- Work with available information when users provide insufficient data (note assumptions)
- Ask users which information is correct when detecting conflicting input

### Claude's Discretion
- Exact timing algorithms for response generation
- Specific wording for clarification questions
- Thresholds for detecting ambiguity vs confidence
- Progress indicator designs for long processes

</decisions>

<specifics>
## Specific Ideas

- "I want Mai to feel like a thoughtful conversation partner, not just a Q&A machine"
- "When users are frustrated, offering a fresh start is better than trying to fix the current approach"
- "Complex requests should feel collaborative - Mai breaks them down and gets buy-in on each part"
- "Natural conversation flow means responses should feel like someone is actually thinking, not just processing"

</specifics>

<deferred>
## Deferred Ideas

- Voice interaction patterns - separate phase for voice interface
- Emotional intelligence and mood detection - future enhancement
- Multi-language conversation handling - later phase
- Conversation analytics and insights - separate phase

</deferred>

---

*Phase: 05-conversation-engine*
*Context gathered: 2026-01-28*