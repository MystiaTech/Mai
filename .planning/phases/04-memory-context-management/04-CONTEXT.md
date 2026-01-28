# Phase 4: Memory & Context Management - Context

**Gathered:** 2026-01-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Build long-term conversation memory and context management system that stores conversation history locally, recalls past conversations efficiently, compresses memory as it grows, distills patterns into personality layers, and proactively surfaces relevant context. Focus on persistent storage that can scale efficiently while maintaining fast access to recent conversations and intelligent retrieval of relevant historical context.

</domain>

<decisions>
## Implementation Decisions

### Storage Format & Persistence Strategy
- Hybrid storage approach: SQLite for active/recent data, JSON archives for long-term storage
- Progressive compression strategy: 7 days/30 days/90 days compression tiers with target reduction ratios
- Smart retention policy: Value-based retention where important conversations (marked by user or high engagement) are kept longer, routine chats auto-archived
- Include memory in existing code/system backups: Conversation history becomes part of regular backup process

### Memory Retrieval & Recall System
- Hybrid semantic + keyword search: Start with semantic embeddings for meaning, fallback to keyword matching for precision
- Context-aware search (current topic): Prioritize conversations related to current discussion topic automatically
- Full timeline search with date range filters: Users can search entire history with date filters and conversation exclusion options
- Broad semantic concepts with conversation snippets: Find by meaning, show relevant conversation excerpts for immediate context

### Memory Compression & Summarization
- Progressive compression levels: Full conversation → key points → brief summary → metadata only approach for different access needs
- Hybrid extractive + abstractive summarization: Extract key quotes/facts, then generate abstract summary preserving important details while being concise
- Age-based compression triggers: Recent 30 days uncompressed for performance, older conversations compressed based on storage efficiency needs

### Pattern Learning & Personality Layer Extraction
- Multi-dimensional learning approach: Learn from topics, sentiment, interaction patterns, time-based preferences, and response styles to create weighted personality profile
- Hybrid with context switching: Mix of system prompt modifications and behavior configuration based on conversation context and importance
- Personality layers work as adaptive overlays that modify Mai's communication patterns while preserving core personality traits
- Cumulative learning where appropriate layers build on previous patterns while maintaining stability

### Claude's Discretion
- Exact compression ratios and timing for each tier
- Semantic embedding model selection and vector indexing approach
- Personality layer weighting algorithms and application thresholds
- Search ranking algorithms and relevance scoring methods
- Backup frequency and integration with existing backup systems

</decisions>

<specifics>
## Specific Ideas

- User wants smart retention that recognizes conversation importance automatically
- Hybrid storage balances performance (SQLite) with human readability (JSON)
- Progressive compression provides different access levels for different conversation ages
- Context-aware search should automatically surface relevant history during ongoing conversations
- Personality layers should be adaptive overlays that enhance rather than replace core personality

</specifics>

<deferred>
## Deferred Ideas

- Real-time conversation synchronization across multiple devices - future phase covering device sync
- Advanced emotion detection and sentiment analysis - potential Phase 9 personality system enhancement
- External integrations with calendar/task systems - future Phase 6 CLI interface consideration

</deferred>

---

*Phase: 04-memory-context-management*
*Context gathered: 2026-01-27*