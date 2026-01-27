# Codebase Concerns

**Analysis Date:** 2026-01-26

## Tech Debt

**Incomplete Memory System Integration:**
- Issue: Memory manager gracefully initializes but may fail silently when dependencies are missing
- Files: `src/mai/memory/manager.py`
- Impact: Memory features degrade ungracefully; users don't know compression or retrieval is disabled
- Fix approach: Add explicit logging and health checks on startup, expose memory system status in CLI

**Large Monolithic Memory Manager:**
- Issue: MemoryManager is 1036 lines with multiple responsibilities (storage, compression, retrieval orchestration)
- Files: `src/mai/memory/manager.py`
- Impact: Difficult to test individual memory subsystems; changes affect multiple concerns simultaneously
- Fix approach: Extract retrieval delegation and compression orchestration into separate coordinator classes

**Conversation Engine Complexity:**
- Issue: ConversationEngine is 648 lines handling timing, state, decomposition, reasoning, interruption, and metrics
- Files: `src/mai/conversation/engine.py`
- Impact: High cognitive load for maintainers; hard to isolate bugs in specific subsystems
- Fix approach: Separate concerns into focused orchestrator (engine) and behavior modules (timing/reasoning/decomposition are already separated but loosely coupled)

**Permission/Approval System Fragility:**
- Issue: ApprovalSystem uses regex pattern matching for risk analysis with hardcoded patterns
- Files: `src/mai/sandbox/approval_system.py`
- Impact: Pattern-matching approach is fragile (false positives/negatives); patterns not maintainable as code evolves
- Fix approach: Replace regex with AST-based code analysis for more reliable risk detection; move risk patterns to configuration

**Docker Executor Dependency Chain:**
- Issue: DockerExecutor falls back silently to unavailable state if Docker isn't installed
- Files: `src/mai/sandbox/docker_executor.py`
- Impact: Approval system thinks code is sandboxed when Docker is missing; security false sense of safety
- Fix approach: Require explicit Docker availability check at startup; block code execution if Docker unavailable and user requests sandboxing

## Known Bugs

**Session Persistence Restoration:**
- Symptoms: "ConversationState object has no attribute 'set_conversation_history'" error when restarting CLI
- Files: `src/mai/conversation/state.py`, `src/app/__main__.py`
- Trigger: Start conversation, exit CLI, restart CLI session
- Workaround: None - session restoration broken; users lose conversation history
- Status: Identified in Phase 6 UAT but remediation code not applied (commit c70ee88 "Complete fresh slate" removed implementation)

**Session File Feedback Missing:**
- Symptoms: Users don't see where/when session files are created
- Files: `src/app/__main__.py`
- Trigger: Create new session or use /session command
- Workaround: Manually check ~/.mai/session.json directory
- Status: Identified in Phase 6 UAT as major issue (test 3 failed)

**Resource Display Color Coding:**
- Symptoms: Resource monitoring displays plain text instead of color-coded status indicators
- Files: `src/app/__main__.py`
- Trigger: Run CLI and observe resource display during conversation
- Workaround: Parse output manually to understand resource status
- Status: Identified in Phase 6 UAT as minor issue (test 5 failed); root cause: Rich console loses color output in non-terminal environments

## Security Considerations

**Approval System Risk Analysis Insufficient:**
- Risk: Regex-based risk detection can be bypassed with obfuscated code (e.g., string concatenation to build dangerous commands)
- Files: `src/mai/sandbox/approval_system.py`
- Current mitigation: Hardcoded high-risk patterns (os.system, exec, eval); fallback to block on unrecognized patterns
- Recommendations:
  - Implement AST-based code analysis for more reliable detection
  - Add code deobfuscation step before risk analysis
  - Create risk assessment database with test cases and known bypasses
  - Require explicit docker verification before allowing code execution

**Docker Fallback Security Gap:**
- Risk: Code could execute without actual sandboxing if Docker unavailable, creating false sense of security
- Files: `src/mai/sandbox/docker_executor.py`
- Current mitigation: AuditLogger records all execution; approval system presents requests regardless
- Recommendations:
  - Fail-safe: Block code execution if Docker unavailable and user hasn't explicitly allowed non-sandboxed execution
  - Add warning dialog explaining sandbox unavailability
  - Log all non-sandboxed execution attempts explicitly
  - Require explicit override from user with confirmation

**Approval Preference Learning Risk:**
- Risk: User can set "auto_allow" on risky code patterns; once learned, code execution auto-approves without user intervention
- Files: `src/mai/sandbox/approval_system.py` (lines with `user_preferences` and `auto_allow`)
- Current mitigation: Auto-allow only applies to LOW risk level code
- Recommendations:
  - Require explicit user confirmation before enabling auto-allow (not just responding "a")
  - Log all auto-approved executions in audit trail with reason
  - Add periodic review mechanism for auto-allow rules (e.g., "You have X auto-approved rules, review them?" on startup)
  - Restrict auto-allow to strictly limited operation types (print, basic math, not file operations)

## Performance Bottlenecks

**Memory Retrieval Search Not Optimized:**
- Problem: ContextRetriever does full database scans for semantic similarity without indexing
- Files: `src/mai/memory/retrieval.py`
- Cause: Vector similarity search likely using brute-force nearest-neighbor without FAISS or similar
- Improvement path:
  - Add FAISS vector index for semantic search acceleration
  - Implement result caching for frequent queries
  - Add search result pagination to avoid loading entire result sets
  - Benchmark retrieval latency and set targets (e.g., <500ms for top-10 similar conversations)

**Conversation State History Accumulation:**
- Problem: ConversationState.conversation_history grows unbounded during long sessions
- Files: `src/mai/conversation/state.py`
- Cause: No automatic truncation or archival of old turns; all conversation turns kept in memory
- Improvement path:
  - Implement sliding window of recent turns (e.g., keep last 50 turns in memory)
  - Archive old turns to disk and load on demand
  - Add compression trigger at configurable message count
  - Monitor memory usage and alert when conversation history exceeds threshold

**Memory Manager Compression Not Scheduled:**
- Problem: Manual `compress_conversation()` calls required; no automatic compression scheduling
- Files: `src/mai/memory/manager.py`
- Cause: Compression is triggered manually or not at all; no background task or event-driven compression
- Improvement path:
  - Implement background compression task triggered by conversation age or message count
  - Add periodic compression sweep for all old conversations
  - Make compression interval configurable (e.g., compress every 500 messages or 24 hours)
  - Track compression effectiveness and adjust thresholds

## Fragile Areas

**Ollama Integration Dependency:**
- Files: `src/mai/model/ollama_client.py`, `src/mai/core/interface.py`
- Why fragile: Hard-coded Ollama endpoint assumption; no fallback model provider; no retry logic for model inference
- Safe modification:
  - Use dependency injection for model provider (interface-based)
  - Add configurable model provider endpoints
  - Implement retry logic with exponential backoff for transient failures
  - Add model availability detection at startup
- Test coverage: Limited tests for model switching and unavailability scenarios

**Git Integration Fragility:**
- Files: `src/mai/git/committer.py`, `src/mai/git/workflow.py`
- Why fragile: Assumes clean git state; no handling for merge conflicts, detached HEAD, or dirty working directory
- Safe modification:
  - Add pre-commit git status validation
  - Handle merge conflict detection and defer commits
  - Implement conflict resolution strategy (manual review or aborting)
  - Test against all git states (detached HEAD, dirty working tree, conflicted merge)
- Test coverage: No tests for edge cases like merge conflicts

**Conversation State Serialization Round-Trip:**
- Files: `src/mai/conversation/state.py`, `src/mai/models/conversation.py`
- Why fragile: ConversationTurn -> Ollama message -> ConversationTurn conversion can lose context
- Safe modification:
  - Add comprehensive unit tests for serialization round-trip
  - Document serialization format and invariants
  - Add validation after deserialization (verify message count, order, role integrity)
  - Create fixture tests with edge cases (unicode, very long messages, special characters)
- Test coverage: No existing tests for message serialization/deserialization

**Docker Configuration Hardcoding:**
- Files: `src/mai/sandbox/docker_executor.py`
- Why fragile: Docker image names, CPU limits, memory limits hardcoded as class constants
- Safe modification:
  - Move Docker config to configuration file
  - Add validation on startup that Docker limits match system resources
  - Document all Docker configuration assumptions
  - Make limits tunable per system resource profile
- Test coverage: Docker integration tests likely mocked; no testing on actual Docker variations

## Scaling Limits

**Memory Database Size Growth:**
- Current capacity: SQLite with no explicit limits; storage grows with every conversation
- Limit: SQLite performance degrades significantly above ~1GB; queries become slow
- Scaling path:
  - Implement database rotation (archive old conversations, start new DB periodically)
  - Add migration path to PostgreSQL for production deployments
  - Implement automatic old conversation archival (move to cold storage after 30 days)
  - Add database vacuum and index optimization on scheduled basis

**Conversation Context Window Management:**
- Current capacity: Model context window determined by Ollama model selection (varies)
- Limit: ConversationEngine doesn't prevent context overflow; will fail when history exceeds model limit
- Scaling path:
  - Track token count of conversation history and refuse new messages before overflow
  - Implement automatic compression trigger at 80% context usage
  - Add model switching logic to use larger-context models if available
  - Document context budget requirements per model

**Approval History Unbounded Growth:**
- Current capacity: ApprovalSystem.approval_history list grows indefinitely
- Limit: Memory accumulation over time; each approval decision stored in memory forever
- Scaling path:
  - Archive approval history to database after threshold (e.g., 1000 decisions)
  - Implement approval history rotation with configurable retention
  - Add aggregate statistics (approval patterns) instead of storing raw history
  - Clean up approval history on startup or scheduled task

## Dependencies at Risk

**Ollama Dependency and Model Availability:**
- Risk: Hard requirement on Ollama being available and having models installed
- Impact: Mai cannot function without Ollama; no fallback to cloud inference or other providers
- Migration plan:
  - Implement abstract model provider interface
  - Add support for OpenAI/other cloud models as fallback (even if v1 is offline-first)
  - Document minimum Ollama model requirements
  - Add diagnostic tool to check Ollama health on startup

**Docker Dependency for Sandboxing:**
- Risk: Docker required for code execution safety; no alternative sandbox implementations
- Impact: Users without Docker can't safely execute generated code; no graceful degradation
- Migration plan:
  - Implement abstract executor interface (not just DockerExecutor)
  - Add noop executor for testing
  - Consider lightweight alternatives (seccomp, chroot, or bubblewrap) for Linux systems
  - Add explicit warning if Docker unavailable

**Rich Library Terminal Detection:**
- Risk: Rich disables colors in non-terminal environments; users see degraded UX
- Impact: Resource monitoring and status displays lack visual feedback in non-terminal contexts
- Migration plan:
  - Use Console(force_terminal=True) to force color output when desired
  - Add configuration option for color preference
  - Implement fallback emoji/unicode indicators for non-color environments
  - Test in various terminal emulators and SSH sessions

## Missing Critical Features

**Session Data Portability:**
- Problem: Session files are JSON but no export/import mechanism; can't backup or migrate sessions
- Blocks: Users can't back up conversations; losing ~/.mai/session.json loses all context
- Fix: Add export/import commands (/export, /import) and document session file format

**Conversation Memory Persistence:**
- Problem: Conversation history is session-scoped (stored in memory); not saved to memory system
- Blocks: Long-term pattern learning relies on memory system but conversations aren't automatically stored
- Fix: Implement automatic conversation archival to memory system after session ends

**User Preference Learning Audit Trail:**
- Problem: User preferences for auto-approval learned silently; no visibility into what patterns auto-approve
- Blocks: Users can't audit their own auto-approval rules; hard to recover from accidentally enabling auto-allow
- Fix: Add /preferences or /audit command to show all learned rules and allow revocation

**Resource Constraint Graceful Degradation:**
- Problem: System shows resource usage but doesn't adapt model selection or conversation behavior
- Blocks: Mai can't suggest switching to smaller models when resources tight
- Fix: Implement resource-aware model recommendation system

**Approval Change Logging:**
- Problem: Approval decisions not tracked in git; can't audit "who approved what when"
- Blocks: No accountability trail for approval decisions
- Fix: Log all approval decisions to git with commit messages including timestamp and user

## Test Coverage Gaps

**Docker Executor Network Isolation:**
- What's not tested: Whether network actually restricted in Docker containers
- Files: `src/mai/sandbox/docker_executor.py`
- Risk: Code might have network access despite supposed isolation
- Priority: High (security-critical)

**Session Persistence Edge Cases:**
- What's not tested: Very large conversations (1000+ messages), unicode characters, special characters
- Files: `src/mai/conversation/state.py`, session persistence code
- Risk: Session files corrupt or lose data with edge case inputs
- Priority: High (data loss)

**Approval System Obfuscation Bypass:**
- What's not tested: Obfuscated code patterns, string concatenation attacks, bytecode approaches
- Files: `src/mai/sandbox/approval_system.py`
- Risk: Risky code could slip through as "low risk" via obfuscation
- Priority: High (security-critical)

**Memory Compression Round-Trip Data Loss:**
- What's not tested: Whether compressed conversations can be exactly reconstructed
- Files: `src/mai/memory/compression.py`, `src/mai/memory/storage.py`
- Risk: Compression could lose important context patterns; compression metrics may be misleading
- Priority: Medium (data integrity)

**Model Switching During Active Conversation:**
- What's not tested: Switching models mid-conversation, context migration, embedding space changes
- Files: `src/mai/model/switcher.py`, `src/mai/conversation/engine.py`
- Risk: Context might not transfer correctly when models switch
- Priority: Medium (feature reliability)

**Offline Queue Conflict Resolution:**
- What's not tested: What happens when offline messages conflict with new context when reconnecting
- Files: `src/mai/conversation/engine.py` (offline queueing)
- Risk: Offline messages might create incoherent conversation when reconnected
- Priority: Medium (conversation coherence)

**Resource Detector System Resource Edge Cases:**
- What's not tested: GPU detection on systems with unusual hardware, CPU count on virtual systems
- Files: `src/mai/model/resource_detector.py`
- Risk: Wrong model selection due to misdetected resources
- Priority: Low (graceful degradation usually handles this)

---

*Concerns audit: 2026-01-26*
