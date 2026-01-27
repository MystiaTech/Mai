# Phase 02: Safety & Sandboxing - Context

**Gathered:** 2026-01-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement sandbox execution environment for generated code, multi-level security assessment, audit logging with tamper detection, and resource-limited container execution.

</domain>

<decisions>
## Implementation Decisions

### Security Assessment Levels
- **BLOCKED triggers:** Code analysis detects malicious patterns AND known threats; behavioral patterns limited to external code (not Mai herself)
- **HIGH triggers:** Privileged access attempts (admin/root access, system file modifications, privilege escalation)
- **BLOCKED response:** Request user override with explanation before proceeding
- **Claude's Discretion:** Specific pattern matching algorithms and threshold tuning

### Audit Logging Scope
- **Logging level:** Comprehensive logging of all code execution, file access, network calls, and system modifications
- **Log content:** Timestamps, code diffs, security events, resource usage, and violation reasons
- **Claude's Discretion:** Log retention period, storage format, and alerting mechanisms

### Sandbox Technology
- **Implementation:** Docker containers for isolation with configurable resource limits and easy cleanup
- **Network policy:** Read-only internet access (can fetch dependencies/documentation but cannot send arbitrary requests)
- **Claude's Discretion:** Container configuration, security policies, and isolation mechanisms

### Resource Limits
- **Policy:** Configurable quotas based on task complexity and trust level
- **Dynamic allocation:** Allow 2 CPU cores, 1GB RAM, 2 minute execution time for trusted code
- **Resource monitoring:** Real-time tracking and automatic termination on limit violations
- **Claude's Discretion:** Specific quota amounts, monitoring frequency, and response to violations

### Claude's Discretion
- Audit log retention: Choose appropriate retention policy balancing security and storage
- Sandbox security policies: Choose appropriate container hardening measures
- Network whitelist: Determine which domains are safe for dependency access
- Performance optimization: Balance security overhead with execution efficiency

</decisions>

<specifics>
## Specific Ideas

- Audit logs should be tamper-proof and include cryptographic signatures
- Docker containers should use read-only filesystems where possible
- Security assessment should be fast to avoid blocking user workflow
- Resource limits should adapt to available system resources

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within Phase 2 scope of safety and sandboxing.

</deferred>

---

*Phase: 02-safety-sandboxing*
*Context gathered: 2026-01-27*