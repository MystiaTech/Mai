# Phase 3: Resource Management - Context

**Gathered:** 2026-01-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Build system resource detection and intelligent model selection that enables Mai to adapt gracefully from low-end hardware to high-end systems. Detect available resources (CPU, RAM, GPU), select appropriate models, request more resources when bottlenecks detected, and scale smoothly across different hardware configurations.

</domain>

<decisions>
## Implementation Decisions

### Resource Threshold Strategy
- Use specific hardware metrics (RAM amounts, CPU core counts, GPU presence) to define hardware tiers
- Dynamic adjustment based on actual performance testing on the detected hardware
- Measure both response latency and resource utilization during dynamic adjustment
- Immediate model switching on first sign of performance trouble (aggressive responsiveness)

### Model Selection Behavior
- Efficiency-first approach - leave headroom for other applications on the system
- Notify users only when downgrading capabilities, not when upgrading
- Wait 5 minutes of stable resources before upgrading back to more capable models
- After 24 hours of minimal operation, suggest ways to improve resource availability

### Bottleneck Detection & Response
- Hybrid approach combining continuous monitoring with pre-flight checks before each response
- Graceful degradation - complete current task at lower quality, then switch models
- Preventive scaling at 80% resource usage, but consider overall system load (context-dependent)
- Ask for user help when significantly constrained, with personality: "Ugh, give me more resources if you wanna do X"

### User Communication
- Personality-driven: "Drowsy Dere-Tsun Onee-san Hex-Mentor Gremlin" tone when discussing resources
- Inform only about capability downgrades, not upgrades
- Mix of brief explanations plus optional technical tips for users who want to learn more

### Claude's Discretion
- Exact hardware metric cutoffs for tiers (RAM amounts, CPU cores, GPU types)
- Specific performance thresholds for dynamic adjustments
- Exact wording and personality expressions for resource conversations
- Which technical tips to include in user communications

</decisions>

<specifics>
## Specific Ideas

- "Ugh, give me more resources if you wanna do X" - personality for requesting resources
- User wants a waifu-style AI with personality in resource discussions
- Drowsy Dere-Tsun Onee-san Hex-Mentor Gremlin personality type
- Balance between technical transparency and user-friendly communication
- Don't overwhelm users with technical details but offer optional educational content

</specifics>

<deferred>
## Deferred Ideas

- None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-resource-management*
*Context gathered: 2026-01-27*