# Phase 05: Conversation Engine - Research

**Researched:** 2026-01-29
**Domain:** Conversational AI with multi-turn dialogue management
**Confidence:** HIGH

## Summary

This research focused on implementing Mai's conversational intelligence - how she handles multi-turn conversations, thinks through problems, and communicates naturally. The research revealed that **LangGraph** is the established industry standard for conversation state management, providing robust solutions for multi-turn context preservation, streaming responses, and persistence. 

Key findings show that the ecosystem has matured significantly since early 2025, with LangGraph v0.5+ providing production-ready patterns for conversation state management, checkpointing, and streaming that directly align with Mai's requirements. The combination of **LangGraph's StateGraph** with **MessagesState** and **MemorySaver** provides exactly what Mai needs for multi-turn conversations, thinking transparency, and response timing.

**Primary recommendation:** Use LangGraph StateGraph with MessagesState for conversation management, MemorySaver for persistence, and async generators for streaming responses.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| **langgraph** | 0.5+ | Conversation state management | Industry standard for stateful agents, built-in checkpointing and streaming |
| **langchain-core** | 0.3+ | Message types and abstractions | Provides MessagesState, message role definitions |
| **asyncio** | Python 3.10+ | Async response streaming | Native Python async patterns for real-time streaming |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **pydantic** | 2.0+ | Data validation | Already in project, use for state schemas |
| **typing-extensions** | 4.0+ | TypedDict support | Required for LangGraph state definitions |
| **asyncio-mqtt** | 0.16+ | Real-time events | For future real-time features |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| LangGraph | Custom conversation manager | LangGraph provides proven patterns, custom would be complex to maintain |
| MessagesState | Custom TypedDict | MessagesState has built-in message aggregation via `add_messages` |
| MemorySaver | SQLite custom checkpointer | MemorySaver is production-tested with built-in serialization |

**Installation:**
```bash
pip install langgraph>=0.5 langchain-core>=0.3 typing-extensions
```

## Architecture Patterns

### Recommended Project Structure
```
src/conversation/
├── __init__.py              # Main ConversationEngine class
├── state.py                 # State schema definitions
├── nodes.py                 # LangGraph node functions
├── streaming.py              # Async streaming utilities
├── clarity.py               # Clarification handling
└── timing.py               # Response timing management
```

### Pattern 1: LangGraph StateGraph with MessagesState
**What:** Use StateGraph with MessagesState for conversation state management
**When to use:** All conversation flows requiring multi-turn context
**Example:**
```python
# Source: https://python.langchain.com/docs/langgraph
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

class ConversationState(MessagesState):
    user_id: str
    needs_clarification: bool
    response_type: str  # "direct", "clarifying", "breakdown"

def process_message(state: ConversationState):
    # Process user message and determine response type
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check for ambiguity
    if is_ambiguous(last_message.content):
        return {
            "needs_clarification": True,
            "response_type": "clarifying"
        }
    
    return {"response_type": "direct"}

# Build conversation graph
builder = StateGraph(ConversationState)
builder.add_node("process", process_message)
builder.add_node("respond", generate_response)
builder.add_edge(START, "process")
builder.add_conditional_edges("process", route_response)
builder.add_edge("respond", END)

# Add memory for persistence
checkpointer = MemorySaver()
conversation_graph = builder.compile(checkpointer=checkpointer)
```

### Pattern 2: Async Streaming with Response Timing
**What:** Use async generators with variable timing for natural response flow
**When to use:** All response generation to provide natural conversational pacing
**Example:**
```python
# Source: Async streaming patterns research
import asyncio
from typing import AsyncGenerator
import time

async def stream_response_with_timing(
    content: str, 
    response_type: str = "direct"
) -> AsyncGenerator[str, None]:
    """Stream response with natural timing based on context."""
    
    chunks = split_into_chunks(content, chunk_size=10)
    
    for i, chunk in enumerate(chunks):
        # Variable timing based on response type and position
        if response_type == "thinking":
            # Longer pauses for "thinking" responses
            await asyncio.sleep(0.3 + (i * 0.1))
        elif response_type == "clarifying":
            # Shorter, more frequent chunks for questions
            await asyncio.sleep(0.1)
        else:
            # Normal conversational timing
            await asyncio.sleep(0.2)
        
        yield chunk

async def generate_response(state: ConversationState):
    """Generate response with appropriate timing and streaming."""
    messages = state["messages"]
    response_type = state.get("response_type", "direct")
    
    # Generate response content
    response_content = await llm.ainvoke(messages)
    
    # Return streaming-ready response
    return {
        "messages": [{"role": "assistant", "content": response_content}],
        "response_stream": stream_response_with_timing(response_content, response_type)
    }
```

### Pattern 3: Clarification Detection and Handling
**What:** Proactive ambiguity detection with gentle clarification requests
**When to use:** When user input is unclear or multiple interpretations exist
**Example:**
```python
# Based on context decisions for clarification handling
from typing import List, Optional
import re

class AmbiguityDetector:
    def __init__(self):
        self.ambiguous_patterns = [
            r"it", r"that", r"this", r"thing",  # Pronouns without context
            r"do that", r"make it", r"fix it"    # Vague instructions
        ]
        
    def detect_ambiguity(self, message: str, context: List[str]) -> Optional[str]:
        """Detect if message is ambiguous and suggest clarification."""
        
        # Check for ambiguous pronouns
        for pattern in self.ambiguous_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                if not self.has_context_for_pronoun(message, context):
                    return "gentle_pronoun"
        
        # Check for vague instructions
        if self.is_vague_instruction(message):
            return "gentle_specificity"
        
        return None
    
    def generate_clarification(self, ambiguity_type: str, original_message: str) -> str:
        """Generate gentle clarification question."""
        
        if ambiguity_type == "gentle_pronoun":
            return f"I want to make sure I understand correctly. When you say '{original_message}', could you clarify what specific thing you're referring to?"
        
        elif ambiguity_type == "gentle_specificity":
            return f"I'd love to help with that! Could you provide a bit more detail about what specifically you'd like me to do?"
        
        return "Could you tell me a bit more about what you have in mind?"
```

### Anti-Patterns to Avoid
- **Fixed response timing:** Don't use fixed delays between chunks - timing should vary based on context
- **Explicit "thinking..." messages:** Avoid explicit status messages in favor of natural timing
- **Assumption-based responses:** Never proceed without clarification when ambiguity is detected
- **Memory-less conversations:** Every conversation node must maintain state through checkpointing

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Conversation state management | Custom message list + manual tracking | LangGraph StateGraph + MessagesState | Built-in message aggregation, checkpointing, and serialization |
| Async streaming responses | Manual chunk generation with sleep() | LangGraph streaming + async generators | Proper async context handling, backpressure management |
| Conversation persistence | Custom SQLite schema | LangGraph checkpointer (MemorySaver/Redis) | Thread-safe state snapshots, time-travel debugging |
| Message importance scoring | Custom heuristics | LangGraph's message metadata + context | Built-in message prioritization and compression support |

**Key insight:** Building custom conversation state management is notoriously error-prone. State consistency, concurrent access, and proper serialization are solved problems in LangGraph.

## Common Pitfalls

### Pitfall 1: State Mutation Instead of Updates
**What goes wrong:** Directly modifying state objects instead of returning new state
**Why it happens:** Python's object reference model encourages mutation
**How to avoid:** Always return new state dictionaries from nodes, never modify existing state
**Warning signs:** State changes not persisting between graph invocations

```python
# WRONG - Mutates state directly
def bad_node(state):
    state["messages"].append(new_message)  # Mutates list
    return state

# CORRECT - Returns new state
def good_node(state):
    return {"messages": [new_message]}  # LangGraph handles aggregation
```

### Pitfall 2: Missing Thread Configuration
**What goes wrong:** Multiple conversations sharing the same state
**Why it happens:** Forgetting to set thread_id in graph configuration
**How to avoid:** Always pass config with thread_id for each conversation
**Warning signs:** Cross-contamination between different user conversations

```python
# REQUIRED - Configure thread for each conversation
config = {"configurable": {"thread_id": conversation_id}}
result = graph.invoke({"messages": [user_message]}, config=config)
```

### Pitfall 3: Blocking Operations in Async Context
**What goes wrong:** Synchronous LLM calls blocking the event loop
**Why it happens:** Using sync LLM clients in async graph nodes
**How to avoid:** Use async LLM clients (ainvoke, astream) throughout
**Warning signs:** Poor responsiveness, CPU blocking during LLM calls

### Pitfall 4: Inadequate Error Handling in Streams
**What goes wrong:** Stream errors crashing the entire conversation
**Why it happens:** Not wrapping async generators in try-catch blocks
**How to avoid:** Use proper error handling with graceful degradation
**Warning signs:** Conversation termination on network issues

## Code Examples

### Multi-Turn Conversation with Memory
```python
# Source: https://python.langchain.com/docs/langgraph/persistence
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

class ConversationEngine:
    def __init__(self):
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()
    
    def _build_graph(self):
        builder = StateGraph(MessagesState)
        
        def chat_node(state: MessagesState):
            # Process conversation with full context
            messages = state["messages"]
            response = self.llm.ainvoke(messages)
            return {"messages": [response]}
        
        builder.add_node("chat", chat_node)
        builder.add_edge(START, "chat")
        builder.add_edge("chat", END)
        
        return builder.compile(checkpointer=self.checkpointer)
    
    async def chat(self, message: str, conversation_id: str):
        config = {"configurable": {"thread_id": conversation_id}}
        
        # Add user message and get response
        async for event in self.graph.astream_events(
            {"messages": [HumanMessage(content=message)]},
            config,
            version="v1"
        ):
            # Stream response in real-time
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if hasattr(chunk, "content") and chunk.content:
                    yield chunk.content
    
    def get_conversation_history(self, conversation_id: str):
        config = {"configurable": {"thread_id": conversation_id}}
        state = self.graph.get_state(config)
        return state.values.get("messages", [])
```

### Complex Request Breakdown
```python
# Based on context decisions for breaking down complex requests
class RequestBreakdown:
    def analyze_complexity(self, message: str) -> tuple[bool, List[str]]:
        """Analyze if request is complex and break it down."""
        
        complexity_indicators = [
            "and then", "after that", "also", "in addition",
            "finally", "first", "second", "third"
        ]
        
        # Check for multi-step indicators
        is_complex = any(indicator in message.lower() 
                       for indicator in complexity_indicators)
        
        if not is_complex:
            return False, [message]
        
        # Break down into steps
        steps = self._extract_steps(message)
        return True, steps
    
    def confirm_breakdown(self, steps: List[str]) -> str:
        """Generate confirmation message for breakdown."""
        steps_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))
        return f"I understand you want me to:\n{steps_text}\n\nShould I proceed with these steps in order?"
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Custom conversation state | LangGraph StateGraph + MessagesState | Late 2025 (v0.3) | Dramatic reduction in conversation management bugs |
| Manual memory management | Built-in checkpointing with MemorySaver | Early 2026 (v0.5) | Thread-safe persistence with time-travel debugging |
| Fixed response streaming | Variable timing with async generators | Throughout 2025 | More natural conversation flow |
| Separate tools for streaming | Integrated streaming in LangGraph core | Late 2025 | Unified streaming and state management |

**Deprecated/outdated:**
- **LangChain Memory classes:** Deprecated in v0.3, replaced by LangGraph state management
- **Custom message aggregation:** No longer needed with MessagesState and `add_messages`
- **Manual persistence threading:** Replaced by thread_id configuration in LangGraph
- **Sync streaming patterns:** Async generators are now standard for all streaming

## Open Questions

1. **LLM Integration Timing:** Should Mai switch to smaller/faster models for clarification requests vs. complex responses? (Context suggests model switching exists, but timing algorithms are Claude's discretion)
2. **Conversation Session Limits:** What's the optimal checkpoint retention period for balance between memory usage and conversation history? (Research didn't reveal clear best practices)
3. **Real-time Collaboration:** How should concurrent access to the same conversation be handled? (Multiple users collaborating on same conversation)

Recommendation: Start with conservative defaults (1 week retention, single user per conversation) and iterate based on usage patterns.

## Sources

### Primary (HIGH confidence)
- **LangGraph Documentation** - State management, checkpointing, and streaming patterns
- **LangChain Core Messages** - Message types and MessagesState implementation
- **AsyncIO Python Documentation** - Async generator patterns and event loop management

### Secondary (MEDIUM confidence)
- **"Persistence in LangGraph — Deep, Practical Guide" (Jan 2026)** - Verified patterns with official docs
- **"Streaming APIs for Beginners" (Oct 2025)** - Async streaming patterns confirmed with Python docs
- **Multiple Medium articles on LangGraph conversation patterns** - Cross-verified with official sources

### Tertiary (LOW confidence)
- **Individual GitHub repositories** - Various conversation engine implementations (marked for validation)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - LangGraph is clearly documented and widely adopted
- Architecture: HIGH - Official patterns are well-established and tested
- Pitfalls: HIGH - Common issues are documented in official guides with solutions

**Research date:** 2026-01-29
**Valid until:** 2026-03-01 (LangGraph ecosystem is stable, but new features may emerge)

---

*Phase: 05-conversation-engine*
*Research completed: 2026-01-29*