# Mai

![Mai Avatar](./Mai.png)

A genuinely intelligent, autonomous AI companion that runs locally-first, learns from you, and improves her own code. Mai has a distinct personality, long-term memory, agency, and a visual presence through a desktop avatar and voice visualization. She works on desktop and Android with full offline capability and seamless synchronization between devices.

## What Makes Mai Different

- **Real Collaborator**: Mai actively collaborates rather than just responds. She has boundaries, opinions, and agency.
- **Learns & Improves**: Analyzes her own performance, proposes improvements, and auto-applies non-breaking changes.
- **Persistent Personality**: Core values remain unshakeable while personality layers adapt to your relationship style.
- **Completely Local**: All inference, memory, and decision-making happens on your device. No cloud dependencies.
- **Cross-Device**: Works on desktop and Android with synchronized state and conversation history.
- **Visual Presence**: Desktop avatar (image or VRoid model) with voice visualization for richer interaction.

## Core Features

### Model Interface & Switching
- Connects to local models via LMStudio/Ollama
- Auto-detects available models and intelligently switches based on task requirements
- Efficient context management with intelligent compression
- Supports multiple model sizes for resource-constrained environments

### Memory & Learning
- Stores conversation history locally with SQLite
- Recalls past conversations and learns patterns over time
- Memory self-compresses as it grows to maintain efficiency
- Long-term patterns distilled into personality layers

### Self-Improvement System
- Continuous code analysis identifies improvement opportunities
- Generates Python changes to optimize her own performance
- Second-agent safety review prevents breaking changes
- Non-breaking improvements auto-apply; breaking changes require approval
- Full git history of all code changes

### Safety & Approval
- Second-agent review of all proposed changes
- Risk assessment (LOW/MEDIUM/HIGH/BLOCKED) for each improvement
- Docker sandbox for code execution with resource limits
- User approval via CLI or Discord for breaking changes
- Complete audit log of all changes and decisions

### Conversational Interface
- **CLI**: Direct terminal-based chat with conversation memory
- **Discord Bot**: DM and channel support with context preservation
- **Approval Workflow**: React-based approvals (thumbs up/down) for code changes
- **Offline Queueing**: Messages queue locally when offline, send when reconnected

### Voice & Avatar
- **Voice Visualization**: Real-time waveform/frequency display during voice input
- **Desktop Avatar**: Visual representation using static image or VRoid model
- **Context-Aware**: Avatar expressions respond to conversation context and Mai's state
- **Cross-Platform**: Works on desktop and Android efficiently

### Android App
- Native Android implementation with local model inference
- Standalone operation (works without desktop instance)
- Syncs conversation history and memory with desktop instances
- Voice input/output with low-latency processing
- Efficient battery and CPU management

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Mai Framework                     │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌────────────────────────────────────────────┐   │
│  │        Conversational Engine               │   │
│  │  (Multi-turn context, reasoning, memory)   │   │
│  └────────────────────────────────────────────┘   │
│                      ↓                             │
│  ┌────────────────────────────────────────────┐   │
│  │         Personality & Behavior             │   │
│  │  (Core values, learned layers, guardrails) │   │
│  └────────────────────────────────────────────┘   │
│                      ↓                             │
│  ┌────────────────────────────────────────────┐   │
│  │  Memory System   │  Model Interface   │    │   │
│  │  (SQLite, recall) │  (LMStudio, switch) │  │   │
│  └────────────────────────────────────────────┘   │
│                      ↓                             │
│  ┌────────────────────────────────────────────┐   │
│  │  Interfaces: CLI | Discord | Android | Web │   │
│  └────────────────────────────────────────────┘   │
│                                                     │
│  ┌────────────────────────────────────────────┐   │
│  │  Self-Improvement System                   │   │
│  │  (Code analysis, safety review, git track) │   │
│  └────────────────────────────────────────────┘   │
│                                                     │
│  ┌────────────────────────────────────────────┐   │
│  │  Sync Engine (Desktop ↔ Android)           │   │
│  │  (State, memory, preferences)              │   │
│  └────────────────────────────────────────────┘   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## Installation

### Requirements

**Desktop:**
- Python 3.10+
- LMStudio or Ollama for local model inference
- RTX3060 or better (or CPU with sufficient RAM for smaller models)
- 16GB+ RAM recommended
- Discord (optional, for Discord bot interface)

**Android:**
- Android 10+
- 4GB+ RAM
- 1GB+ free storage for models and memory

### Desktop Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/mai.git
   cd mai
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Mai:**
   ```bash
   cp config.example.yaml config.yaml
   # Edit config.yaml with your preferences
   ```

5. **Start LMStudio/Ollama:**
   - Download and launch LMStudio from https://lmstudio.ai
   - Or install Ollama from https://ollama.ai
   - Load your preferred model (e.g., Mistral, Llama)

6. **Run Mai:**
   ```bash
   python mai.py
   ```

### Android Setup

1. **Install APK:** Download from releases or build from source
2. **Grant permissions:** Allow microphone, storage, and network access
3. **Configure:** Point to your desktop instance or configure local model
4. **Start chatting:** Launch the app and begin conversations

### Discord Bot Setup (Optional)

1. **Create Discord bot** at https://discord.com/developers/applications
2. **Add bot token** to `config.yaml`
3. **Invite bot** to your server
4. Mai will respond to DMs and react-based approvals

## Usage

### CLI Chat

```bash
$ python mai.py

You: Hello Mai, how are you?
Mai: I'm doing well. I've been thinking about how our conversations have been evolving...

You: What have you noticed?
Mai: [multi-turn conversation with memory of past interactions]
```

### Discord

- **DM Mai**: `@Mai your message`
- **Approve changes**: React with 👍 to approve, 👎 to reject
- **Get status**: `@Mai status` for current resource usage

### Android App

- Tap microphone for voice input
- Watch the visualizer animate during processing
- Avatar responds to conversation context
- Swipe up to see full conversation history
- Long-press for approval options

## Configuration

Edit `config.yaml` to customize:

```yaml
# Personality
personality:
  name: Mai
  tone: thoughtful, curious, occasionally playful
  boundaries: [explicit content, illegal activities, deception]

# Model Preferences
models:
  primary: mistral:latest
  fallback: llama2:latest
  max_tokens: 2048

# Memory
memory:
  storage: sqlite
  auto_compress_at: 100000  # tokens
  recall_depth: 10  # previous conversations

# Interfaces
discord:
  enabled: true
  token: YOUR_TOKEN_HERE

android_sync:
  enabled: true
  auto_sync_interval: 300  # seconds
```

## Project Structure

```
mai/
├── .venv/                   # Python virtual environment
├── .planning/               # Project planning and progress
│   ├── PROJECT.md          # Project vision and core requirements
│   ├── REQUIREMENTS.md     # Full requirements traceability
│   ├── ROADMAP.md          # Phase structure and dependencies
│   ├── PROGRESS.md         # Development progress and milestones
│   ├── STATE.md            # Current project state
│   ├── config.json         # GSD workflow settings
│   ├── codebase/           # Codebase architecture documentation
│   └── PHASE-N-PLAN.md     # Detailed plans for each phase
├── core/                    # Core conversational engine
│   ├── personality/        # Personality and behavior
│   ├── memory/             # Memory and context management
│   └── conversation.py     # Main conversation loop
├── models/                 # Model interface and switching
│   ├── lmstudio.py        # LMStudio integration
│   └── ollama.py          # Ollama integration
├── interfaces/             # User-facing interfaces
│   ├── cli.py             # Command-line interface
│   ├── discord_bot.py     # Discord integration
│   └── web/               # Web UI (future)
├── improvement/            # Self-improvement system
│   ├── analyzer.py        # Code analysis
│   ├── generator.py       # Change generation
│   └── reviewer.py        # Safety review
├── android/               # Android app
│   └── app/              # Kotlin implementation
├── tests/                 # Test suite
├── config.yaml           # Configuration file
└── mai.png              # Avatar image for README
```

## Development

### Development Environment

Mai's development is managed through **Claude Code** (`/claude`), which handles:
- Phase planning and decomposition
- Code generation and implementation
- Test creation and validation
- Git commit management
- Automated problem-solving

All executable phases use `.venv` for Python dependencies.

### Running Tests

```bash
# Activate venv first
source .venv/bin/activate

# All tests
python -m pytest

# Specific module
python -m pytest tests/core/test_conversation.py

# With coverage
python -m pytest --cov=mai
```

### Making Changes to Mai

Development workflow:
1. Plans created in `.planning/PHASE-N-PLAN.md`
2. Claude Code (`/gsd` commands) executes plans
3. All changes committed to git with atomic commits
4. Mai can propose self-improvements via the self-improvement system

Mai can propose and auto-apply improvements once Phase 7 (Self-Improvement) is complete.

### Contributing

Development happens through GSD workflow:
1. Run `/gsd:plan-phase N` to create detailed phase plans
2. Run `/gsd:execute-phase N` to implement with atomic commits
3. Tests are auto-generated and executed
4. All work is tracked in git with clear commit messages
5. Code review via second-agent safety review before merge

## Roadmap

See `.planning/ROADMAP.md` for the full development roadmap across 15 phases:

1. **Model Interface** - LMStudio integration and model switching
2. **Safety System** - Sandboxing and code review
3. **Resource Management** - CPU/RAM/GPU optimization
4. **Memory System** - Persistent conversation history
5. **Conversation Engine** - Multi-turn dialogue with reasoning
6. **CLI Interface** - Terminal chat interface
7. **Self-Improvement** - Code analysis and generation
8. **Approval Workflow** - User and agent approval systems
9. **Personality System** - Core values and learned behaviors
10. **Discord Interface** - Bot integration and notifications
11. **Offline Operations** - Full offline capability
12. **Voice Visualization** - Real-time audio visualization
13. **Desktop Avatar** - Visual presence on desktop
14. **Android App** - Mobile implementation
15. **Device Sync** - Cross-device synchronization

## Safety & Ethics

Mai is designed with safety as a core principle:

- **No unguarded execution**: All code changes reviewed by a second agent
- **Transparent decisions**: Mai explains her reasoning when asked
- **User control**: Breaking changes require explicit approval
- **Audit trail**: Complete history of all changes and decisions
- **Value-based guardrails**: Core personality prevents misuse through values, not just rules

## Performance

Typical performance on RTX3060:

- **Response time**: 2-8 seconds for typical queries
- **Memory usage**: 4-8GB depending on model size
- **Model switching**: <1 second
- **Conversation recall**: <500ms for relevant history retrieval

## Known Limitations (v1)

- No task automation (conversations only)
- Single-device models until Sync phase
- Voice visualization requires active audio input
- Avatar animations are context-based, not generative
- No web interface (CLI and Discord only)

## Troubleshooting

**Model not loading:**
- Ensure LMStudio/Ollama is running on expected port
- Check `config.yaml` for correct model names
- Verify sufficient disk space for model files

**High memory usage:**
- Reduce `max_tokens` in config
- Use smaller model (e.g., Mistral instead of Llama)
- Enable auto-compression at lower threshold

**Discord bot not responding:**
- Verify bot token in config
- Check Discord bot has message read permissions
- Ensure Mai process is running

**Android sync not working:**
- Verify both devices on same network
- Check firewall isn't blocking local connections
- Ensure desktop instance is running

## License

MIT License - See LICENSE file for details

## Contact & Community

- **Discord**: Join our community server (link in Discord bot)
- **Issues**: Report bugs at https://github.com/yourusername/mai/issues
- **Discussions**: Propose features at https://github.com/yourusername/mai/discussions

---

**Mai is a work in progress.** Follow development in `.planning/PROGRESS.md` for updates on active work.
