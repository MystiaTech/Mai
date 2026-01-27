"""CLI entry point for Mai."""

import argparse
import asyncio
import sys
import signal
from typing import Optional

from .mai import Mai


def setup_argparser() -> argparse.ArgumentParser:
    """Setup command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="mai",
        description="Mai - Intelligent AI companion with model switching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mai chat                    # Start interactive chat mode
  mai status                  # Show current model and system status
  mai models                  # List available models
  mai switch qwen2.5-7b      # Switch to specific model
  mai --help                  # Show this help message
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Chat command
    chat_parser = subparsers.add_parser(
        "chat", help="Start interactive conversation mode"
    )
    chat_parser.add_argument(
        "--model", "-m", type=str, help="Override model for this session"
    )
    chat_parser.add_argument(
        "--conversation-id",
        "-c",
        type=str,
        default="default",
        help="Conversation ID to use (default: default)",
    )

    # Status command
    status_parser = subparsers.add_parser(
        "status", help="Show current model and system status"
    )
    status_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed status information"
    )

    # Models command
    models_parser = subparsers.add_parser(
        "models", help="List available models and their status"
    )
    models_parser.add_argument(
        "--available-only",
        "-a",
        action="store_true",
        help="Show only available models (hide unavailable)",
    )

    # Switch command
    switch_parser = subparsers.add_parser(
        "switch", help="Manually switch to a specific model"
    )
    switch_parser.add_argument(
        "model_key",
        type=str,
        help="Model key to switch to (e.g., qwen/qwen2.5-7b-instruct)",
    )
    switch_parser.add_argument(
        "--conversation-id",
        "-c",
        type=str,
        default="default",
        help="Conversation ID context for switch",
    )

    return parser


async def chat_command(args, mai: Mai) -> None:
    """Handle interactive chat mode."""
    print("🤖 Starting Mai chat interface...")
    print("Type 'quit', 'exit', or press Ctrl+C to end conversation")
    print("-" * 50)

    conversation_id = args.conversation_id

    # Try to set initial model if specified
    if args.model:
        print(f"🔄 Attempting to switch to model: {args.model}")
        success = await mai.switch_model(args.model)
        if success:
            print(f"✅ Successfully switched to {args.model}")
        else:
            print(f"❌ Failed to switch to {args.model}")
            print("Continuing with current model...")

    # Start background tasks
    mai.running = True
    mai.start_background_tasks()

    try:
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("\n👋 Goodbye!")
                    break

                if not user_input:
                    continue

                # Process message
                print("🤔 Thinking...")
                response = await mai.process_message_async(user_input, conversation_id)

                print(f"\n🤖 Mai: {response}")

            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except EOFError:
                print("\n\n👋 End of input. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again or type 'quit' to exit.")

    finally:
        mai.shutdown()


def status_command(args, mai: Mai) -> None:
    """Handle status display command."""
    status = mai.get_system_status()

    print("📊 Mai System Status")
    print("=" * 40)

    # Main status
    mai_status = status.get("mai_status", "unknown")
    print(f"🤖 Mai Status: {mai_status}")

    # Model information
    model_info = status.get("model", {})
    if model_info:
        print(f"\n📋 Current Model:")
        model_key = model_info.get("current_model_key", "None")
        display_name = model_info.get("model_display_name", "Unknown")
        category = model_info.get("model_category", "unknown")
        model_loaded = model_info.get("model_loaded", False)

        status_icon = "✅" if model_loaded else "❌"
        print(f"  {status_icon} {display_name} ({category})")
        print(f"  🔑 Key: {model_key}")

        if args.verbose:
            context_window = model_info.get("context_window", "Unknown")
            print(f"  📝 Context Window: {context_window} tokens")

    # Resource information
    resources = status.get("system_resources", {})
    if resources:
        print(f"\n📈 System Resources:")
        print(
            f"  💾 Memory: {resources.get('memory_percent', 0):.1f}% ({resources.get('available_memory_gb', 0):.1f}GB available)"
        )
        print(f"  🖥️  CPU: {resources.get('cpu_percent', 0):.1f}%")
        gpu_vram = resources.get("gpu_vram_gb", 0)
        if gpu_vram > 0:
            print(f"  🎮 GPU VRAM: {gpu_vram:.1f}GB available")
        else:
            print(f"  🎮 GPU: Not available or not detected")

    # Conversation information
    conversations = status.get("conversations", {})
    if conversations:
        print(f"\n💬 Conversations:")
        for conv_id, stats in conversations.items():
            msg_count = stats.get("total_messages", 0)
            tokens_used = stats.get("context_tokens_used", 0)
            tokens_max = stats.get("context_tokens_max", 0)

            print(f"  📝 {conv_id}: {msg_count} messages")
            if args.verbose:
                usage_pct = stats.get("context_usage_percentage", 0)
                print(
                    f"     📊 Context: {usage_pct:.1f}% ({tokens_used}/{tokens_max} tokens)"
                )

    # Available models
    available_count = model_info.get("available_models", 0)
    print(f"\n🔧 Available Models: {available_count}")

    # Error state
    if "error" in status:
        print(f"\n❌ Error: {status['error']}")


def models_command(args, mai: Mai) -> None:
    """Handle model listing command."""
    models = mai.list_available_models()

    print("🤖 Available Models")
    print("=" * 50)

    if not models:
        print(
            "❌ No models available. Check LM Studio connection and downloaded models."
        )
        return

    current_model_key = mai.model_manager.current_model_key

    for model in models:
        key = model.get("key", "Unknown")
        display_name = model.get("display_name", "Unknown")
        category = model.get("category", "unknown")
        available = model.get("available", False)
        estimated_size = model.get("estimated_size_gb", 0)

        if args.available_only and not available:
            continue

        # Status indicator
        if key == current_model_key:
            status = "🟢 CURRENT"
        elif available:
            status = "✅ Available"
        else:
            status = "❌ Unavailable"

        print(
            f"{status:<12} {display_name:<30} ({category:<7}) [{estimated_size:.1f}GB]"
        )
        print(f"{' ':>12} 🔑 {key}")
        print()


async def switch_command(args, mai: Mai) -> None:
    """Handle manual model switch command."""
    model_key = args.model_key
    conversation_id = args.conversation_id

    print(f"🔄 Switching to model: {model_key}")

    success = await mai.switch_model(model_key)

    if success:
        print(f"✅ Successfully switched to {model_key}")

        # Show new status
        new_status = mai.get_system_status()
        model_info = new_status.get("model", {})
        display_name = model_info.get("model_display_name", model_key)
        print(f"📋 Now using: {display_name}")

    else:
        print(f"❌ Failed to switch to {model_key}")
        print("Possible reasons:")
        print("  • Model not found in configuration")
        print("  • Insufficient system resources")
        print("  • Model failed to load")
        print("\nTry 'mai models' to see available models.")


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    print(f"\n\n👋 Received signal {signum}. Shutting down gracefully...")
    sys.exit(0)


def main():
    """Main entry point for CLI."""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Parse arguments
    parser = setup_argparser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize Mai
    try:
        mai = Mai()
    except Exception as e:
        print(f"❌ Failed to initialize Mai: {e}")
        sys.exit(1)

    try:
        # Route to appropriate command
        if args.command == "chat":
            # Run chat mode with asyncio
            asyncio.run(chat_command(args, mai))
        elif args.command == "status":
            status_command(args, mai)
        elif args.command == "models":
            models_command(args, mai)
        elif args.command == "switch":
            # Run switch with asyncio
            asyncio.run(switch_command(args, mai))
        else:
            print(f"❌ Unknown command: {args.command}")
            parser.print_help()

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
    except Exception as e:
        print(f"❌ Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
