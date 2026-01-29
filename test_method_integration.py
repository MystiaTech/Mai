"""
Simple verification script for SQLiteManager methods integration.

This script verifies that the newly added methods can be called
without AttributeError and return data in proper format.
"""

import sys
import os
from datetime import datetime, timedelta
import tempfile

# Add project path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from memory.storage.sqlite_manager import SQLiteManager


def test_methods_exist():
    """Verify methods exist and can be called."""
    print("Testing method existence...")

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name

    try:
        # Initialize manager
        manager = SQLiteManager(db_path)

        # Create test conversation
        conv_id = "test_conv_001"
        manager.create_conversation(conv_id, title="Test Conversation")

        # Add test messages
        manager.add_message(
            "msg_001",
            conv_id,
            "user",
            "Hello, this is a test message",
            token_count=10
        )
        manager.add_message(
            "msg_002",
            conv_id,
            "assistant",
            "Hello! This is a response",
            token_count=8
        )

        # Test get_conversations_by_date_range
        print("\n1. Testing get_conversations_by_date_range...")
        start_date = datetime.now() - timedelta(days=1)
        end_date = datetime.now() + timedelta(days=1)

        conversations = manager.get_conversations_by_date_range(start_date, end_date)

        assert isinstance(conversations, list), "Should return a list"
        assert len(conversations) > 0, "Should find at least one conversation"
        assert "id" in conversations[0], "Conversation should have 'id' field"
        assert "title" in conversations[0], "Conversation should have 'title' field"
        assert "metadata" in conversations[0], "Conversation should have 'metadata' field"
        print(f"   ✓ Found {len(conversations)} conversation(s)")
        print(f"   ✓ Data format verified: {list(conversations[0].keys())}")

        # Test get_conversation_messages
        print("\n2. Testing get_conversation_messages...")
        messages = manager.get_conversation_messages(conv_id)

        assert isinstance(messages, list), "Should return a list"
        assert len(messages) == 2, "Should find 2 messages"
        assert "id" in messages[0], "Message should have 'id' field"
        assert "role" in messages[0], "Message should have 'role' field"
        assert "content" in messages[0], "Message should have 'content' field"
        assert "timestamp" in messages[0], "Message should have 'timestamp' field"
        print(f"   ✓ Found {len(messages)} message(s)")
        print(f"   ✓ Data format verified: {list(messages[0].keys())}")

        # Test with PersonalityLearner context
        print("\n3. Testing integration with PersonalityLearner context...")

        # Simulate how PersonalityLearner would use these methods
        conversation_range = (start_date, end_date)
        conversations = manager.get_conversations_by_date_range(
            conversation_range[0], conversation_range[1]
        )

        for conv in conversations:
            messages = manager.get_conversation_messages(conv["id"])
            assert isinstance(messages, list), "Messages should be a list"
            assert all("role" in msg for msg in messages), "All messages should have role"
            assert all("content" in msg for msg in messages), "All messages should have content"

        print(f"   ✓ Successfully retrieved {len(conversations)} conversation(s) with messages")
        print(f"   ✓ All data structures compatible with PersonalityLearner")

        # Clean up
        manager.close()
        os.unlink(db_path)

        print("\n" + "="*60)
        print("SUCCESS: All method integration tests passed!")
        print("="*60)
        return True

    except AttributeError as e:
        print(f"\n✗ FAILED: Method not found - {e}")
        if os.path.exists(db_path):
            os.unlink(db_path)
        return False
    except AssertionError as e:
        print(f"\n✗ FAILED: Assertion error - {e}")
        if os.path.exists(db_path):
            os.unlink(db_path)
        return False
    except Exception as e:
        print(f"\n✗ FAILED: Unexpected error - {e}")
        import traceback
        traceback.print_exc()
        if os.path.exists(db_path):
            os.unlink(db_path)
        return False


if __name__ == "__main__":
    success = test_methods_exist()
    sys.exit(0 if success else 1)
