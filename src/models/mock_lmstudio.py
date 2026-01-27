"""Mock lmstudio module for testing without dependencies."""


class Client:
    """Mock LM Studio client."""

    def close(self):
        pass

    class llm:
        """Mock LLM interface."""

        @staticmethod
        def list_downloaded_models():
            """Return empty list for testing."""
            return []

        @staticmethod
        def model(model_key):
            """Return mock model."""
            return MockModel(model_key)


class MockModel:
    """Mock model for testing."""

    def __init__(self, model_key):
        self.model_key = model_key
        self.display_name = model_key
        self.context_length = 4096

    def respond(self, prompt, max_tokens=100):
        """Return mock response."""
        return "mock response"
