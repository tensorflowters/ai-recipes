import ambiguity_clarity as ac
from ambiguity_clarity.openai_costs import enable_cost_tracking, disable_cost_tracking
from langchain_community.chat_models import FakeListChatModel


def test_cost_tracking_with_fake_model():
    """CostTrackingCallback should estimate cost even without real token usage."""
    # Prepare fake model with two responses
    fake_model = FakeListChatModel(responses=["Response one.", "Response two longer answer."])

    # Patch global _llm
    original_llm = ac._llm
    ac._llm = fake_model

    # Enable tracking
    cb = enable_cost_tracking()

    try:
        # Call two library functions that invoke the LLM
        ac.improve_prompt_clarity("Test prompt one")
        ac.improve_prompt_clarity("Another short prompt")
    finally:
        disable_cost_tracking()
        ac._llm = original_llm

    # Assertions
    summary = cb.get_summary()
    # Print a human-readable summary so developers can see the cost while running tests
    cb.print_summary()
    assert summary["total_calls"] == 2
    # Because we estimate tokens >0, cost should be >0
    assert summary["total_cost"] > 0
