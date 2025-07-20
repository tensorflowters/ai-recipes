
from langchain_community.chat_models import FakeListChatModel

import ambiguity_clarity as ac


def test_reload_llm_respects_env(monkeypatch):
    """Changing OPENAI_API_KEY and calling reload_llm should swap the global model."""
    # Ensure a dummy key is not already set
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # Patch ChatOpenAI with FakeListChatModel so no real network is used
    monkeypatch.setattr(ac, "ChatOpenAI", FakeListChatModel)

    # Set env var and reload llm
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy-123")
    ac.reload_llm(llm_class=FakeListChatModel, responses=["hi"])

    # Global _llm should now be instance of FakeListChatModel
    assert isinstance(ac._llm, FakeListChatModel)

    # Invoke a library function and ensure it returns the fake response
    result = ac.improve_prompt_clarity("test")
    assert result == "hi"

    # Cost tracking still works
    from ambiguity_clarity.openai_costs import enable_cost_tracking, disable_cost_tracking, get_global_cost_callback

    enable_cost_tracking()
    try:
        _ = ac.improve_prompt_clarity("another call")
    finally:
        summary = get_global_cost_callback().get_summary()
        disable_cost_tracking()

    assert summary["total_calls"] >= 1
    assert summary["total_cost"] >= 0.0 
