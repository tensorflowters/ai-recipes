"""Test fixtures and utilities for cost tracking during test execution."""

import pytest
from typing import Dict, List, Optional
from unittest.mock import MagicMock
import time
from ambiguity_clarity.openai_costs import get_price
from ambiguity_clarity.openai_costs import (
    enable_cost_tracking,
    disable_cost_tracking,
    print_cost_summary,
)


class CostTracker:
    """Tracks OpenAI API costs during test execution."""

    def __init__(self):
        self.calls: List[Dict] = []
        self.total_cost = 0.0
        self.start_time = time.time()

    def add_call(self, model: str, prompt_tokens: int, completion_tokens: int, cost: Optional[float] = None):
        """Add a tracked API call."""
        if cost is None:
            prompt_price = get_price(model, "prompt")
            completion_price = get_price(model, "completion")

            if prompt_price is not None and completion_price is not None:
                cost = prompt_tokens / 1000 * prompt_price + completion_tokens / 1000 * completion_price
            else:
                cost = 0.0  # Unknown model

        call_data = {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": cost,
            "timestamp": time.time(),
        }

        self.calls.append(call_data)
        self.total_cost += cost

    def get_summary(self) -> Dict:
        """Get a summary of tracked costs."""
        if not self.calls:
            return {
                "total_calls": 0,
                "total_cost": 0.0,
                "duration_seconds": time.time() - self.start_time,
                "models_used": [],
                "average_cost_per_call": 0.0,
            }

        models_used = list(set(call["model"] for call in self.calls))
        total_prompt_tokens = sum(call["prompt_tokens"] for call in self.calls)
        total_completion_tokens = sum(call["completion_tokens"] for call in self.calls)

        return {
            "total_calls": len(self.calls),
            "total_cost": self.total_cost,
            "duration_seconds": time.time() - self.start_time,
            "models_used": models_used,
            "average_cost_per_call": self.total_cost / len(self.calls),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "calls": self.calls,
        }

    def reset(self):
        """Reset the tracker."""
        self.calls = []
        self.total_cost = 0.0
        self.start_time = time.time()


# Global cost tracker instance
_cost_tracker = CostTracker()


def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker instance."""
    return _cost_tracker


def pytest_sessionfinish(session, exitstatus):
    """Called after whole test run finishes, before returning exit status."""
    summary = _cost_tracker.get_summary()

    # ------------------------------------------------------------------
    # If our legacy _cost_tracker saw no calls, but the newer LangChain
    # CostTrackingCallback (enabled via enable_cost_tracking()) *did*, merge
    # that data so we always surface a correct cost receipt at session end.
    # ------------------------------------------------------------------
    if summary["total_calls"] == 0:
        try:
            from ambiguity_clarity.openai_costs import get_global_cost_callback

            cb_summary = get_global_cost_callback().get_summary()

            if cb_summary["total_calls"] > 0:
                summary = cb_summary
        except Exception:  # noqa: BLE001
            # Fallback if LangChain callback not available
            pass

    # Always print a separator
    print("\n" + "=" * 70)
    print("🔍 OPENAI COST SUMMARY")
    print("=" * 70)

    if summary["total_calls"] == 0:
        print("📄 No API calls tracked (tests used mocked responses)")
        print("💡 To see real costs, run integration tests with: pytest -m integration")
    else:
        print(f"📊 Total API calls: {summary['total_calls']}")
        print(f"💰 Total estimated cost: ${summary['total_cost']:.6f}")
        print(f"⏱️  Test duration: {summary['duration_seconds']:.1f} seconds")
        print(f"🔧 Models used: {', '.join(summary['models_used'])}")
        print(f"📈 Average cost per call: ${summary['average_cost_per_call']:.6f}")
        print(f"🔤 Total prompt tokens: {summary['total_prompt_tokens']:,}")
        print(f"✍️  Total completion tokens: {summary['total_completion_tokens']:,}")

        # Cost breakdown by model
        model_costs = {}
        for call in summary["calls"]:
            model = call["model"]
            if model not in model_costs:
                model_costs[model] = {"cost": 0.0, "calls": 0, "tokens": 0}
            model_costs[model]["cost"] += call["cost"]
            model_costs[model]["calls"] += 1
            model_costs[model]["tokens"] += call["prompt_tokens"] + call["completion_tokens"]

        if len(model_costs) > 1:
            print("\n📋 Cost breakdown by model:")
            for model, stats in sorted(model_costs.items(), key=lambda x: x[1]["cost"], reverse=True):
                print(f"  {model}: ${stats['cost']:.6f} ({stats['calls']} calls, {stats['tokens']:,} tokens)")

        # Efficiency metrics
        if summary["duration_seconds"] > 0:
            calls_per_second = summary["total_calls"] / summary["duration_seconds"]
            cost_per_second = summary["total_cost"] / summary["duration_seconds"]
            print("\n⚡ Performance metrics:")
            print(f"  Calls per second: {calls_per_second:.2f}")
            print(f"  Cost per second: ${cost_per_second:.6f}")

    print("=" * 70)


@pytest.fixture(scope="session", autouse=True)
def cost_tracking_session():
    """Session-wide fixture that tracks costs."""
    _cost_tracker.reset()
    yield _cost_tracker


@pytest.fixture
def mock_llm_with_cost_tracking(mocker):
    """Mock LLM that tracks costs automatically."""

    def mock_invoke(prompt):
        # Estimate token counts (rough approximation)
        prompt_text = (
            str(prompt) if hasattr(prompt, "__str__") else str(prompt.content if hasattr(prompt, "content") else prompt)
        )
        prompt_tokens = max(1, len(prompt_text.split()) * 1.3)  # Rough estimate
        completion_tokens = max(1, len("Mock response".split()) * 1.3)

        # Track the cost assuming gpt-4o-mini (the default model)
        _cost_tracker.add_call(model="gpt-4o-mini", prompt_tokens=int(prompt_tokens), completion_tokens=int(completion_tokens))

        mock_response = MagicMock()
        mock_response.content = "Mock response from cost-tracking LLM"
        return mock_response

    mock_llm = MagicMock()
    mock_llm.invoke = mock_invoke

    mocker.patch("ambiguity_clarity._llm", new=mock_llm)
    return mock_llm


@pytest.fixture
def integration_cost_tracking():
    """Fixture for integration tests that tracks real API costs."""
    # Reset tracker for this specific integration test
    tracker = CostTracker()

    # Patch the ChatOpenAI class to track real API calls
    original_invoke = None

    def track_real_call(self, prompt):
        result = original_invoke(prompt)

        # Extract token counts from the actual response
        # Note: In real usage, you'd get these from the OpenAI response
        # For now, we'll estimate
        prompt_text = str(prompt)
        prompt_tokens = max(1, len(prompt_text.split()) * 1.3)
        completion_tokens = max(1, len(result.content.split()) * 1.3)

        # Get model name from the ChatOpenAI instance
        model_name = getattr(self, "model_name", "gpt-4o-mini")

        tracker.add_call(model=model_name, prompt_tokens=int(prompt_tokens), completion_tokens=int(completion_tokens))

        return result

    # Monkey patch for integration tests
    import ambiguity_clarity

    if hasattr(ambiguity_clarity._llm, "invoke"):
        original_invoke = ambiguity_clarity._llm.invoke
        ambiguity_clarity._llm.invoke = lambda prompt: track_real_call(ambiguity_clarity._llm, prompt)

    yield tracker

    # Restore original method
    if original_invoke:
        ambiguity_clarity._llm.invoke = original_invoke


def estimate_prompt_cost(prompt: str, model: str = "gpt-4o-mini", response_length: int = 100) -> float:
    """Estimate the cost of a prompt before making the API call."""
    # Rough token estimation (1 token ≈ 0.75 words)
    prompt_tokens = len(prompt.split()) * 1.33
    completion_tokens = response_length * 1.33

    prompt_price = get_price(model, "prompt")
    completion_price = get_price(model, "completion")

    if prompt_price is None or completion_price is None:
        return 0.0

    return prompt_tokens / 1000 * prompt_price + completion_tokens / 1000 * completion_price


def print_cost_estimate(prompt: str, model: str = "gpt-4o-mini", response_length: int = 100):
    """Print a cost estimate for a given prompt."""
    cost = estimate_prompt_cost(prompt, model, response_length)
    print(f"💰 Estimated cost for '{prompt[:50]}...' on {model}: ${cost:.6f}")


@pytest.fixture(autouse=True)
def _integration_cost_tracker(request):
    """Enable cost tracking automatically for tests marked with `integration`."""
    if request.node.get_closest_marker("integration") is None:
        # Not an integration test
        yield
        return

    # Integration test: attach callback
    enable_cost_tracking()
    try:
        yield
    finally:
        # Print summary once test finishes and detach callback
        print_cost_summary()
        disable_cost_tracking()
