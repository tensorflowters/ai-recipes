"""LangChain integration for cost tracking using callbacks."""

from typing import Any, Dict, List
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from ambiguity_clarity.openai_costs.pricing import get_price
import time


class CostTrackingCallback(BaseCallbackHandler):
    """Callback handler that tracks OpenAI API costs using our pricing resolver."""

    def __init__(self):
        super().__init__()
        self.calls: List[Dict[str, Any]] = []
        self.total_cost = 0.0
        self.start_time = time.time()

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Called when LLM starts running."""
        # Store the prompt for later cost calculation
        self._current_prompt = prompts[0] if prompts else ""
        self._current_model = serialized.get("kwargs", {}).get("model_name", "gpt-4o-mini")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Called when LLM ends running."""
        if not hasattr(self, "_current_prompt"):
            return

        # Extract token usage from the response
        if response.llm_output and "token_usage" in response.llm_output:
            token_usage = response.llm_output["token_usage"]
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)
        else:
            # Fallback: estimate tokens from text length
            prompt_tokens = max(1, int(len(self._current_prompt.split()) * 1.33))
            completion_text = response.generations[0][0].text if response.generations else ""
            completion_tokens = max(1, int(len(completion_text.split()) * 1.33))

        # ------------------------------------------------------------------
        # Robust price lookup – handle versioned or Azure-prefixed models
        # ------------------------------------------------------------------

        def _lookup_price(model_name: str, token_type: str):
            """Try exact, then normalized, then family fallback."""
            price = get_price(model_name, token_type)
            if price is not None:
                return price

            # 1) Strip date/version suffix (e.g. gpt-4o-mini-2024-07-18)
            if "-" in model_name:
                base_try = "-".join(model_name.split("-")[:2])  # keep family and variant
                price = get_price(base_try, token_type)
                if price is not None:
                    return price

            # 2) Map by broad family key
            if model_name.startswith("gpt-4o"):
                return get_price("gpt-4o", token_type)
            if model_name.startswith("gpt-4"):
                return get_price("gpt-4", token_type)
            if model_name.startswith("gpt-3.5-turbo"):
                return get_price("gpt-3.5-turbo", token_type)

            # 3) Unknown – let caller decide
            return None

        prompt_price = _lookup_price(self._current_model, "prompt") or 0.0
        completion_price = _lookup_price(self._current_model, "completion") or 0.0

        cost = prompt_tokens / 1000 * prompt_price + completion_tokens / 1000 * completion_price

        # Record the call
        call_data = {
            "model": self._current_model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": cost,
            "timestamp": time.time(),
            "prompt": self._current_prompt[:100] + "..." if len(self._current_prompt) > 100 else self._current_prompt,
        }

        self.calls.append(call_data)
        self.total_cost += cost

        # Clean up temporary data
        delattr(self, "_current_prompt")
        delattr(self, "_current_model")

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Called when LLM errors."""
        # Clean up temporary data on error
        if hasattr(self, "_current_prompt"):
            delattr(self, "_current_prompt")
        if hasattr(self, "_current_model"):
            delattr(self, "_current_model")

    def get_summary(self) -> Dict[str, Any]:
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

    def print_summary(self):
        """Print a formatted cost summary."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("🔍 LANGCHAIN COST TRACKING SUMMARY")
        print("=" * 60)

        if summary["total_calls"] == 0:
            print("📄 No API calls tracked")
        else:
            print(f"📊 Total API calls: {summary['total_calls']}")
            print(f"💰 Total estimated cost: ${summary['total_cost']:.6f}")
            print(f"⏱️  Duration: {summary['duration_seconds']:.1f} seconds")
            print(f"🔧 Models used: {', '.join(summary['models_used'])}")
            print(f"📈 Average cost per call: ${summary['average_cost_per_call']:.6f}")
            print(f"🔤 Total prompt tokens: {summary['total_prompt_tokens']:,}")
            print(f"✍️  Total completion tokens: {summary['total_completion_tokens']:,}")

            # Show recent calls
            if len(self.calls) <= 5:
                print("\n📋 All calls:")
                for i, call in enumerate(self.calls, 1):
                    print(
                        f"  {i}. {call['model']}: ${call['cost']:.6f} ({call['prompt_tokens']}+{call['completion_tokens']} tokens)"
                    )
                    print(f"     Prompt: {call['prompt']}")
            else:
                print("\n📋 Recent calls:")
                for i, call in enumerate(self.calls[-3:], len(self.calls) - 2):
                    print(
                        f"  {i}. {call['model']}: ${call['cost']:.6f} ({call['prompt_tokens']}+{call['completion_tokens']} tokens)"
                    )

        print("=" * 60)

    def reset(self):
        """Reset the tracker."""
        self.calls = []
        self.total_cost = 0.0
        self.start_time = time.time()


# Global cost tracking callback instance
_global_cost_callback = CostTrackingCallback()


def get_global_cost_callback() -> CostTrackingCallback:
    """Get the global cost tracking callback instance."""
    return _global_cost_callback


def enable_cost_tracking():
    """Enable cost tracking by adding the callback to the global LLM."""
    import ambiguity_clarity

    # Reset the callback
    _global_cost_callback.reset()

    # Initialize callbacks list if it doesn't exist
    if not hasattr(ambiguity_clarity._llm, "callbacks") or ambiguity_clarity._llm.callbacks is None:
        ambiguity_clarity._llm.callbacks = []

    # Add callback if not already present
    if _global_cost_callback not in ambiguity_clarity._llm.callbacks:
        ambiguity_clarity._llm.callbacks.append(_global_cost_callback)

    print("✅ Cost tracking enabled")
    return _global_cost_callback


def disable_cost_tracking():
    """Disable cost tracking by removing the callback from the global LLM."""
    import ambiguity_clarity

    if hasattr(ambiguity_clarity._llm, "callbacks") and _global_cost_callback in ambiguity_clarity._llm.callbacks:
        ambiguity_clarity._llm.callbacks.remove(_global_cost_callback)
        print("🔒 Cost tracking disabled")
    else:
        print("ℹ️  Cost tracking was not enabled")


def print_cost_summary():
    """Print the current cost summary."""
    _global_cost_callback.print_summary()
    return _global_cost_callback.get_summary()
