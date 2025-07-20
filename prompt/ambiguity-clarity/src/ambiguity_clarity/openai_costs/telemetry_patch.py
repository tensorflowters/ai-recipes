"""OpenAI Chat Completions telemetry monkey-patch.

This module provides monkey-patching functionality for the OpenAI client
to capture telemetry data including latency, usage, and costs.
"""

import time
from threading import Lock
from typing import Dict, Any, List
import functools
from ambiguity_clarity.openai_costs.pricing import get_price

# Global registry and lock for thread-safe updates
telemetry_registry: List[Dict[str, Any]] = []
telemetry_lock = Lock()


def log_telemetry(data: Dict[str, Any]) -> None:
    """Log telemetry data in a thread-safe manner."""
    with telemetry_lock:
        telemetry_registry.append(data)


def get_telemetry_registry() -> List[Dict[str, Any]]:
    """Get a copy of the current telemetry registry."""
    with telemetry_lock:
        return telemetry_registry.copy()


def clear_telemetry_registry() -> None:
    """Clear the telemetry registry."""
    with telemetry_lock:
        telemetry_registry.clear()


def calculate_cost(usage: Dict[str, Any], model: str) -> float:
    """Calculate the cost of the API call based on usage and model pricing."""
    prompt_price = get_price(model, "prompt")
    completion_price = get_price(model, "completion")
    return usage.get("prompt_tokens", 0) / 1000 * (prompt_price or 0) + usage.get("completion_tokens", 0) / 1000 * (
        completion_price or 0
    )


def patch_openai_completions():
    """Apply the telemetry patch to OpenAI chat completions."""
    try:
        from openai.resources.chat.completions import Completions
    except ImportError:
        print("Warning: Could not import OpenAI Completions class for patching")
        return

    # Store original create method
    original_create = Completions.create

    @functools.wraps(original_create)
    def patched_create(self, *args, **kwargs):
        """Patched version of Completions.create with telemetry."""
        # Extract custom tracking parameters before API call
        test_nodeid = kwargs.pop("test_nodeid", "unknown")

        start_time = time.time()
        response = original_create(self, *args, **kwargs)
        end_time = time.time()

        # Extract usage information from response
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        else:
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        latency_ms = int((end_time - start_time) * 1000)
        model = kwargs.get("model", "unknown")
        cost = calculate_cost(usage, model)

        telemetry_data = {
            "test_nodeid": test_nodeid,
            "model": model,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "cost_$": cost,
            "latency_ms": latency_ms,
            "timestamp": time.time(),
        }
        log_telemetry(telemetry_data)

        return response

    # Apply the monkey patch
    Completions.create = patched_create
    print("✅ OpenAI Completions telemetry patch applied")


# Auto-apply patch when module is imported
if __name__ != "__main__":
    patch_openai_completions()
