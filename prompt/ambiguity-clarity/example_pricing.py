#!/usr/bin/env python3
"""Example usage of the pricing resolver with configuration."""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ambiguity_clarity.openai_costs.pricing import (
    get_price,
    get_model_pricing,
    list_available_models,
)


def demonstrate_pricing_resolver():
    """Demonstrate the pricing resolver functionality."""
    print("OpenAI Pricing Resolver Demo")
    print("=" * 40)

    # Basic usage
    print("\n1. Basic pricing lookup:")
    models_to_check = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "o1"]

    for model in models_to_check:
        prompt_price = get_price(model, "prompt")
        completion_price = get_price(model, "completion")
        print(f"  {model}:")
        print(f"    Prompt: ${prompt_price:.6f} per 1K tokens")
        print(f"    Completion: ${completion_price:.6f} per 1K tokens")

        # Calculate cost for a typical request
        if prompt_price and completion_price:
            prompt_tokens = 1000  # Example: 1K prompt tokens
            completion_tokens = 500  # Example: 500 completion tokens
            total_cost = prompt_tokens / 1000 * prompt_price + completion_tokens / 1000 * completion_price
            print(f"    Example cost (1K prompt + 500 completion): ${total_cost:.4f}")
        print()

    # Demonstrate model comparison
    print("\n2. Model cost comparison (for 10K prompt + 2K completion tokens):")
    print("  Model               | Prompt Cost | Completion Cost | Total Cost")
    print("  -------------------|-------------|-----------------|----------")

    for model in models_to_check:
        pricing = get_model_pricing(model)
        if pricing:
            prompt_price, completion_price = pricing
            prompt_cost = 10 * prompt_price  # 10K tokens
            completion_cost = 2 * completion_price  # 2K tokens
            total_cost = prompt_cost + completion_cost
            print(f"  {model:<18} | ${prompt_cost:>9.4f} | ${completion_cost:>13.4f} | ${total_cost:>8.4f}")

    # Show all available models
    print("\n3. All available models:")
    all_models = list_available_models()
    print(f"  Total models: {len(all_models)}")

    # Group by model family
    model_families = {
        "GPT-4o": [m for m in all_models if m.startswith("gpt-4o")],
        "GPT-4": [m for m in all_models if m.startswith("gpt-4") and not m.startswith("gpt-4o")],
        "GPT-3.5": [m for m in all_models if m.startswith("gpt-3.5")],
        "o1": [m for m in all_models if m.startswith("o1")],
        "Embedding": [m for m in all_models if "embedding" in m],
        "Audio": [m for m in all_models if m in ["whisper-1", "tts-1", "tts-1-hd"]],
        "Other": [
            m
            for m in all_models
            if not any(m.startswith(prefix) for prefix in ["gpt-", "o1", "text-embedding"])
            and m not in ["whisper-1", "tts-1", "tts-1-hd"]
        ],
    }

    for family, models in model_families.items():
        if models:
            print(f"  {family}: {len(models)} models")
            for model in sorted(models):
                pricing = get_model_pricing(model)
                if pricing:
                    prompt_price, completion_price = pricing
                    if completion_price > 0:
                        print(f"    {model}: ${prompt_price:.6f}/${completion_price:.6f}")
                    else:
                        print(f"    {model}: ${prompt_price:.6f} (no completion cost)")

    # Demonstrate configuration
    print("\n4. Configuration options:")
    print("  Current configuration uses static pricing with optional live updates")
    print("  To enable live pricing from a custom endpoint:")
    print("    configure_pricing_resolver(")
    print("        cache_duration_hours=12,")
    print("        pricing_url='https://your-internal-api.com/pricing.json',")
    print("        enable_live_pricing=True")
    print("    )")
    print("  Note: Live pricing requires httpx: pip install httpx")

    print("\n5. Cache information:")
    from pathlib import Path

    cache_file = Path.home() / ".cache" / "ambiguity_clarity" / "pricing_cache.json"
    if cache_file.exists():
        import time

        cache_age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        print(f"  Cache file: {cache_file}")
        print(f"  Cache age: {cache_age_hours:.1f} hours")
        print(f"  Cache size: {cache_file.stat().st_size} bytes")
    else:
        print("  No cache file found (using static pricing only)")


if __name__ == "__main__":
    demonstrate_pricing_resolver()
