#!/usr/bin/env python3
"""Simple test script to verify pricing resolver functionality."""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ambiguity_clarity.openai_costs import get_price
from ambiguity_clarity.openai_costs.pricing import get_model_pricing, list_available_models


def test_pricing_resolver():
    """Test the pricing resolver functionality."""
    print("Testing OpenAI Pricing Resolver")
    print("=" * 40)

    # Test individual price lookups
    test_cases = [
        ("gpt-4o", "prompt", 0.0025),
        ("gpt-4o", "completion", 0.01),
        ("gpt-4o-mini", "prompt", 0.00015),
        ("gpt-4o-mini", "completion", 0.0006),
        ("gpt-3.5-turbo", "prompt", 0.0015),
        ("o1", "prompt", 0.015),
        ("unknown-model", "prompt", None),
    ]

    print("\n1. Testing get_price function:")
    for model, token_type, expected in test_cases:
        result = get_price(model, token_type)
        status = "✅" if result == expected else "❌"
        print(f"  {status} get_price('{model}', '{token_type}') = {result} (expected: {expected})")

    # Test model pricing tuples
    print("\n2. Testing get_model_pricing function:")
    model_test_cases = [
        ("gpt-4o", (0.0025, 0.01)),
        ("gpt-4o-mini", (0.00015, 0.0006)),
        ("unknown-model", None),
    ]

    for model, expected in model_test_cases:
        result = get_model_pricing(model)
        status = "✅" if result == expected else "❌"
        print(f"  {status} get_model_pricing('{model}') = {result} (expected: {expected})")

    # Test list of available models
    print("\n3. Testing list_available_models function:")
    models = list_available_models()
    print(f"  Found {len(models)} models")
    print(f"  Sample models: {models[:5]}...")

    # Check that key models are present
    key_models = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "o1"]
    missing_models = [model for model in key_models if model not in models]
    if not missing_models:
        print("  ✅ All key models found")
    else:
        print(f"  ❌ Missing models: {missing_models}")

    print("\n4. Testing edge cases:")
    # Test invalid token type
    result = get_price("gpt-4o", "invalid_type")
    print(f"  get_price('gpt-4o', 'invalid_type') = {result} (should be None)")

    print("\nTest completed!")


if __name__ == "__main__":
    test_pricing_resolver()
