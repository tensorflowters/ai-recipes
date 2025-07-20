#!/usr/bin/env python3
"""Demo script to show cost tracking with real API calls."""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ambiguity_clarity import (
    improve_prompt_clarity,
    compare_prompt_clarity,
    create_structured_prompt,
)
from ambiguity_clarity.openai_costs import get_price
from tests.conftest import CostTracker, estimate_prompt_cost


def demo_cost_tracking_with_real_api():
    """Demonstrate cost tracking with real API calls."""

    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set. Skipping real API cost demo.")
        print("Set your API key with: export OPENAI_API_KEY='your-key-here'")
        return

    print("🚀 OpenAI Cost Tracking Demo with Real API Calls")
    print("=" * 60)

    # Create a cost tracker for this demo
    tracker = CostTracker()

    # Test prompts
    test_prompts = ["Explain it", "What's the difference?", "Tell me about machine learning"]

    print("\n1. Cost estimates before API calls:")
    for prompt in test_prompts:
        estimate = estimate_prompt_cost(prompt, "gpt-4o-mini", 150)
        print(f"  '{prompt}' → Estimated: ${estimate:.6f}")

    print("\n2. Making real API calls and tracking costs...")

    # Patch the LLM to track calls (simplified version)
    import ambiguity_clarity

    original_invoke = ambiguity_clarity._llm.invoke

    def track_invoke(prompt):
        # Make the real API call
        result = original_invoke(prompt)

        # Estimate tokens (in real implementation, you'd get this from the API response)
        prompt_text = str(prompt)
        prompt_tokens = max(1, int(len(prompt_text.split()) * 1.33))
        completion_tokens = max(1, int(len(result.content.split()) * 1.33))

        # Track the call
        tracker.add_call(
            model="gpt-4o-mini",  # Default model
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        return result

    # Apply the tracking patch
    ambiguity_clarity._llm.invoke = track_invoke

    try:
        # Test 1: Improve prompt clarity
        print("\n   Testing improve_prompt_clarity...")
        improved = improve_prompt_clarity(test_prompts[0])
        print(f"   Original: '{test_prompts[0]}'")
        print(f"   Improved: '{improved[:80]}...'")

        # Test 2: Compare prompts
        print("\n   Testing compare_prompt_clarity...")
        original_resp, improved_resp = compare_prompt_clarity(
            test_prompts[1], "What are the key differences between Python and JavaScript?"
        )
        print(f"   Compared '{test_prompts[1]}' vs improved version")
        print(f"   Original response length: {len(original_resp)} chars")
        print(f"   Improved response length: {len(improved_resp)} chars")

        # Test 3: Structured prompt
        print("\n   Testing create_structured_prompt...")
        structured = create_structured_prompt(
            topic="artificial intelligence", aspects=["applications", "challenges", "future"], tone="educational"
        )
        print(f"   Generated structured response: {len(structured)} chars")

    finally:
        # Restore original invoke method
        ambiguity_clarity._llm.invoke = original_invoke

    # Print cost summary
    summary = tracker.get_summary()
    print("\n" + "=" * 60)
    print("💰 REAL API COST SUMMARY")
    print("=" * 60)

    if summary["total_calls"] == 0:
        print("No calls were tracked (something went wrong)")
    else:
        print(f"📊 Total API calls: {summary['total_calls']}")
        print(f"💰 Total cost: ${summary['total_cost']:.6f}")
        print(f"⏱️  Duration: {summary['duration_seconds']:.1f} seconds")
        print(f"🔤 Total prompt tokens: {summary['total_prompt_tokens']:,}")
        print(f"✍️  Total completion tokens: {summary['total_completion_tokens']:,}")
        print(f"📈 Average cost per call: ${summary['average_cost_per_call']:.6f}")

        # Show individual calls
        print("\n📋 Individual call breakdown:")
        for i, call in enumerate(summary["calls"], 1):
            print(
                f"  Call {i}: {call['prompt_tokens']:,} prompt + {call['completion_tokens']:,} completion = ${call['cost']:.6f}"
            )

    print("=" * 60)

    # Show pricing information
    print("\n💡 Current Pricing (gpt-4o-mini):")
    prompt_price = get_price("gpt-4o-mini", "prompt")
    completion_price = get_price("gpt-4o-mini", "completion")
    print(f"  Prompt tokens: ${prompt_price:.6f} per 1K tokens")
    print(f"  Completion tokens: ${completion_price:.6f} per 1K tokens")


if __name__ == "__main__":
    demo_cost_tracking_with_real_api()
