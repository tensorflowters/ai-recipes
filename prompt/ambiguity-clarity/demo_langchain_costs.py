#!/usr/bin/env python3
"""Demo script showing LangChain-integrated cost tracking."""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ambiguity_clarity import (
    resolve_ambiguity,
    improve_prompt_clarity,
    compare_prompt_clarity,
)
from ambiguity_clarity.openai_costs import (
    enable_cost_tracking, 
    disable_cost_tracking, 
    print_cost_summary,
    get_price
)


def demo_langchain_cost_tracking():
    """Demonstrate LangChain-integrated cost tracking."""
    
    print("🚀 LangChain Cost Tracking Demo")
    print("=" * 50)
    
    # Show current pricing
    print("\n💰 Current Pricing (gpt-4o-mini):")
    prompt_price = get_price("gpt-4o-mini", "prompt")
    completion_price = get_price("gpt-4o-mini", "completion")
    print(f"  Prompt: ${prompt_price:.6f} per 1K tokens")
    print(f"  Completion: ${completion_price:.6f} per 1K tokens")
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ OPENAI_API_KEY not set. Using mock responses for demo.")
        print("Set your API key with: export OPENAI_API_KEY='your-key-here'")
        print("\n📝 Running with estimated costs based on text length...")
    else:
        print("\n✅ API key found. Will track real costs.")
    
    # Enable cost tracking
    print("\n🔍 Enabling cost tracking...")
    cost_callback = enable_cost_tracking()
    
    try:
        # Test some functions
        print("\n📝 Testing improve_prompt_clarity...")
        unclear_prompt = "Explain it"
        improved = improve_prompt_clarity(unclear_prompt)
        print(f"  Original: '{unclear_prompt}'")
        print(f"  Improved: '{improved[:80]}...'")
        
        print("\n🔍 Testing resolve_ambiguity...")
        ambiguous = "Tell me about the bank"
        context = "You are a financial advisor"
        resolved = resolve_ambiguity(ambiguous, context)
        print(f"  Prompt: '{ambiguous}'")
        print(f"  Context: '{context}'")
        print(f"  Response length: {len(resolved)} characters")
        
        print("\n⚖️  Testing compare_prompt_clarity...")
        original = "How does it work?"
        improved_prompt = "How does a car engine work?"
        orig_resp, imp_resp = compare_prompt_clarity(original, improved_prompt)
        print(f"  Original prompt: '{original}' → {len(orig_resp)} chars")
        print(f"  Improved prompt: '{improved_prompt}' → {len(imp_resp)} chars")
        
    except Exception as e:
        print(f"⚠️  Error during API calls: {e}")
        print("This is expected if no API key is set.")
    
    # Print cost summary
    print("\n📊 Final Cost Summary:")
    print_cost_summary()
    
    # Disable cost tracking
    print("\n🔒 Disabling cost tracking...")
    disable_cost_tracking()
    
    # Show some additional cost estimates
    print("\n💡 Example cost estimates:")
    example_prompts = [
        "Write a short story about AI",
        "Explain quantum computing in simple terms",
        "Create a Python function to sort a list"
    ]
    
    for prompt in example_prompts:
        # Estimate prompt tokens (rough)
        prompt_tokens = len(prompt.split()) * 1.33
        # Assume typical response length
        completion_tokens = 150 * 1.33
        
        cost = (prompt_tokens / 1000 * prompt_price + 
               completion_tokens / 1000 * completion_price)
        
        print(f"  '{prompt}' → ~${cost:.6f}")
    
    print("\n✨ Demo completed!")


if __name__ == "__main__":
    demo_langchain_cost_tracking()
