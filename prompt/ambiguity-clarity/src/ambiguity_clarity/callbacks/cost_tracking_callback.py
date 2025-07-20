"""Callback handler for tracking API call costs."""

from langchain.callbacks.base import BaseCallbackHandler
from ambiguity_clarity.openai_costs import get_price

class CostTrackingCallback(BaseCallbackHandler):
    def __init__(self):
        self.total_cost = 0.0
        self.calls = []

    def on_llm_end(self, details):
        # Assume details contain model name and token counts
        model = details.get('model', 'gpt-4o-mini')
        tokens = details.get('tokens', {})
        prompt_tokens = tokens.get('prompt', 0)
        completion_tokens = tokens.get('completion', 0)

        prompt_price = get_price(model, 'prompt')
        completion_price = get_price(model, 'completion')

        if prompt_price and completion_price:
            cost = (prompt_tokens / 1000 * prompt_price) + (completion_tokens / 1000 * completion_price)
            self.total_cost += cost
            self.calls.append({
                'model': model,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'cost': cost
            })

    def print_summary(self):
        print("\n🔍 API Call Summary")
        print(f"Total API calls: {len(self.calls)}")
        print(f"Total cost: ${self.total_cost:.6f}")
        for i, call in enumerate(self.calls, 1):
            print(f"Call {i}: Model={call['model']}, Cost=${call['cost']:.6f}")
