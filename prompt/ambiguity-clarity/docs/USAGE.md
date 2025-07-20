# Usage Guide

## Quick Start Examples

### 1. Resolving Ambiguous Prompts

```python
from ambiguity_clarity import resolve_ambiguity

# Example: Banking ambiguity
ambiguous = "Tell me about the bank."
contexts = [
    "You are a financial advisor discussing investment strategies.",
    "You are a geographer explaining river formations."
]

for context in contexts:
    response = resolve_ambiguity(ambiguous, context)
    print(f"Context: {context}")
    print(f"Response: {response[:200]}...")
```

### 2. Improving Prompt Clarity

```python
from ambiguity_clarity import improve_prompt_clarity

unclear = "What's the difference?"
improved = improve_prompt_clarity(unclear)
print(f"❌ Unclear: {unclear}")
print(f"✅ Clear: {improved}")
```

### 3. Batch Processing

```python
from ambiguity_clarity import batch_improve_clarity

unclear_prompts = [
    "What's the difference?",
    "How does it work?",
    "Why is it important?"
]

improved = batch_improve_clarity(unclear_prompts)
for original, better in zip(unclear_prompts, improved):
    print(f"{original} → {better}")
```

## Real-World Scenarios

### Customer Support

```python
from ambiguity_clarity import resolve_ambiguity

vague_query = "It's not working"
context = "You are a technical support agent for a mobile banking app."
response = resolve_ambiguity(vague_query, context)
```

### Content Creation

```python
from ambiguity_clarity import create_structured_prompt

blog_prompt = create_structured_prompt(
    topic="artificial intelligence for beginners",
    aspects=["definition", "benefits", "getting started", "common mistakes"],
    tone="friendly and informative"
)
```
