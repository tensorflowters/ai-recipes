# API Documentation

## Core Functions

### resolve_ambiguity(prompt, context)

Resolve ambiguous prompts by providing context.

**Parameters:**

- `prompt` (str): Original ambiguous prompt
- `context` (str): Additional context to resolve ambiguity

**Returns:**

- `str`: AI response to clarified prompt

**Example:**

```python
response = resolve_ambiguity("Tell me about Python", "You are teaching programming")
```

### improve_prompt_clarity(unclear_prompt)

Improve clarity of unclear prompts.

**Parameters:**

- `unclear_prompt` (str): Unclear prompt to improve

**Returns:**

- `str`: Improved, clearer prompt

**Example:**

```python
improved = improve_prompt_clarity("What's the difference?")
# Returns: "What are the differences between these two concepts/objects?"
```

### compare_prompt_clarity(original, improved)

Compare original vs improved prompt responses.

**Parameters:**

- `original` (str): Original prompt
- `improved` (str): Improved prompt

**Returns:**

- `tuple`: (original_response, improved_response)

### analyze_ambiguity(prompt)

Analyze prompt for ambiguity.

**Parameters:**

- `prompt` (str): Prompt to analyze

**Returns:**

- `dict`: Analysis with reasons and interpretations

### create_structured_prompt(topic, aspects, tone="neutral")

Create structured prompts for consistent outputs.

**Parameters:**

- `topic` (str): Main topic
- `aspects` (list): Specific aspects to cover
- `tone` (str): Desired tone (default: "neutral")

**Example:**

```python
prompt = create_structured_prompt(
    topic="machine learning",
    aspects=["definition", "applications", "challenges"],
    tone="educational"
)
```

### batch_improve_clarity(unclear_prompts)

Process multiple prompts in batch.

**Parameters:**

- `unclear_prompts` (list): List of unclear prompts

**Returns:**

- `list`: List of improved prompts
