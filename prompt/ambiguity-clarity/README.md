# Ambiguity Clarity Library

A Python library for identifying and resolving ambiguous prompts in prompt engineering, providing tools to improve clarity and effectiveness of AI interactions.

## Overview

The ambiguity-clarity library provides essential tools for prompt engineering, focusing on two critical aspects:

1. **Identifying and resolving ambiguous prompts**
2. **Techniques for writing clearer prompts**

These skills are essential for effective communication with AI models and obtaining more accurate and relevant responses.

## Installation

```bash
pip install ambiguity-clarity
```

## Quick Start

```python
from ambiguity_clarity import (
    resolve_ambiguity,
    improve_prompt_clarity,
    compare_prompt_clarity,
    analyze_ambiguity
)

# Resolve ambiguous prompts
response = resolve_ambiguity("Tell me about the bank.", 
                           "You are a financial advisor discussing savings accounts.")

# Improve unclear prompts
improved = improve_prompt_clarity("What's the difference?")

# Compare prompt effectiveness
original, improved = compare_prompt_clarity("How do I make it?", 
                                          "Provide a step-by-step guide for making pizza")
```

## Core Functions 1

### `resolve_ambiguity(prompt, context)`

Resolve ambiguity in prompts by providing additional context.

**Parameters:**

- `prompt` (str): The original ambiguous prompt
- `context` (str): Additional context to resolve ambiguity

**Returns:**

- `str`: The AI's response to the clarified prompt

**Example:**

```python
ambiguous_prompt = "Tell me about the bank."
context = "You are a financial advisor discussing savings accounts."
response = resolve_ambiguity(ambiguous_prompt, context)
```

### `improve_prompt_clarity(unclear_prompt)`

Improve the clarity of a given prompt by making it more specific and actionable.

**Parameters:**

- `unclear_prompt` (str): The original unclear prompt

**Returns:**

- `str`: An improved, clearer version of the prompt

**Example:**

```python
unclear = "What's the difference?"
improved = improve_prompt_clarity(unclear)
# Returns: "What are the differences between these two concepts/objects?"
```

### `compare_prompt_clarity(original_prompt, improved_prompt)`

Compare the responses to an original prompt and an improved, clearer version.

**Parameters:**

- `original_prompt` (str): The original, potentially unclear prompt
- `improved_prompt` (str): An improved, clearer version of the prompt

**Returns:**

- `tuple`: (original_response, improved_response)

**Example:**

```python
original = "How do I make it?"
improved = "Provide a step-by-step guide for making a classic margherita pizza"
original_response, improved_response = compare_prompt_clarity(original, improved)
```

### `analyze_ambiguity(prompt)`

Analyze a prompt for ambiguity and provide detailed insights.

**Parameters:**

- `prompt` (str): The prompt to analyze

**Returns:**

- `dict`: Analysis containing ambiguity reasons and possible interpretations

## Usage Patterns

### Identifying Ambiguous Prompts

Common sources of ambiguity:

- **Vague terms**: Words like "it", "thing", "bank", "theory"
- **Missing context**: No indication of domain or perspective
- **Unclear criteria**: Terms like "best" without specifying metrics
- **Unspecified scope**: No indication of depth or breadth expected

### Resolving Ambiguity Strategies

1. **Provide Context**: Add domain-specific information
2. **Specify Parameters**: Define what "best" means in your context
3. **Clarify Scope**: Indicate expected depth and format
4. **Use Examples**: Provide concrete examples of desired output

### Writing Clearer Prompts

**Before:** "What's the difference?"
**After:** "What are the differences between supervised and unsupervised machine learning algorithms?"

**Before:** "How does it work?"
**After:** "Can you explain the process behind how a neural network learns from training data?"

**Before:** "Why is it important?"
**After:** "What is the significance of data normalization in machine learning, and how does it impact model performance?"

## Advanced Usage

### Structured Prompt Templates

Use structured templates for consistent, high-quality outputs:

```python
from ambiguity_clarity import create_structured_prompt

prompt_template = create_structured_prompt(
    topic="the impact of social media on society",
    aspects=["communication patterns", "mental health", "information spread"],
    tone="balanced and objective",
    format="sectioned_analysis"
)
```

### Batch Processing

Process multiple prompts efficiently:

```python
from ambiguity_clarity import batch_improve_clarity

unclear_prompts = [
    "What's the difference?",
    "How does it work?",
    "Why is it important?"
]

improved_prompts = batch_improve_clarity(unclear_prompts)
```

## Examples

### Example 1: Resolving Domain Ambiguity

```python
# Ambiguous prompt
prompt = "Tell me about Python."

# Different contexts yield different responses
programming_context = "You are a software engineer teaching programming concepts."
animal_context = "You are a zoologist describing reptiles."

programming_response = resolve_ambiguity(prompt, programming_context)
animal_response = resolve_ambiguity(prompt, animal_context)
```

### Example 2: Improving Technical Prompts

```python
# Original unclear prompt
unclear = "Explain the algorithm"

# Improved version
improved = improve_prompt_clarity(unclear)
# Returns: "Can you explain the step-by-step process of how the quicksort algorithm sorts an array of integers?"
```

### Example 3: Educational Content Creation

```python
# Create clear educational prompts
subject = "photosynthesis"
audience = "high school students"
depth = "conceptual understanding"

educational_prompt = f"Explain {subject} to {audience} focusing on {depth}"
clear_prompt = improve_prompt_clarity(educational_prompt)
```

## Best Practices

1. **Always specify context**: Provide background information when possible
2. **Define technical terms**: Avoid jargon unless targeting experts
3. **Use concrete examples**: Illustrate abstract concepts with specific instances
4. **Indicate desired format**: Specify if you want lists, paragraphs, code, etc.
5. **Set scope boundaries**: Indicate expected length and depth
6. **Include constraints**: Mention any limitations or requirements

## API Reference

### Core Functions 2

| Function | Description | Parameters | Returns |
|----------|-------------|------------|---------|
| `resolve_ambiguity` | Resolves ambiguous prompts with context | prompt, context | str |
| `improve_prompt_clarity` | Improves unclear prompts | unclear_prompt | str |
| `compare_prompt_clarity` | Compares original vs improved | original, improved | tuple |
| `analyze_ambiguity` | Analyzes prompt ambiguity | prompt | dict |

### Utility Functions

| Function | Description |
|----------|-------------|
| `create_structured_prompt` | Creates template-based prompts |
| `batch_improve_clarity` | Processes multiple prompts |
| `validate_prompt` | Checks prompt quality |

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

### v0.1.0

- Initial release
- Core ambiguity resolution functions
- Prompt clarity improvement tools
- Structured prompt templates
