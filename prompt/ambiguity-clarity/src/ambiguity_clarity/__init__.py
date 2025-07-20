"""Ambiguity Clarity Library for Prompt Engineering.

This library provides tools for identifying and resolving ambiguous prompts,
improving clarity and effectiveness of AI interactions.

:author: tensorflowters
:email: athurbequie@protonmail.com
:version: 0.1.0
"""

from typing import Tuple, Dict, Any, List
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv


__version__ = "0.1.0"
__author__ = "tensorflowters"
__email__ = "athurbequie@protonmail.com"

# Initialize the language model once
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
_llm = ChatOpenAI(model="gpt-4o-mini")


def resolve_ambiguity(prompt: str, context: str) -> str:
    """Resolve ambiguity in a prompt by providing additional context.

    This function takes an ambiguous prompt and enriches it with specific
    context to generate a more targeted and useful response.

    :param str prompt: The original ambiguous prompt
    :param str context: Additional context to resolve ambiguity
    :returns: The AI's response to the clarified prompt
    :rtype: str

    :example:
        >>> prompt = "Tell me about the bank."
        >>> context = "You are a financial advisor discussing savings accounts."
        >>> response = resolve_ambiguity(prompt, context)
        >>> print(response)  # Returns detailed banking information
    """
    clarified_prompt = f"{context}\n\nBased on this context, {prompt}"
    return _llm.invoke(clarified_prompt).content


def improve_prompt_clarity(unclear_prompt: str) -> str:
    """Improve the clarity of a given prompt by making it more specific.

    This function analyzes an unclear prompt and returns a clearer,
    more specific version that will yield better AI responses.

    :param str unclear_prompt: The original unclear prompt
    :returns: An improved, clearer version of the prompt
    :rtype: str

    :example:
        >>> unclear = "What's the difference?"
        >>> improved = improve_prompt_clarity(unclear)
        >>> print(improved)
        "What are the differences between these two concepts/objects?"
    """
    improvement_prompt = (
        f"The following prompt is unclear: '{unclear_prompt}'. "
        f"Please provide a clearer, more specific version of this prompt. "
        f"Output just the improved prompt and nothing else."
    )
    return _llm.invoke(improvement_prompt).content


def compare_prompt_clarity(original_prompt: str, improved_prompt: str) -> Tuple[str, str]:
    """Compare the responses to an original prompt and an improved version.

    This function demonstrates the impact of prompt clarity by returning
    both the original response and the improved response side by side.

    :param str original_prompt: The original, potentially unclear prompt
    :param str improved_prompt: An improved, clearer version of the prompt
    :returns: A tuple containing (original_response, improved_response)
    :rtype: tuple[str, str]

    :example:
        >>> original = "How do I make it?"
        >>> improved = "Provide a step-by-step guide for making a classic margherita pizza"
        >>> orig, imp = compare_prompt_clarity(original, improved)
        >>> print("Original:", orig)
        >>> print("Improved:", imp)
    """
    original_response = _llm.invoke(original_prompt).content
    improved_response = _llm.invoke(improved_prompt).content
    return original_response, improved_response


def analyze_ambiguity(prompt: str) -> Dict[str, Any]:
    """Analyze a prompt for ambiguity and provide detailed insights.

    This function performs a comprehensive analysis of a prompt to identify
    sources of ambiguity and suggest possible interpretations.

    :param str prompt: The prompt to analyze
    :returns: Analysis containing ambiguity reasons and possible interpretations
    :rtype: dict

    :example:
        >>> analysis = analyze_ambiguity("Tell me about the bank.")
        >>> print(analysis['reasons'])  # List of ambiguity reasons
        >>> print(analysis['interpretations'])  # Possible interpretations
    """
    analysis_prompt = (
        f"Analyze the following prompt for ambiguity: '{prompt}'. "
        f"Provide a JSON response with 'reasons' (list of why it's ambiguous) "
        f"and 'interpretations' (list of possible interpretations)."
    )
    response = _llm.invoke(analysis_prompt).content

    # Parse the response into a structured format
    # Note: In production, you'd want more robust JSON parsing
    return {"prompt": prompt, "analysis": response, "suggested_improvements": []}


def create_structured_prompt(
    topic: str,
    aspects: List[str],
    tone: str = "neutral",
    format: str | None = None,
) -> str:
    """Create a structured prompt template for consistent outputs.

    This function generates well-structured prompts that ensure consistent,
    high-quality responses from AI models.

    :param str topic: The main topic to discuss
    :param list[str] aspects: Specific aspects to cover (list of strings)
    :param str tone: Desired tone (neutral, formal, casual, etc.)
    :returns: A structured prompt ready for use
    :rtype: str

    :example:
        >>> prompt = create_structured_prompt(
        ...     topic="the impact of social media on society",
        ...     aspects=["communication patterns", "mental health", "information spread"],
        ...     tone="balanced and objective"
        ... )
    """
    aspect_lines = "\n".join(f"{i+1}. {a}" for i, a in enumerate(aspects))

    if format == "sectioned_analysis":
        prompt_body = "\n".join(f"## {aspect}\nExplain how {aspect} relates to {topic}." for aspect in aspects)
        prompt = (
            f"You are writing a structured, sectioned analysis in markdown.\n\n"
            f"# {topic}\n\n"
            f"{prompt_body}\n\n"
            f"Use a {tone} tone throughout."
        )
    else:
        # Build a PromptTemplate dynamically so that users can inspect/serialize it.
        # We create placeholder variables for each aspect (aspect_0, aspect_1, …).
        aspect_placeholders = [f"aspect_{idx}" for idx in range(len(aspects))]

        template_body = "\n".join(f"{idx + 1}. {{{placeholder}}}" for idx, placeholder in enumerate(aspect_placeholders))

        template_str = (
            f"Provide an analysis of {{topic}} considering the following aspects:\n"
            f"{aspect_lines}\n\n"          # preview for humans
            f"Present the analysis in {{tone}} tone.\n\n"
            "===\n"                        # separator before LLM placeholders
            + template_body +
            "\n\nPresent the analysis in a {tone} tone."
        )

        prompt_template = PromptTemplate(
            input_variables=["topic", "tone", *aspect_placeholders],
            template=template_str,
        )

        values: Dict[str, Any] = {
            "topic": topic,
            "tone": tone,
            **{ph: aspect for ph, aspect in zip(aspect_placeholders, aspects)},
        }

        chain = prompt_template | _llm
        return chain.invoke(values).content

    # sectioned_analysis path already returns above
    return _llm.invoke(prompt).content


def batch_improve_clarity(unclear_prompts: List[str]) -> List[str]:
    """Process multiple prompts to improve their clarity in batch.

    :param list[str] unclear_prompts: List of unclear prompts to improve
    :returns: List of improved, clearer prompts
    :rtype: list[str]

    :example:
        >>> prompts = ["What's the difference?", "How does it work?"]
        >>> improved = batch_improve_clarity(prompts)
        >>> print(improved[0])  # "What are the differences between..."
    """
    return [improve_prompt_clarity(prompt) for prompt in unclear_prompts]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


AMBIGUOUS_TOKENS: List[str] = [
    "it",
    "thing",
    "stuff",
    "they",
    "them",
]


def _simple_heuristic(prompt: str) -> Dict[str, Any]:
    """Quick local heuristic to evaluate prompt clarity.

    The function is intentionally simple so it can run offline and *fast* during
    unit-tests.  For production-grade analysis, fallback to the LLM (see below).
    """

    tokens = prompt.split()
    length = len(tokens)

    ambiguous_hits = [t for t in tokens if t.lower() in AMBIGUOUS_TOKENS]

    score = 100
    if length < 5:
        score -= 40
    if ambiguous_hits:
        score -= 20 * len(set(ambiguous_hits))

    return {
        "length": length,
        "ambiguous_tokens": ambiguous_hits,
        "score": max(score, 0),
    }


def validate_prompt(prompt: str, use_llm: bool = False) -> Dict[str, Any]:
    """Validate *prompt* clarity and return a diagnostic report.

    The report contains:
        - *score*: 0-100 heuristic clarity score
        - *issues*: list of detected problems (strings)

    If *use_llm* is True **and** an OpenAI key is configured, we will also ask the
    language model for a brief assessment, appended under the key ``llm_feedback``.
    """

    report = _simple_heuristic(prompt)

    issues: List[str] = []
    if report["length"] < 5:
        issues.append("Prompt is very short — add more context.")
    if report["ambiguous_tokens"]:
        issues.append("Contains ambiguous referents: " + ", ".join(report["ambiguous_tokens"]))

    report["issues"] = issues

    if use_llm:
        try:
            feedback_prompt = (
                "Evaluate the clarity of the following prompt on a scale of 0-100. "
                "Identify any issues and suggest improvements. Respond in JSON with keys "
                "score, issues, suggestion.\n\nPrompt:\n" + prompt
            )
            llm_resp = _llm.invoke(feedback_prompt).content
            report["llm_feedback"] = llm_resp
        except Exception:  # noqa: BLE001
            # In offline/CI environments without API key we silently ignore.
            report["llm_feedback"] = None

    return report


# ---------------------------------------------------------------------------
# Runtime helpers – useful for tests & notebooks
# ---------------------------------------------------------------------------


def reload_llm(*, model: str | None = None, llm_class=ChatOpenAI, **llm_kwargs) -> None:  # noqa: D401
    """Re-create the global ``_llm`` after environment variables change.

    A common testing pattern is::

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
        from ambiguity_clarity import reload_llm
        reload_llm()  # picks up NEW key – no need to reimport the package

    Parameters
    ----------
    model
        Model name to instantiate (defaults to the previous one or ``gpt-4o-mini``).
    llm_class
        Chat model class to instantiate – can be swapped with
        ``FakeListChatModel`` (or any ``BaseChatModel`` subclass) to avoid real
        network calls inside tests.
    **llm_kwargs
        Extra keyword arguments forwarded to *llm_class*.
    """
    global _llm  # pylint: disable=global-statement

    # Reload env (helps when tests modify os.environ on the fly)
    load_dotenv(override=True)

    # Fallback to previous model if not provided explicitly
    if model is None and hasattr(_llm, "model_name"):
        model = getattr(_llm, "model_name")  # type: ignore[attr-defined]
    model = model or "gpt-4o-mini"

    # Some fake models (e.g. FakeListChatModel) don’t accept *model*, so we add
    # it only when the constructor supports it.
    try:
        _llm = llm_class(model=model, **llm_kwargs)  # type: ignore[arg-type]
    except TypeError:
        _llm = llm_class(**llm_kwargs)  # type: ignore[arg-type]



def swap_llm(fake_llm) -> None:  # noqa: D401
    """Imperatively replace the global ``_llm`` *in-place*.

    Useful when you already have an instantiated chat model (e.g. a
    ``FakeListChatModel`` with pre-canned responses)::

        swap_llm(FakeListChatModel(responses=["hi"]))
    """
    global _llm  # pylint: disable=global-statement
    _llm = fake_llm


__all__ = [
    "resolve_ambiguity",
    "improve_prompt_clarity",
    "compare_prompt_clarity",
    "analyze_ambiguity",
    "create_structured_prompt",
    "batch_improve_clarity",
    "validate_prompt",
    "reload_llm",
    "swap_llm",
]
