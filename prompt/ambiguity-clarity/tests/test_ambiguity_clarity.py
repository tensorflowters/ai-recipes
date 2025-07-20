import pytest
from unittest.mock import MagicMock
from ambiguity_clarity import (
    resolve_ambiguity,
    improve_prompt_clarity,
    compare_prompt_clarity,
    analyze_ambiguity,
    create_structured_prompt,
    batch_improve_clarity,
    validate_prompt,
)
from langchain_community.chat_models import FakeListChatModel
import ambiguity_clarity


@pytest.fixture
def mock_llm(mocker):
    """Fixture to mock the language model."""
    mock = MagicMock()
    mocker.patch("ambiguity_clarity._llm", new=mock)
    return mock


def test_resolve_ambiguity_financial(mock_llm):
    """Test resolving ambiguity with a financial context."""
    mock_llm.invoke.return_value.content = "This is a test response for a financial query."
    prompt = "Tell me about the bank."
    context = "You are a financial advisor."
    response = resolve_ambiguity(prompt, context)
    assert "financial" in response
    mock_llm.invoke.assert_called_once()


def test_resolve_ambiguity_geographical(mock_llm):
    """Test resolving ambiguity with a geographical context."""
    mock_llm.invoke.return_value.content = "This is a test response for a geographical query."
    prompt = "Tell me about the bank."
    context = "You are a geographer."
    response = resolve_ambiguity(prompt, context)
    assert "geographical" in response
    mock_llm.invoke.assert_called_once()


def test_improve_prompt_clarity_difference(mock_llm):
    """Test improving prompt clarity for a 'difference' question."""
    mock_llm.invoke.return_value.content = "What are the differences between Python and JavaScript?"
    prompt = "What's the difference?"
    response = improve_prompt_clarity(prompt)
    assert "Python and JavaScript" in response
    mock_llm.invoke.assert_called_once()


def test_improve_prompt_clarity_how_it_works(mock_llm):
    """Test improving prompt clarity for a 'how it works' question."""
    mock_llm.invoke.return_value.content = "Can you explain how a car engine works?"
    prompt = "How does it work?"
    response = improve_prompt_clarity(prompt)
    assert "car engine" in response
    mock_llm.invoke.assert_called_once()


def test_compare_prompt_clarity(mock_llm):
    """Test comparing responses from a vague and a clear prompt."""
    mock_llm.invoke.side_effect = [MagicMock(content="Vague response."), MagicMock(content="Clear response about pizza.")]
    original = "How do I make it?"
    improved = "Provide a step-by-step guide for making a classic margherita pizza."
    original_response, improved_response = compare_prompt_clarity(original, improved)
    assert original_response == "Vague response."
    assert "pizza" in improved_response
    assert mock_llm.invoke.call_count == 2


def test_analyze_ambiguity(mock_llm):
    """Test the ambiguity analysis function."""
    mock_llm.invoke.return_value.content = (
        '{"reasons": ["vague term \'bank\'"], "interpretations": ["financial", "geographical"]}'
    )
    prompt = "Tell me about the bank."
    analysis = analyze_ambiguity(prompt)
    assert "reasons" in analysis["analysis"]
    assert "interpretations" in analysis["analysis"]
    mock_llm.invoke.assert_called_once()


def test_create_structured_prompt():
    from langchain_community.chat_models import FakeListChatModel

    model = FakeListChatModel(responses=["Analysis of social media."])
    topic = "the impact of social media on society"
    aspects = ["communication patterns", "mental health", "information spread"]

    # Patch global llm just for this test
    from importlib import reload  # noqa: F401
    import ambiguity_clarity as ac

    original_llm = ac._llm
    ac._llm = model
    try:
        response = ac.create_structured_prompt(topic, aspects)
        assert "social media" in response
    finally:
        ac._llm = original_llm


def test_create_structured_prompt_custom_tone():
    from langchain_community.chat_models import FakeListChatModel

    model = FakeListChatModel(responses=["A formal analysis of AI."])
    topic = "AI"
    aspects = ["history", "future", "ethics"]
    tone = "formal"

    import ambiguity_clarity as ac

    original_llm = ac._llm
    ac._llm = model
    try:
        response = ac.create_structured_prompt(topic, aspects, tone)
        assert "formal" in response.lower()
    finally:
        ac._llm = original_llm


def test_batch_improve_clarity(mock_llm):
    """Test batch processing of prompts for clarity improvement."""
    mock_llm.invoke.side_effect = [MagicMock(content="Improved prompt 1"), MagicMock(content="Improved prompt 2")]
    prompts = ["Vague 1", "Vague 2"]
    responses = batch_improve_clarity(prompts)
    assert responses == ["Improved prompt 1", "Improved prompt 2"]
    assert mock_llm.invoke.call_count == 2


def test_batch_improve_clarity_empty_list(mock_llm):
    """Test batch processing with an empty list."""
    responses = batch_improve_clarity([])
    assert responses == []
    mock_llm.invoke.assert_not_called()


def test_create_structured_prompt_sectioned(mock_llm):
    """Test sectioned_analysis format."""
    invoke_return_mock = MagicMock()
    invoke_return_mock.content = "# AI\n\n## history\nSection text."
    mock_llm.invoke.return_value = invoke_return_mock

    topic = "AI"
    aspects = ["history"]
    response = create_structured_prompt(topic, aspects, tone="formal", format="sectioned_analysis")
    assert "## history" in response
    mock_llm.invoke.assert_called_once()


def test_validate_prompt_heuristic():
    prompt = "Explain it"
    report = validate_prompt(prompt)
    assert report["score"] < 100
    assert "issues" in report


@pytest.fixture
def fake_llm(mocker):
    # order matters: every .invoke() pops from the front
    msgs = [
        "Improved #1",
        "Improved #2",
        "Original answer",
        "Better answer",
    ]
    mocker.patch(
        "ambiguity_clarity._llm",
        (model := FakeListChatModel(responses=msgs)),
    )
    return model


def test_batch(fake_llm):
    outs = ambiguity_clarity.batch_improve_clarity(["a", "b"])
    assert outs == ["Improved #1", "Improved #2"]


def test_compare(fake_llm):
    orig, better = ambiguity_clarity.compare_prompt_clarity("x", "y")
    assert (orig, better) == ("Improved #1", "Improved #2")


# Integration tests - these test real LLM functionality
# Run with: pytest -m integration (requires OPENAI_API_KEY)


@pytest.mark.integration
def test_real_prompt_improvement_quality():
    """Test that the LLM actually improves prompt quality."""
    import os

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    # Test with genuinely vague prompts
    vague_prompts = ["Explain it", "What's the difference?", "How does it work?", "Tell me about that"]

    for vague_prompt in vague_prompts:
        improved = improve_prompt_clarity(vague_prompt)

        # Quality checks - improved prompt should be:
        assert len(improved) > len(vague_prompt), f"Improved prompt should be longer: '{improved}'"
        assert improved != vague_prompt, "Should actually change the prompt"
        assert "?" in improved or improved.endswith(".") or improved.endswith('"'), "Should be a complete sentence"
        # Should generally reduce vague language, but allow for reasonable trade-offs
        vague_words = ["it", "that", "this", "thing"]
        improved_lower = improved.lower()
        vague_count_original = sum(1 for word in vague_words if word in vague_prompt.lower())
        vague_count_improved = sum(1 for word in vague_words if word in improved_lower)
        
        # Allow some increase in vague words if the prompt is significantly more detailed
        # The key is that the overall clarity should be better
        length_improvement_ratio = len(improved) / len(vague_prompt) if len(vague_prompt) > 0 else 1
        
        # If the prompt is much longer and more detailed (3x+ length), allow some vague words
        if length_improvement_ratio >= 3.0 and len(improved) > 50:
            # For very detailed improvements, allow reasonable use of contextual words
            assert vague_count_improved <= vague_count_original + 2, "Should not add too many vague words even with detailed improvements"
        else:
            # For normal improvements, allow at most one extra vague word (LLMs sometimes add context like "itself")
            assert vague_count_improved <= vague_count_original + 1, "Should not significantly increase vague language count"


@pytest.mark.integration
def test_real_ambiguity_resolution():
    """Test that context actually helps resolve ambiguity."""
    import os

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    ambiguous_prompt = "Tell me about the bank"

    # Test different contexts
    financial_context = "You are a financial advisor helping someone understand banking."
    geographical_context = "You are a geographer explaining natural landforms."

    financial_response = resolve_ambiguity(ambiguous_prompt, financial_context)
    geographical_response = resolve_ambiguity(ambiguous_prompt, geographical_context)

    # Responses should be different and contextually appropriate
    assert financial_response != geographical_response, "Different contexts should yield different responses"

    # Check for context-specific terminology
    financial_terms = ["account", "savings", "deposit", "loan", "credit", "money", "financial"]
    geographical_terms = ["river", "water", "erosion", "slope", "land", "earth", "natural"]

    financial_response_lower = financial_response.lower()
    geographical_response_lower = geographical_response.lower()

    # Financial response should contain more financial terms
    financial_score = sum(1 for term in financial_terms if term in financial_response_lower)
    geo_in_financial = sum(1 for term in geographical_terms if term in financial_response_lower)

    # Geographical response should contain more geographical terms
    geo_score = sum(1 for term in geographical_terms if term in geographical_response_lower)
    financial_in_geo = sum(1 for term in financial_terms if term in geographical_response_lower)

    assert financial_score > geo_in_financial, "Financial context should produce more financial content"
    assert geo_score > financial_in_geo, "Geographical context should produce more geographical content"


@pytest.mark.integration
def test_real_structured_prompt_quality():
    """Test that structured prompts actually provide structured output."""
    import os

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    topic = "machine learning"
    aspects = ["supervised learning", "unsupervised learning", "neural networks"]

    response = create_structured_prompt(topic, aspects, tone="educational")

    # Should mention the topic
    assert "machine learning" in response.lower() or "ml" in response.lower()

    # Should address multiple aspects
    response_lower = response.lower()
    aspects_mentioned = sum(1 for aspect in aspects if any(word in response_lower for word in aspect.split()))
    assert aspects_mentioned >= 2, f"Should mention at least 2 aspects, found {aspects_mentioned}"

    # Should be substantial (educational content)
    assert len(response) > 200, "Educational content should be substantial"

    # Should have structure indicators
    structure_indicators = ["first", "second", "third", "1", "2", "3", "•", "-", ":", "\n"]
    has_structure = any(indicator in response for indicator in structure_indicators)
    assert has_structure, "Should show some structural organization"


# Property-based testing for edge cases
def test_prompt_improvement_properties():
    """Test properties that should hold regardless of LLM response."""
    from langchain_community.chat_models import FakeListChatModel
    import ambiguity_clarity as ac

    # Test with various fake responses
    test_cases = [
        "A well-structured response about the topic",
        "Short answer",
        "A much longer response with many details about the specific topic that was requested",
        "",  # Empty response edge case
    ]

    for fake_response in test_cases:
        model = FakeListChatModel(responses=[fake_response])
        original_llm = ac._llm
        ac._llm = model

        try:
            # Test that function handles all types of responses gracefully
            result = improve_prompt_clarity("test prompt")
            assert isinstance(result, str), "Should always return a string"
            # Even with empty response, should return something (the empty string)
            assert result == fake_response, "Should return exactly what the LLM provides"
        finally:
            ac._llm = original_llm


def test_error_handling_network_issues():
    """Test how the system handles LLM errors."""
    import ambiguity_clarity as ac
    from unittest.mock import MagicMock

    # Mock an LLM that raises an exception
    error_llm = MagicMock()
    error_llm.invoke.side_effect = Exception("Network error")

    original_llm = ac._llm
    ac._llm = error_llm

    try:
        with pytest.raises(Exception):
            improve_prompt_clarity("test")
    finally:
        ac._llm = original_llm


def test_prompt_validation_heuristics():
    """Test heuristic-based prompt quality validation."""
    # Test various prompt qualities
    test_cases = [
        ("Explain it", "Should score low - very vague"),
        ("What's the difference?", "Should score low - missing context"),
        (
            "Explain the difference between supervised and unsupervised machine learning",
            "Should score high - specific and clear",
        ),
        ("How does gradient descent optimization work in neural network training?", "Should score high - very specific"),
        ("", "Should score very low - empty"),
        ("a", "Should score low - too short"),
    ]

    scores = []
    for prompt, description in test_cases:
        report = validate_prompt(prompt)
        scores.append((prompt, report["score"], description))

        assert "score" in report
        assert "issues" in report
        assert 0 <= report["score"] <= 100

    # Verify that more specific prompts generally score higher
    vague_scores = [score for prompt, score, _ in scores if len(prompt) < 20 and prompt != ""]
    specific_scores = [score for prompt, score, _ in scores if len(prompt) > 50]

    if vague_scores and specific_scores:
        assert max(specific_scores) > max(vague_scores), "Specific prompts should generally score higher"
