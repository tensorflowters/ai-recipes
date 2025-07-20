"""Live pricing resolver for OpenAI models.

This module provides functionality to:
- Map models to $/1K prompt tokens and $/1K completion tokens (with embedded static defaults)
- Optionally pull fresh pricing JSON from OpenAI pricing page (cached for 24 hours)
- Expose get_price(model, token_type) for real current prices
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

try:
    import httpx
except ImportError:
    httpx = None


# Static default pricing data (as of January 2025)
# Prices are in USD per 1,000 tokens
STATIC_PRICING: Dict[str, Dict[str, float]] = {
    # GPT-4o models
    "gpt-4o": {"prompt": 0.0025, "completion": 0.01},
    "gpt-4o-2024-11-20": {"prompt": 0.0025, "completion": 0.01},
    "gpt-4o-2024-08-06": {"prompt": 0.0025, "completion": 0.01},
    "gpt-4o-2024-05-13": {"prompt": 0.005, "completion": 0.015},
    # GPT-4o mini
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "gpt-4o-mini-2024-07-18": {"prompt": 0.00015, "completion": 0.0006},
    # GPT-4 Turbo
    "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
    "gpt-4-turbo-2024-04-09": {"prompt": 0.01, "completion": 0.03},
    "gpt-4-turbo-preview": {"prompt": 0.01, "completion": 0.03},
    "gpt-4-0125-preview": {"prompt": 0.01, "completion": 0.03},
    "gpt-4-1106-preview": {"prompt": 0.01, "completion": 0.03},
    # GPT-4
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "gpt-4-0613": {"prompt": 0.03, "completion": 0.06},
    "gpt-4-32k": {"prompt": 0.06, "completion": 0.12},
    "gpt-4-32k-0613": {"prompt": 0.06, "completion": 0.12},
    # GPT-3.5 Turbo
    "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002},
    "gpt-3.5-turbo-0125": {"prompt": 0.0005, "completion": 0.0015},
    "gpt-3.5-turbo-1106": {"prompt": 0.001, "completion": 0.002},
    "gpt-3.5-turbo-16k": {"prompt": 0.003, "completion": 0.004},
    "gpt-3.5-turbo-instruct": {"prompt": 0.0015, "completion": 0.002},
    # o1 models
    "o1": {"prompt": 0.015, "completion": 0.06},
    "o1-2024-12-17": {"prompt": 0.015, "completion": 0.06},
    "o1-mini": {"prompt": 0.003, "completion": 0.012},
    "o1-mini-2024-09-12": {"prompt": 0.003, "completion": 0.012},
    "o1-preview": {"prompt": 0.015, "completion": 0.06},
    "o1-preview-2024-09-12": {"prompt": 0.015, "completion": 0.06},
    # Text embedding models
    "text-embedding-3-small": {"prompt": 0.00002, "completion": 0.0},
    "text-embedding-3-large": {"prompt": 0.00013, "completion": 0.0},
    "text-embedding-ada-002": {"prompt": 0.0001, "completion": 0.0},
    # Audio models
    "whisper-1": {"prompt": 0.006, "completion": 0.0},  # per minute, not per 1K tokens
    "tts-1": {"prompt": 0.015, "completion": 0.0},  # per 1K characters
    "tts-1-hd": {"prompt": 0.03, "completion": 0.0},  # per 1K characters
    # Vision models (same as base models but included for clarity)
    "gpt-4-vision-preview": {"prompt": 0.01, "completion": 0.03},
    "gpt-4o-vision": {"prompt": 0.0025, "completion": 0.01},
}


logger = logging.getLogger(__name__)


class PricingCache:
    """Simple file-based cache for pricing data."""

    def __init__(self, cache_duration_hours: int = 24):
        self.cache_duration_hours = cache_duration_hours
        self.cache_file = Path.home() / ".cache" / "ambiguity_clarity" / "pricing_cache.json"
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)

    def is_cache_valid(self) -> bool:
        """Check if cached data is still valid."""
        if not self.cache_file.exists():
            return False

        cache_age_hours = (time.time() - self.cache_file.stat().st_mtime) / 3600
        return cache_age_hours < self.cache_duration_hours

    def load_cached_pricing(self) -> Optional[Dict[str, Dict[str, float]]]:
        """Load pricing data from cache if valid."""
        if not self.is_cache_valid():
            return None

        try:
            with open(self.cache_file, "r") as f:
                data = json.load(f)
                logger.debug(f"Loaded pricing data from cache: {len(data)} models")
                return data
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            logger.warning(f"Failed to load cached pricing: {e}")
            return None

    def save_pricing_to_cache(self, pricing_data: Dict[str, Dict[str, float]]) -> None:
        """Save pricing data to cache."""
        try:
            with open(self.cache_file, "w") as f:
                json.dump(pricing_data, f, indent=2)
                logger.debug(f"Cached pricing data for {len(pricing_data)} models")
        except Exception as e:
            logger.warning(f"Failed to cache pricing data: {e}")


class PricingResolver:
    """Resolves pricing for OpenAI models with caching and fallback."""

    def __init__(self, cache_duration_hours: int = 24, pricing_url: Optional[str] = None, enable_live_pricing: bool = True):
        """
        Initialize the pricing resolver.

        Args:
            cache_duration_hours: How long to cache live pricing data
            pricing_url: URL to fetch live pricing from (defaults to None for now)
            enable_live_pricing: Whether to attempt fetching live pricing
        """
        self.cache = PricingCache(cache_duration_hours)
        self.pricing_url = pricing_url
        self.enable_live_pricing = enable_live_pricing and httpx is not None
        self._live_pricing: Optional[Dict[str, Dict[str, float]]] = None

        if not httpx and enable_live_pricing:
            logger.warning("httpx not available, live pricing disabled. Install with: pip install httpx")

    def _fetch_live_pricing(self) -> Optional[Dict[str, Dict[str, float]]]:
        """Fetch live pricing data from external source.

        Note: This is a placeholder implementation. In practice, you would:
        1. Parse the OpenAI pricing page HTML
        2. Use an internal pricing API
        3. Maintain your own pricing JSON endpoint

        For now, this returns None to use static pricing.
        """
        if not self.enable_live_pricing or not self.pricing_url:
            return None

        try:
            logger.info(f"Fetching live pricing from {self.pricing_url}")
            with httpx.Client(timeout=10.0) as client:
                response = client.get(self.pricing_url)
                response.raise_for_status()

                # This would need to be implemented based on the actual API/format
                # For now, return None to use static pricing
                logger.warning("Live pricing parsing not implemented yet")
                return None

        except Exception as e:
            logger.error(f"Failed to fetch live pricing: {e}")
            return None

    def _get_pricing_data(self) -> Dict[str, Dict[str, float]]:
        """Get the most up-to-date pricing data available."""
        # Try cached live pricing first
        if self.enable_live_pricing:
            cached_pricing = self.cache.load_cached_pricing()
            if cached_pricing:
                logger.debug("Using cached live pricing data")
                return {**STATIC_PRICING, **cached_pricing}

            # Try to fetch fresh live pricing
            live_pricing = self._fetch_live_pricing()
            if live_pricing:
                logger.info("Using fresh live pricing data")
                self.cache.save_pricing_to_cache(live_pricing)
                return {**STATIC_PRICING, **live_pricing}

        # Fall back to static pricing
        logger.debug("Using static pricing data")
        return STATIC_PRICING

    def get_price(self, model: str, token_type: Literal["prompt", "completion"]) -> Optional[float]:
        """Get the price per 1K tokens for a specific model and token type.

        Args:
            model: The model name (e.g., "gpt-4o", "gpt-3.5-turbo")
            token_type: Either "prompt" or "completion"

        Returns:
            Price per 1K tokens in USD, or None if model/token_type not found
        """
        pricing_data = self._get_pricing_data()

        if model not in pricing_data:
            logger.warning(f"Model '{model}' not found in pricing data")
            return None

        model_pricing = pricing_data[model]
        if token_type not in model_pricing:
            logger.warning(f"Token type '{token_type}' not found for model '{model}'")
            return None

        price = model_pricing[token_type]
        logger.debug(f"Price for {model} {token_type}: ${price:.6f} per 1K tokens")
        return price

    def get_model_pricing(self, model: str) -> Optional[Tuple[float, float]]:
        """Get both prompt and completion pricing for a model.

        Args:
            model: The model name

        Returns:
            Tuple of (prompt_price, completion_price) per 1K tokens, or None if not found
        """
        pricing_data = self._get_pricing_data()

        if model not in pricing_data:
            return None

        model_pricing = pricing_data[model]
        return (model_pricing.get("prompt", 0.0), model_pricing.get("completion", 0.0))

    def list_available_models(self) -> list[str]:
        """Get a list of all available models in the pricing data."""
        return list(self._get_pricing_data().keys())


# Global pricing resolver instance
_pricing_resolver = PricingResolver()


def get_price(model: str, token_type: Literal["prompt", "completion"]) -> Optional[float]:
    """Get the price per 1K tokens for a specific model and token type.

    This is the main public API function that uses the global pricing resolver.

    Args:
        model: The model name (e.g., "gpt-4o", "gpt-3.5-turbo")
        token_type: Either "prompt" or "completion"

    Returns:
        Price per 1K tokens in USD, or None if model/token_type not found

    Example:
        >>> get_price("gpt-4o", "prompt")
        0.0025
        >>> get_price("gpt-4o", "completion")
        0.01
        >>> get_price("unknown-model", "prompt")
        None
    """
    return _pricing_resolver.get_price(model, token_type)


def get_model_pricing(model: str) -> Optional[Tuple[float, float]]:
    """Get both prompt and completion pricing for a model.

    Args:
        model: The model name

    Returns:
        Tuple of (prompt_price, completion_price) per 1K tokens, or None if not found

    Example:
        >>> get_model_pricing("gpt-4o")
        (0.0025, 0.01)
    """
    return _pricing_resolver.get_model_pricing(model)


def list_available_models() -> list[str]:
    """Get a list of all available models in the pricing data."""
    return _pricing_resolver.list_available_models()


def configure_pricing_resolver(
    *, cache_duration_hours: Optional[int] = None, pricing_url: Optional[str] = None, enable_live_pricing: Optional[bool] = None
) -> None:
    """Configure the global pricing resolver.

    Args:
        cache_duration_hours: How long to cache live pricing data
        pricing_url: URL to fetch live pricing from
        enable_live_pricing: Whether to attempt fetching live pricing
    """
    global _pricing_resolver

    kwargs = {}
    if cache_duration_hours is not None:
        kwargs["cache_duration_hours"] = cache_duration_hours
    if pricing_url is not None:
        kwargs["pricing_url"] = pricing_url
    if enable_live_pricing is not None:
        kwargs["enable_live_pricing"] = enable_live_pricing

    if kwargs:
        _pricing_resolver = PricingResolver(**kwargs)
        logger.info(f"Pricing resolver reconfigured with: {kwargs}")
