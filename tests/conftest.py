"""Pytest configuration and fixtures."""

import pytest


def pytest_collection_modifyitems(config, items):
    """
    Mark tests that require torch for skipping when torch is unavailable.

    Note: This only works for tests that can be imported. Tests in modules
    that directly import torch-dependent code will fail during collection.
    Use --ignore flags in pytest command to skip those modules entirely.
    """
    try:
        import torch  # noqa: F401

        torch_available = True
    except ImportError:
        torch_available = False

    if not torch_available:
        skip_torch = pytest.mark.skip(
            reason="PyTorch not installed (required for training/export)"
        )

        for item in items:
            # Skip if test has requires_torch marker
            if "requires_torch" in item.keywords:
                item.add_marker(skip_torch)
