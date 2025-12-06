"""Pytest configuration and fixtures."""

import pytest


def pytest_collection_modifyitems(config, items):
    """Automatically skip tests in training modules if torch is not available."""
    try:
        import torch  # noqa: F401

        torch_available = True
    except ImportError:
        torch_available = False

    if not torch_available:
        skip_torch = pytest.mark.skip(
            reason="PyTorch not installed (required for training/export)"
        )

        # Skip entire test modules that require torch
        torch_required_modules = [
            "test_east.py",  # PyTorch model tests
            "test_dataset.py",  # Dataset tests
            "test_loss.py",  # Loss function tests
            "test_sam.py",  # SAM optimizer tests
            "test_train_utils.py",  # Training utilities
            "test_east_train_export.py",  # Train/export tests
        ]

        for item in items:
            # Skip if test is in a module that requires torch
            if any(module in str(item.fspath) for module in torch_required_modules):
                item.add_marker(skip_torch)

            # Skip if test has requires_torch marker
            if "requires_torch" in item.keywords:
                item.add_marker(skip_torch)
