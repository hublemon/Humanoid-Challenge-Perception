# Copyright 2026 perception
#
# Licensed under the Apache License, Version 2.0.
"""Run copyright checks."""

from ament_copyright.main import main
import pytest


@pytest.mark.copyright
@pytest.mark.linter
def test_copyright():
    """Check copyright headers."""
    rc = main(argv=['.', 'test'])
    assert rc == 0, 'Found errors'
