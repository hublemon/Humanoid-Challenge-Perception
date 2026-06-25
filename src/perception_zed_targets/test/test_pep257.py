# Copyright 2026 perception
#
# Licensed under the Apache License, Version 2.0.
"""Run pep257 docstring checks."""

from ament_pep257.main import main
import pytest


@pytest.mark.linter
@pytest.mark.pep257
def test_pep257():
    """Check pep257 conformance."""
    rc = main(argv=['.', 'test'])
    assert rc == 0, 'Found code style errors / warnings'
