"""Sensitivity-analyzer tests (disabled).

The original test targeted a pre-refactor layout (src.engine.*, src.analyzer.*,
src.models.model_builder) that no longer exists. Skipped so test collection does
not error; rewrite against the current API (src.core.Runner, src.core.build,
src.datasets.factory) when the sensitivity analyzer is reintroduced.
"""

import pytest

pytest.skip("legacy test: pre-refactor modules removed; rewrite against current API",
            allow_module_level=True)
