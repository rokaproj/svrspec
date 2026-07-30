from pathlib import Path

import pytest

from svrspec.catalog import Catalog
from svrspec.perf import Efficiency

DATA = Path(__file__).parent / "data"


@pytest.fixture
def catalog() -> Catalog:
    return Catalog(DATA)


@pytest.fixture
def model_8b(catalog):
    return catalog.model("test-8b-gqa")


@pytest.fixture
def model_3b(catalog):
    return catalog.model("test-3b")


@pytest.fixture
def moe(catalog):
    return catalog.model("test-30b-moe")


@pytest.fixture
def q4(catalog):
    return catalog.quant("Q4_K_M")


@pytest.fixture
def eff(catalog) -> Efficiency:
    return Efficiency.from_catalog(catalog.coefficients)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """No test may reach GitHub.

    The update check is the only outbound call in the codebase; disabling it by
    default means a test can never depend on the network being up, and the tests
    that exercise the check opt back in explicitly.
    """
    from svrspec.update import DISABLE_ENV

    monkeypatch.setenv(DISABLE_ENV, "1")
