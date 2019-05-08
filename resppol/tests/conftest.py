import pytest


def pytest_addoption(parser):
    """Add the pytest command line option --runslow and --failwip.
    If --runslow is not given, tests marked with pytest.mark.slow are
    skipped.
    If --failwip is not give, tests marked with pytest.mark.wip are
    xfailed.
    """
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
