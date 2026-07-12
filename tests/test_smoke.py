"""Smoke test — imports work and version is exposed."""


def test_import_rain():
    import rain

    assert rain.__version__ == "0.0.0"


def test_python_version():
    import sys

    assert sys.version_info >= (3, 11)
