"""Smoke tests: the package and its sub-modules import cleanly."""

import importlib
import importlib.metadata
from pathlib import Path

import pytest

import caelum

SUBMODULES = [
    "caelum.openai",
    "caelum.anthropic",
    "caelum.metrics",
    "caelum.fallback",
    "caelum.utils",
    "caelum.cache",
    "caelum.cache.exact",
    "caelum.cache.semantic",
    "caelum.cache.stores",
    "caelum.router",
    "caelum.router.config",
    "caelum.router.engine",
]


def test_version_is_single_sourced() -> None:
    assert caelum.__version__ == importlib.metadata.version("caelum-ai")


def test_package_license_matches_the_repository_license() -> None:
    package_dir = Path(__file__).resolve().parents[1]
    packaged = (package_dir / "LICENSE").read_text(encoding="utf-8")
    assert packaged == (package_dir.parents[1] / "LICENSE").read_text(encoding="utf-8")


@pytest.mark.parametrize("module_name", SUBMODULES)
def test_submodules_import(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


def test_missing_extra_messages_name_the_real_package() -> None:
    from caelum.utils import missing_extra_message

    assert "'google-genai' package" in missing_extra_message("gemini", "caelum.genai")
    assert "'mistralai' package" in missing_extra_message("mistral")
    assert "caelum-ai[gemini]" in missing_extra_message("gemini")
    assert "'openai' package" in missing_extra_message("openai")


def test_format_usd_keeps_small_amounts_visible() -> None:
    from caelum.utils import format_usd

    assert format_usd(0) == "$ 0.00"
    assert format_usd(24.3) == "$ 24.30"
    assert format_usd(0.0421) == "$ 0.0421"
    assert format_usd(0.000174) == "$ 0.000174"
