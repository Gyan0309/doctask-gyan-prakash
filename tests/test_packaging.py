"""The image contains what the app imports.

Written after `integrations/` was added, tested entirely from a local venv, and never
copied into the Dockerfile — so `docker compose up` died at import with
`ModuleNotFoundError: No module named 'integrations'`. Every unit test passed. The
container simply did not contain the code.

The Dockerfile lists packages one COPY at a time, which is good for layer caching and
means adding a package silently omits it. This closes that class: any top-level package
that is not shipped fails here, at the cost of a directory listing.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Directories that are deliberately not in the runtime image.
NOT_SHIPPED = {"tests", "migrations", "web", "corpus", "docker", "scripts", "inbox"}


def _top_level_packages() -> set[str]:
    return {
        path.name
        for path in ROOT.iterdir()
        if path.is_dir()
        and (path / "__init__.py").exists()
        and not path.name.startswith((".", "_"))
        and path.name not in NOT_SHIPPED
    }


def test_every_package_is_copied_into_the_image() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    missing = [
        package
        for package in sorted(_top_level_packages())
        if f"COPY {package}/" not in dockerfile
    ]

    assert not missing, (
        f"these packages exist but are not COPY'd into the image: {missing}. "
        f"The container will fail at import while every test still passes."
    )


def test_the_module_entrypoints_are_copied() -> None:
    """`main.py` is the API and `mcp_server.py` is the machine surface. Both are
    entrypoints someone runs, and neither is inside a package."""
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    for entrypoint in ("main.py", "mcp_server.py"):
        assert entrypoint in dockerfile, f"{entrypoint} is not copied into the image"
