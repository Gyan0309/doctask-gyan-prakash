"""Uploading documents through the browser.

The endpoint is a doorway onto the watcher, not a second ingestion path — so most of
what matters here is what it *refuses*. An uploaded filename is caller-controlled, and
this endpoint writes to disk, which is the exact shape of a path-traversal write.

No database and no API key: these exercise the boundary, and the run behind it is
already covered by the watcher and floor tests.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

import main
from services.watcher import Watcher


def _client(tmp_path, monkeypatch) -> TestClient:
    """A live watcher over tmp_path, with runs stubbed out.

    `start_run=None` makes `poll()` report what it found without starting anything, so
    these tests stay fast and keyless while still exercising the real endpoint.
    """
    settings = main.get_settings()
    monkeypatch.setattr(settings, "watch_dir", tmp_path)
    monkeypatch.setattr(
        main, "_watcher", Watcher(tmp_path, corpus_name="test", start_run=None)
    )
    return TestClient(main.app)


class TestAcceptedUploads:
    def test_a_supported_document_lands_in_the_watched_directory(
        self, tmp_path, monkeypatch
    ) -> None:
        client = _client(tmp_path, monkeypatch)

        response = client.post(
            "/documents/upload",
            files={"files": ("msa.md", b"# Master Services Agreement", "text/markdown")},
        )

        assert response.status_code == 200
        assert response.json()["saved"] == ["msa.md"]
        assert (tmp_path / "msa.md").read_bytes() == b"# Master Services Agreement"

    def test_the_upload_is_reported_to_the_watcher_not_ingested_separately(
        self, tmp_path, monkeypatch
    ) -> None:
        """The file must come back as something the watcher *found*. If it did not,
        the upload wrote to a directory nothing is looking at."""
        client = _client(tmp_path, monkeypatch)

        response = client.post(
            "/documents/upload", files={"files": ("a.md", b"contract", "text/markdown")}
        )

        assert response.json()["added"] == ["a.md"]

    def test_several_documents_arrive_together(self, tmp_path, monkeypatch) -> None:
        client = _client(tmp_path, monkeypatch)

        response = client.post(
            "/documents/upload",
            files=[
                ("files", ("a.md", b"one", "text/markdown")),
                ("files", ("b.txt", b"two", "text/plain")),
            ],
        )

        assert sorted(response.json()["saved"]) == ["a.md", "b.txt"]


class TestRefusedUploads:
    def test_a_traversing_filename_cannot_escape_the_watched_directory(
        self, tmp_path, monkeypatch
    ) -> None:
        """The filename comes from the caller. Written unexamined, `../../x.md` is an
        arbitrary file write, and the fact that it is also a valid document name is
        exactly why it would be easy to miss.
        """
        client = _client(tmp_path, monkeypatch)
        outside = tmp_path.parent / "escaped.md"

        response = client.post(
            "/documents/upload",
            files={"files": ("../../escaped.md", b"payload", "text/markdown")},
        )

        assert response.status_code == 200
        assert not outside.exists(), "upload escaped the watched directory"
        assert (tmp_path / "escaped.md").exists(), "it should land here, flattened"

    def test_an_unsupported_format_is_refused_with_the_supported_list(
        self, tmp_path, monkeypatch
    ) -> None:
        """Refusing without saying what would work makes the user guess."""
        client = _client(tmp_path, monkeypatch)

        response = client.post(
            "/documents/upload",
            files={"files": ("notes.docx", b"...", "application/octet-stream")},
        )

        assert response.status_code == 400
        detail = response.json()["detail"]
        assert ".md" in detail and ".pdf" in detail
        assert not list(tmp_path.iterdir()), "a refused upload must write nothing"

    def test_a_name_that_is_only_a_path_is_refused(self, tmp_path, monkeypatch) -> None:
        """`/` and `.` survive as filenames but basename to nothing. Unguarded, the
        endpoint would try to write the watched directory itself."""
        client = _client(tmp_path, monkeypatch)

        for filename in ("/", "."):
            response = client.post(
                "/documents/upload",
                files={"files": (filename, b"payload", "text/markdown")},
            )
            assert response.status_code == 400, filename

        assert not list(tmp_path.iterdir())
