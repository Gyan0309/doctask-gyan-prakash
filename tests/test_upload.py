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


def _client(tmp_path, monkeypatch, *, record: list | None = None) -> TestClient:
    """A live watcher over tmp_path, with the run itself stubbed out.

    The stub still calls `on_started`, because that callback is what lets the endpoint
    answer without waiting out the whole run — the behaviour most worth protecting
    here. It records the paths it was handed so a test can assert the handoff really
    happened rather than inferring it from a status string.
    """

    def _fake_start_run(*, corpus_name, document_paths, on_started=None, **kwargs):
        if record is not None:
            record.extend(document_paths)
        if on_started is not None:
            on_started("11111111-2222-3333-4444-555555555555")
        return {"run_id": "11111111-2222-3333-4444-555555555555"}

    settings = main.get_settings()
    monkeypatch.setattr(settings, "watch_dir", tmp_path)
    monkeypatch.setattr(
        main,
        "_watcher",
        Watcher(tmp_path, corpus_name="test", start_run=_fake_start_run),
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

    def test_the_upload_is_handed_to_the_watcher_not_ingested_separately(
        self, tmp_path, monkeypatch
    ) -> None:
        """The uploaded file must reach the run through the watcher. If it did not,
        the upload wrote to a directory nothing is looking at."""
        handed: list[str] = []
        client = _client(tmp_path, monkeypatch, record=handed)

        client.post(
            "/documents/upload", files={"files": ("a.md", b"contract", "text/markdown")}
        )

        assert any(p.endswith("a.md") for p in handed), (
            f"the run was never handed the uploaded file; got {handed}"
        )

    def test_it_answers_with_a_run_id_rather_than_waiting_out_the_run(
        self, tmp_path, monkeypatch
    ) -> None:
        """Holding the request open for the whole run was honest and unusable: a cold
        corpus is a minute of a dropzone saying nothing, and the page cannot show the
        stages it is waiting on because it is blocked on the same request.

        The id exists the moment the run row is inserted, so that is when the caller
        gets it — and the page polls from there.
        """
        client = _client(tmp_path, monkeypatch)

        body = client.post(
            "/documents/upload", files={"files": ("a.md", b"contract", "text/markdown")}
        ).json()

        assert body["run_id"], "the caller needs an id it can poll"
        assert body["status"] == "running"
        assert body["poll"] == f"/runs/{body['run_id']}"

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
