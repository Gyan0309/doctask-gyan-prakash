"""The MCP surface — behavior 4 in the shape SuperDocs uses itself.

These drive the server through a real `mcp.Client` over the protocol, not by calling
the decorated functions directly. Calling the functions would prove the service layer
works, which the other tests already do; it would say nothing about whether the tools
are registered, whether their schemas are valid, or whether a caller can actually reach
them — which is the entire question this file exists to answer.

The headline test runs the whole flow end to end over MCP with no human and no HTTP:
start a run, read the findings it parked on, approve some and reject others, and
confirm the register committed with the verdicts intact.
"""

from __future__ import annotations

import os
import sys
import uuid

import pytest
from mcp import Client, ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import services.service as service
from mcp_server import server

pytestmark = pytest.mark.integration

MSA = """Northwind Analytics Master Services Agreement

The hourly rate is $180 per hour for all professional services.

Payment terms are net 45 days from the date of invoice.

Governing law is the State of Delaware.

Estimated annual fees under this Agreement are $600,000.
"""


@pytest.fixture(autouse=True)
def _fresh(engine):
    service.reset_checkpointer()
    yield
    service.reset_checkpointer()


def _unique(name: str) -> str:
    return f"{name}-{uuid.uuid4().hex[:8]}"


def _payload(result):
    """The JSON a tool returned, or fail loudly with the server's own message.

    A failed call comes back as a result flagged `is_error` rather than raising, so a
    test that ignored the flag would pass on an error response — the exact failure mode
    this surface exists to prevent.

    Read from `structured_content`, not `content`. A tool returning a list emits **one
    content block per item**, so `content[0]` is the first element rather than the
    whole answer — a helper that read it would quietly assert against one row of a
    collection and pass. `structured_content` carries the entire typed payload, with
    lists wrapped as `{"result": [...]}` because MCP structured output is JSON Schema
    and that requires an object at the top level.
    """
    assert not result.is_error, f"tool call failed: {result.content}"
    payload = result.structured_content
    assert payload is not None, f"tool returned no structured content: {result.content}"
    if set(payload) == {"result"}:
        return payload["result"]
    return payload


class TestTheSurfaceExists:
    async def test_every_operation_the_flow_needs_is_exposed(self) -> None:
        """A machine interface missing the gate is not a machine interface for this
        system — approval is the operation that matters most."""
        async with Client(server) as client:
            names = {t.name for t in (await client.list_tools()).tools}

        assert {
            "start_run",
            "get_run",
            "submit_decisions",
            "get_deliverable",
            "get_provenance",
            "get_changes",
            "get_decisions",
            "resume_interrupted_run",
            "list_runs",
        } <= names

    async def test_tools_describe_themselves(self) -> None:
        """The caller is a model. A tool with no description is a tool it will use
        wrongly or not at all."""
        async with Client(server) as client:
            tools = (await client.list_tools()).tools

        for tool in tools:
            assert tool.description, f"{tool.name} has no description"
            assert tool.input_schema, f"{tool.name} has no input schema"


class TestDrivingTheWholeFlow:
    async def test_a_machine_runs_the_entire_flow_including_the_gate(self, tmp_path) -> None:
        """Behavior 4, proven over the actual protocol: no browser, no HTTP, no human.

        The assertion that matters is not that it finished — it is that the gate was
        genuinely answered through this surface, per item, and that the verdicts
        survived into the record.
        """
        (tmp_path / "msa.md").write_text(MSA, encoding="utf-8")
        (tmp_path / "broken.tiff").write_bytes(b"not a document")

        async with Client(server) as client:
            started = _payload(
                await client.call_tool(
                    "start_run",
                    {
                        "corpus_name": _unique("mcp-flow"),
                        "document_paths": [
                            str(tmp_path / "msa.md"),
                            str(tmp_path / "broken.tiff"),
                        ],
                    },
                )
            )
            run_id = started["run_id"]
            assert started["awaiting_review"] is True, "the run must park at the gate"

            # Read what we are deciding on, the way a caller actually would — by asking
            # the run about itself rather than trusting the response that started it.
            seen = _payload(await client.call_tool("get_run", {"run_id": run_id}))
            findings = seen["pending_findings"]
            assert findings, "the gate must present its findings through this surface"

            # Some approved, some rejected, in one pass.
            decisions = {
                str(f["index"]): ("approved" if i % 2 == 0 else "rejected")
                for i, f in enumerate(findings)
            }
            result = _payload(
                await client.call_tool(
                    "submit_decisions", {"run_id": run_id, "decisions": decisions}
                )
            )

            while result.get("awaiting_review"):
                result = _payload(
                    await client.call_tool(
                        "submit_decisions",
                        {
                            "run_id": run_id,
                            "decisions": {
                                str(f["index"]): "approved"
                                for f in result["pending_findings"]
                            },
                        },
                    )
                )

            assert result["status"] == "completed"

            deliverable = _payload(
                await client.call_tool("get_deliverable", {"run_id": run_id})
            )
            recorded = _payload(await client.call_tool("get_decisions", {"run_id": run_id}))

        assert deliverable["sections"], "the register must exist after committing"

        if len(findings) > 1:
            assert {d["verdict"] for d in recorded} == {"approved", "rejected"}, (
                "per-item verdicts submitted over MCP must survive distinctly"
            )

    async def test_the_actor_records_that_a_program_decided(self, tmp_path) -> None:
        """A verdict reached by a program must be distinguishable from one reached by a
        person. The gate exists to put a responsible party behind the commit, and
        'approved' with no idea who approved it does not do that."""
        (tmp_path / "msa.md").write_text(MSA, encoding="utf-8")
        (tmp_path / "broken.tiff").write_bytes(b"not a document")

        async with Client(server) as client:
            started = _payload(
                await client.call_tool(
                    "start_run",
                    {
                        "corpus_name": _unique("mcp-actor"),
                        "document_paths": [
                            str(tmp_path / "msa.md"),
                            str(tmp_path / "broken.tiff"),
                        ],
                    },
                )
            )
            run_id = started["run_id"]
            await client.call_tool(
                "submit_decisions",
                {
                    "run_id": run_id,
                    "decisions": {
                        str(f["index"]): "approved" for f in started["pending_findings"]
                    },
                    "actor": "agent-under-test",
                },
            )
            recorded = _payload(await client.call_tool("get_decisions", {"run_id": run_id}))

        assert recorded
        assert all(d["actor"] == "agent-under-test" for d in recorded)


@pytest.mark.slow
class TestTheEntrypointActuallyLaunches:
    async def test_a_client_can_launch_it_over_stdio(self) -> None:
        """Everything above talks to the server object in-process, which proves the
        tools work and nothing about whether `python mcp_server.py` starts.

        That gap matters: stdio is how a desktop client actually launches this, and an
        import error, a stray `print` on stdout, or logging misdirected to stdout would
        corrupt the JSON-RPC stream while every in-process test stayed green. This is
        the only test that would notice.
        """
        params = StdioServerParameters(
            command=sys.executable,
            args=["mcp_server.py"],
            env={**os.environ, "PYTHONPATH": "."},
        )

        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools
                result = await session.call_tool("list_runs", {"limit": 1})

        assert {t.name for t in tools} >= {"start_run", "submit_decisions", "get_run"}
        assert not result.is_error


class TestItRefusesRatherThanGuessing:
    async def test_an_unknown_run_is_an_error_not_an_empty_result(self) -> None:
        """An empty success is the worst possible answer here: the caller is a model,
        and it will summarise "no findings" as "nothing to review"."""
        async with Client(server) as client:
            result = await client.call_tool(
                "get_run", {"run_id": "00000000-0000-0000-0000-000000000000"}
            )

        assert result.is_error, "a missing run must surface as an error"

    async def test_an_invalid_verdict_is_refused_before_it_reaches_the_gate(self) -> None:
        """'maybe' is not a verdict. Passing it through would record something the
        gate has no meaning for."""
        async with Client(server) as client:
            result = await client.call_tool(
                "submit_decisions",
                {"run_id": str(uuid.uuid4()), "decisions": {"0": "maybe"}},
            )

        assert result.is_error
        assert "approved" in str(result.content).lower()

    async def test_a_run_with_no_documents_is_refused(self) -> None:
        async with Client(server) as client:
            result = await client.call_tool(
                "start_run", {"corpus_name": "empty", "document_paths": []}
            )

        assert result.is_error
