"""C-1009 (#7229 item 1): the run token is the bearer of the run's whole platform client when Main sent no other
credential (an admitted run under enforce). A client with a personal or system token or a session keeps it exactly
as before; a run without a run token is untouched. The pgvector warm-up does not fire without a credential."""

import pathlib
import types

import pytest

from utils import internal_run_token as irt
from utils.context_guard import REJECTION_MESSAGES, rejection_message

PLUGIN_ROOT = pathlib.Path(__file__).resolve().parents[1]
TOKEN = "eio1." + "a" * 40 + "." + "b" * 43


@pytest.fixture(autouse=True)
def sdk_with_the_guard(monkeypatch):
    """The installed SDK keeps the run token out of the sandbox (``is_run_token_bearer``), unless a test says not."""
    monkeypatch.setitem(irt._SDK_GUARD, "ok", True)  # pylint: disable=W0212


class Client:
    """The parts of ``EliteAClient`` the attach reads and writes: the bearer, the session and the request headers."""

    def __init__(self, auth_token=None, auth_session=None, base_url="http://pylon_main:8080"):
        self.base_url = base_url
        self.auth_token, self.auth_session = auth_token, (auth_session if not auth_token else None)
        self.headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
        self.headers["X-Project-Id"] = "1"


def attached(client, delivered=True):
    kwargs = {irt.TOKEN_KWARG: {"token": TOKEN, "origins": ["http://pylon_main:8080"], "identity": "id"}} \
        if delivered else {}
    marker = irt.begin_run_token(kwargs)
    try:
        return irt.attach_run_token(client)
    finally:
        irt.end_run_token(marker)


def test_a_client_main_gave_no_credential_takes_the_run_token_as_its_bearer():
    client = Client()
    auth = attached(client)
    assert client.internal_run_auth is auth
    assert client.auth_token == TOKEN and client.headers == {"Authorization": f"Bearer {TOKEN}", "X-Project-Id": "1"}


@pytest.mark.parametrize("token,session", [("pat-value", None), (None, "session-reference")])
def test_a_client_with_its_own_credential_keeps_it_byte_for_byte(token, session):
    client = Client(token, session)
    before = (client.auth_token, client.auth_session, dict(client.headers))
    attached(client)
    assert (client.auth_token, client.auth_session, client.headers) == before
    assert client.internal_run_auth.token == TOKEN   # the internal MCP still gets the run token (#7214)


def test_without_a_run_token_nothing_changes():
    client = Client()
    assert attached(client, delivered=False) is None
    assert client.auth_token is None and client.headers == {"X-Project-Id": "1"}
    assert not hasattr(client, "internal_run_auth")


@pytest.mark.parametrize("path", ["methods/indexer_agent.py", "methods/indexer_predict_agent.py",
                                  "methods/indexer_test_toolkit.py"])
def test_every_admitted_task_hangs_the_run_token_on_its_client_right_after_building_it(path):
    text = (PLUGIN_ROOT / path).read_text()
    assert text.count("attach_run_token(client)") == 1 and text.count("begin_run_token(kwargs)") == 1


def _warm_module(asked):
    """``utils/pgvector_warm.py`` run from its source (its pylon and relative imports replaced), with ``request_warm``
    recording what it is asked."""
    source = (PLUGIN_ROOT / "utils" / "pgvector_warm.py").read_text()
    source = source.replace("from pylon.core.tools import log  # pylint: disable=E0611,E0401", "")
    source = source.replace("from ..methods.agent_common import pgvector_connstr_needed  # pylint: disable=C0415",
                            "pgvector_connstr_needed = lambda config: True")
    namespace = {"log": types.SimpleNamespace(warning=lambda *a, **k: None)}
    exec(compile(source, "pgvector_warm.py", "exec"), namespace)  # pylint: disable=W0122
    namespace["request_warm"] = lambda *args: asked.append(args)
    return namespace


def test_the_pgvector_warm_up_fires_only_with_a_credential():
    asked = []
    approver = _warm_module(asked)["agent_task_approver"]({"agent_memory_config": {"type": "postgres"}})

    def payload(**llm):
        return {"kwargs": {"llm": {"kwargs": {"project_id": 1, **llm}}}}
    assert approver("start", payload()) is True and asked == []   # a governed run: no token, no warm-up
    approver("start", payload(api_key="pat-value"))
    approver("start", payload(auth_session="session-reference"))
    assert [args[2] for args in asked] == ["pat-value", None]


def test_a_refused_sub_agent_fetch_is_explained_readably():
    assert rejection_message("DENIED_DEVELOPMENT_CHILD") == REJECTION_MESSAGES["DENIED_DEVELOPMENT_CHILD"]
    assert "sub-agent" in rejection_message("DENIED_DEVELOPMENT_CHILD")


def _langfuse_fetch():
    """``fetch_langfuse_config`` of methods/agent_common.py, run from its source with ``requests`` recorded."""
    import ast  # pylint: disable=C0415
    source = (PLUGIN_ROOT / "methods" / "agent_common.py").read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "fetch_langfuse_config")
    asked = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"items": []}

    namespace = {"Optional": __import__("typing").Optional, "Dict": dict, "Any": object,
                 "requests": types.SimpleNamespace(get=lambda url, **kwargs: asked.append(url) or Response()),
                 "log": types.SimpleNamespace(debug=lambda *a, **k: None, warning=lambda *a, **k: None),
                 "_unsecret_vault_references": lambda data, client: data}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "agent_common.py", "exec"), namespace)  # pylint: disable=W0122
    return namespace["fetch_langfuse_config"], asked


def test_a_run_on_its_run_token_does_not_ask_for_the_langfuse_credentials():
    fetch, asked = _langfuse_fetch()
    governed = Client()
    governed.project_id = 1
    attached(governed)
    assert fetch(governed) is None and asked == []
    legacy = Client("pat-value")
    legacy.project_id = 1
    attached(legacy)   # a client with its own credential asks as before
    fetch(legacy)
    assert len(asked) == 1


@pytest.mark.parametrize("base_url", ["https://elsewhere.example", "http://pylon_main:9090", "ftp://pylon_main:8080",
                                      None])
def test_the_run_token_is_never_the_bearer_of_a_client_for_another_origin(base_url):
    client = Client(base_url=base_url)
    attached(client)
    assert client.auth_token is None and "Authorization" not in client.headers


def test_a_base_url_with_a_path_prefix_on_the_tokens_origin_takes_it():
    client = Client(base_url="http://pylon_main:8080/elitea")
    attached(client)
    assert client.auth_token == TOKEN


def test_an_authorization_header_the_client_already_has_is_kept():
    client = Client()
    client.headers["authorization"] = "Bearer caller-value"
    attached(client)
    assert client.auth_token is None and client.headers["authorization"] == "Bearer caller-value"
    assert "Authorization" not in client.headers


def test_the_short_lived_pgvector_client_gets_the_run_token_only_without_a_credential():
    kwargs = {irt.TOKEN_KWARG: {"token": TOKEN, "origins": ["http://pylon_main:8080"], "identity": "id"}}
    marker = irt.begin_run_token(kwargs)
    try:
        assert irt.platform_bearer(None, {"deployment": "http://pylon_main:8080"}) == TOKEN
        assert irt.platform_bearer("pat-value", {"deployment": "http://pylon_main:8080"}) == "pat-value"
        assert irt.platform_bearer(None, {"deployment": "http://pylon_main:8080", "auth_session": "s"}) is None
        assert irt.platform_bearer(None, {"deployment": "https://elsewhere.example"}) is None
    finally:
        irt.end_run_token(marker)
    assert irt.platform_bearer(None, {"deployment": "http://pylon_main:8080"}) is None   # no run token: as before


class _ModelError(Exception):
    """What a model client raises for a refused call: the status and Main's body in the text."""


@pytest.mark.parametrize("code", ["CONTEXT_NOT_LIVE", "INTERNAL_TOKEN_EXPIRED", "INTERNAL_TOKEN_INVALID"])
def test_a_platform_call_refused_on_the_run_token_ends_the_run_readably(code):
    from utils.credential_resolver import governed_credential_code
    from utils.context_guard import rejection_message
    error = _ModelError(f"Error code: 403 - {{'error': '{code}', 'message': 'x'}}")
    wrapped = RuntimeError("agent failed")
    wrapped.__cause__ = error
    kwargs = {irt.TOKEN_KWARG: {"token": TOKEN, "origins": ["http://pylon_main:8080"], "identity": "id"}}
    marker = irt.begin_run_token(kwargs)
    try:
        assert governed_credential_code(error) == code and governed_credential_code(wrapped) == code
        assert rejection_message(code) != rejection_message("SOMETHING_NEW")
    finally:
        irt.end_run_token(marker)
    assert governed_credential_code(error) is None   # a run without a run token: a model error as before


def test_without_an_sdk_that_keeps_the_token_out_of_the_sandbox_the_run_token_is_never_the_bearer(monkeypatch):
    monkeypatch.setitem(irt._SDK_GUARD, "ok", False)  # pylint: disable=W0212
    client = Client()
    attached(client)
    assert client.auth_token is None and "Authorization" not in client.headers
    assert client.internal_run_auth.token == TOKEN   # the internal MCP still gets it (it never reaches the sandbox)


@pytest.mark.parametrize("path", ["methods/indexer_agent.py", "methods/indexer_predict_agent.py"])
def test_both_agent_tasks_give_the_pgvector_fallback_the_run_bearer(path):
    text = (PLUGIN_ROOT / path).read_text()
    call = text[text.index("pgvector_connstr = resolve_pgvector_connstr("):]
    call = call[:call.index(")", call.index("prefetched="))]
    assert "platform_bearer(api_token, client_args)" in call and "api_token, api_extra_headers" not in call


def test_the_card_names_the_sub_agent_the_sdk_says_was_refused():
    from utils.credential_resolver import governed_text
    error = types.SimpleNamespace(subagent="public helper")
    assert governed_text(error, "PUBLIC_AGENT_NOT_ADMITTED").endswith("(sub-agent “public helper”)")
    assert governed_text(Exception("x"), "PUBLIC_AGENT_NOT_ADMITTED") == rejection_message("PUBLIC_AGENT_NOT_ADMITTED")


def test_a_withheld_langfuse_configuration_is_counted_and_said(monkeypatch):
    from utils import credential_resolver
    said = []
    monkeypatch.setattr(credential_resolver, "_log_info", said.append)
    before = credential_resolver.CREDENTIAL_SLOT_RESOLUTION.total("skip", "TRACING_NOT_RELEASED")
    credential_resolver.record_tracing_withheld()
    assert credential_resolver.CREDENTIAL_SLOT_RESOLUTION.total("skip", "TRACING_NOT_RELEASED") == before + 1
    assert said == ["security_metric name=credential_slot_resolution_total result=skip reason_code=TRACING_NOT_RELEASED"]
