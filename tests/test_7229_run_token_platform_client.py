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
