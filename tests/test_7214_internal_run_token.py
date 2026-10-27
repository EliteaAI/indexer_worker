"""C-1009 (#7214): the run token leaves the task kwargs first, lives for the task only, and is hung on the run's
Elitea client with the origins it may go to. A malformed delivery is no token; nothing shows it."""

import copy
import pathlib
import re

import pytest

from utils import internal_run_token as irt

PLUGIN_ROOT = pathlib.Path(__file__).resolve().parents[1]
TOKEN = "eio1." + "a" * 40 + "." + "b" * 43


def delivery(**over):
    value = {"token": TOKEN, "origins": ["http://pylon_main:8080", "https://elitea.example"], "identity": "id-1"}
    value.update(over)
    return value


class Client:
    pass


def test_the_token_is_taken_out_of_the_kwargs_and_hung_on_the_client_for_the_task_only():
    kwargs = {"user_input": "hi", irt.TOKEN_KWARG: delivery()}
    marker = irt.begin_run_token(kwargs)
    assert irt.TOKEN_KWARG not in kwargs   # a child or reconcile payload built from these never carries it
    client = Client()
    auth = irt.attach_run_token(client)
    assert client.internal_run_auth is auth and auth.token == TOKEN
    assert auth.origins == ("http://pylon_main:8080", "https://elitea.example") and auth.identity == "id-1"
    irt.end_run_token(marker)
    later = Client()
    assert irt.attach_run_token(later) is None and not hasattr(later, "internal_run_auth")


@pytest.mark.parametrize("bad", [
    None, "eio1.x", {"token": "eyJ.pat", "origins": ["http://a"], "identity": "i"},
    delivery(token="eio1." + "a" * 5000), delivery(origins=[]), delivery(origins=["ftp://x"]),
    delivery(origins=["http://pylon_main:8080/app/2/mcp"]), delivery(origins=["http://a"] * 5), delivery(identity=""),
])
def test_a_malformed_delivery_is_no_token_and_is_still_removed(bad):
    kwargs = {irt.TOKEN_KWARG: bad}
    marker = irt.begin_run_token(kwargs)
    try:
        assert irt.TOKEN_KWARG not in kwargs and irt.attach_run_token(Client()) is None
    finally:
        irt.end_run_token(marker)


def test_a_legacy_run_has_no_token_and_its_kwargs_are_untouched():
    kwargs = {"user_input": "hi"}
    before = copy.deepcopy(kwargs)
    marker = irt.begin_run_token(kwargs)
    irt.end_run_token(marker)
    assert kwargs == before


def test_the_token_does_not_show_in_repr_and_copies_share_it():
    auth = irt.take_run_token({irt.TOKEN_KWARG: delivery()})
    assert TOKEN not in repr(auth) and copy.deepcopy(auth) is auth


@pytest.mark.parametrize("task", ["indexer_agent", "indexer_predict_agent"])
def test_both_agent_tasks_take_the_token_before_the_envelope_and_attach_it_to_the_client(task):
    source = (PLUGIN_ROOT / "methods" / f"{task}.py").read_text()
    body = source[source.index(f"def {task}("):]
    assert body.index("begin_run_token(kwargs)") < body.index("enforce_envelope(kwargs)")
    assert re.search(r"client = create_elitea_client\([^)]*\)\n\s+#[^\n]*\n\s+attach_run_token\(client\)", source)
    assert "end_run_token(run_token)" in body[:body.index("def _")]
