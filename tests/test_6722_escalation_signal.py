"""The regenerate/upgrade escalation signal reaches the graph run config (#6722)."""
import ast
from pathlib import Path

import pytest

root = Path(__file__).resolve().parents[1]
helper_ns = {}
exec(compile((root / 'utils/routing_escalation.py').read_text(), 'routing_escalation.py', 'exec'), helper_ns)
routing_escalation = helper_ns['routing_escalation']
KEY = helper_ns['ROUTING_ESCALATION_CONFIG_KEY']

GOOD = {'event': 'regenerate', 'from_message_id': 'msg-1'}


def test_config_key_matches_the_sdk_contract():
    assert KEY == 'elitea_routing_escalation'


@pytest.mark.parametrize('value,expected', [
    (GOOD, GOOD),
    ({'event': 'upgrade_control'}, {'event': 'upgrade_control', 'from_message_id': None}),
    ({'event': 'upgrade_control', 'from_message_id': 7}, {'event': 'upgrade_control', 'from_message_id': None}),
    ({'event': 'regenerate', 'from_message_id': 'm', 'extra': 'dropped'}, {'event': 'regenerate', 'from_message_id': 'm'}),
])
def test_well_formed_signal_is_normalised(value, expected):
    assert routing_escalation(value) == expected


@pytest.mark.parametrize('value', [None, {}, 'regenerate', ['regenerate'], {'from_message_id': 'm'},
                                   {'event': ''}, {'event': None}, {'event': 3}])
def test_malformed_or_absent_signal_is_omitted(value):
    assert routing_escalation(value) is None


def _config_for(entry, kwargs):
    """Run the real invoke_config assembly of one worker entrypoint."""
    path = root / 'methods' / entry
    tree = ast.parse(path.read_text())
    stmts = None
    for node in ast.walk(tree):
        body = getattr(node, 'body', None)
        if not isinstance(body, list):
            continue
        idx = next((i for i, n in enumerate(body) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'invoke_config' for t in n.targets)), None)
        if idx is not None and idx + 2 < len(body):
            stmts = body[idx:idx + 3]
            break
    assert stmts is not None
    assert isinstance(stmts[1], ast.Assign) and 'routing_escalation' in ast.unparse(stmts[1])
    assert isinstance(stmts[2], ast.If)
    ns = {'kwargs': kwargs, 'PREDICT_RUN_ID_KWARGS_KEY': '_elitea_predict_run_id', 'callbacks': [],
          'thread_id': 'thread-1', 'meta': {}, 'hitl_resume': False, 'mcp_auth_resume': False,
          'should_continue': False, 'current_state_hash': None,
          'routing_escalation': routing_escalation, 'ROUTING_ESCALATION_CONFIG_KEY': KEY}
    exec(compile(ast.Module(body=stmts, type_ignores=[]), str(path), 'exec'), ns)
    return ns['invoke_config']['configurable']


@pytest.mark.parametrize('entry', ['indexer_agent.py', 'indexer_predict_agent.py'])
def test_both_worker_entrypoints_bind_the_escalation_to_graph_config(entry):
    configurable = _config_for(entry, {'_elitea_predict_run_id': 'trusted-run', 'escalation': GOOD})
    assert configurable[KEY] == GOOD
    assert configurable['elitea_routing_run_id'] == 'trusted-run'


@pytest.mark.parametrize('entry', ['indexer_agent.py', 'indexer_predict_agent.py'])
@pytest.mark.parametrize('kwargs', [{}, {'escalation': None}, {'escalation': {}}, {'escalation': 'regenerate'},
                                    {'escalation': {'event': 3}}])
def test_absent_or_malformed_escalation_is_omitted_from_graph_config(entry, kwargs):
    assert KEY not in _config_for(entry, kwargs)
