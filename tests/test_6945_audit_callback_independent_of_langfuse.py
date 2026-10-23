"""The audit callback attaches independently of Langfuse (#6945, C-1009).

Before, the audit callback was returned by create_langfuse_callback only as a fallback when no
Langfuse config existed at all. A tenant with Langfuse configured therefore lost its audit trail,
and so did one whose Langfuse config was incomplete. The run now builds the two handlers
separately and the callers attach both.
"""

import importlib.util
import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

FULL_LANGFUSE = {'base_url': 'http://lf', 'public_key': 'pk', 'secret_key': 'sk'}
TASK_META = {'project_id': 7, 'user_context': {'user_id': 3, 'user_email': 'a@b.c'}}


class _AuditCallback:
    def __init__(self, user_id=None, user_email=None, project_id=None):
        self.user_id, self.user_email, self.project_id = user_id, user_email, project_id


class _LangfuseHandler:
    def __init__(self, **_kwargs):
        pass


@pytest.fixture
def lf(monkeypatch):
    """Load utils/langfuse_callback.py by path with pylon, tools, langfuse and tracing stubs."""
    state = types.SimpleNamespace(tracing_enabled=True, audit_enabled=True)
    tracing_module = types.SimpleNamespace()
    tracing_module.config = {}

    def sync_tracing_module():
        tracing_module.enabled = state.tracing_enabled
        tracing_module.config = {'audit_trail': {'enabled': state.audit_enabled}}

    log = types.SimpleNamespace(**{n: (lambda *_a, **_k: None) for n in ('info', 'debug', 'warning', 'error')})
    stubs = {
        'pylon': types.ModuleType('pylon'),
        'pylon.core': types.ModuleType('pylon.core'),
        'pylon.core.tools': types.SimpleNamespace(log=log),
        'tools': types.SimpleNamespace(this=types.SimpleNamespace(
            for_module=lambda _name: (sync_tracing_module(), types.SimpleNamespace(module=tracing_module))[1],
        )),
        'plugins': types.ModuleType('plugins'),
        'plugins.tracing': types.ModuleType('plugins.tracing'),
        'plugins.tracing.utils': types.ModuleType('plugins.tracing.utils'),
        'plugins.tracing.utils.audit_langchain_callback': types.SimpleNamespace(
            AuditLangChainCallback=_AuditCallback,
        ),
        'langfuse': types.SimpleNamespace(Langfuse=lambda **_k: object()),
        'langfuse.langchain': types.SimpleNamespace(CallbackHandler=_LangfuseHandler),
    }
    for name, stub in stubs.items():
        monkeypatch.setitem(sys.modules, name, stub)
    spec = importlib.util.spec_from_file_location('langfuse_callback_6945', ROOT / 'utils' / 'langfuse_callback.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.state = state
    return module


def _handlers(lf, langfuse_config):
    """What a caller attaches: the Langfuse handler (if any) plus the run audit handler (if any)."""
    _client, langfuse_handler, _attrs = lf.create_langfuse_callback(
        langfuse_config, metadata={'project_id': '7'}, user_id='3',
    )
    audit_handler = lf.create_run_audit_callback(TASK_META)
    return [h for h in (langfuse_handler, audit_handler) if h is not None]


def test_langfuse_and_audit_both_attach(lf):
    handlers = _handlers(lf, FULL_LANGFUSE)

    assert [type(h).__name__ for h in handlers] == ['EliteaLangfuseCallbackHandler', '_AuditCallback']


def test_incomplete_langfuse_config_still_attaches_audit(lf):
    handlers = _handlers(lf, {'base_url': 'http://lf'})

    assert [type(h).__name__ for h in handlers] == ['_AuditCallback']


def test_no_langfuse_attaches_audit_only(lf):
    assert [type(h).__name__ for h in _handlers(lf, None)] == ['_AuditCallback']


def test_no_langfuse_and_audit_disabled_attaches_nothing(lf):
    lf.state.audit_enabled = False

    assert _handlers(lf, None) == []


def test_langfuse_only_when_audit_disabled(lf):
    lf.state.audit_enabled = False

    assert [type(h).__name__ for h in _handlers(lf, FULL_LANGFUSE)] == ['EliteaLangfuseCallbackHandler']


def test_create_langfuse_callback_no_longer_returns_the_audit_handler(lf):
    assert lf.create_langfuse_callback(None) == (None, None, None)


def test_run_audit_callback_carries_attribution(lf):
    handler = lf.create_run_audit_callback(TASK_META)

    assert (handler.user_id, handler.user_email, handler.project_id) == (3, 'a@b.c', 7)


def test_run_audit_callback_falls_back_to_chat_project(lf):
    handler = lf.create_run_audit_callback({'chat_project_id': 9, 'user_context': {}})

    assert handler.project_id == 9


@pytest.mark.parametrize('method', ['indexer_agent.py', 'indexer_predict_agent.py'])
def test_both_run_paths_attach_the_audit_callback(method):
    source = (ROOT / 'methods' / method).read_text()

    assert 'create_run_audit_callback(tasknode_task.meta)' in source
    assert source.count('callbacks.append(run_audit_callback)') == 1
