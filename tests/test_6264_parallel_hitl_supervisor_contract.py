"""Focused worker-side contract checks for issue #6264."""

import pathlib

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_supervisor_is_enabled_and_bounded_by_default():
    config = yaml.safe_load((ROOT / 'config.yml').read_text())
    assert config['independent_parallel_hitl'] is True
    assert config['parallel_hitl_max_concurrency'] == 8


def test_both_agent_entrypoints_register_and_unregister_decision_router():
    for path in ('methods/indexer_agent.py', 'methods/indexer_predict_agent.py'):
        source = (ROOT / path).read_text()
        assert 'parallel_hitl_router.register(' in source
        assert 'parallel_hitl_router.unregister(thread_id)' in source
        assert 'independent_parallel_hitl=' in source
        assert '__parallel_hitl_root_thread_id__' in source


def test_router_uses_two_phase_exact_decision_transport():
    source = (ROOT / 'utils/parallel_hitl_router.py').read_text()
    assert 'parallel_hitl_decision_offer' in source
    assert 'parallel_hitl_decision_commit' in source
    assert '_registry.offer(thread_id, decision)' in source
    assert '_registry.commit(thread_id, decision)' in source
    assert 'decision_id' in source
    assert 'interrupt_id' in source


def test_router_expands_live_oauth_tokens_before_sdk_mailbox_delivery():
    source = (ROOT / 'utils/parallel_hitl_router.py').read_text()
    normalize_index = source.index("decision['_mcp_tokens'] = expand_mcp_token_aliases(")
    offer_index = source.index('_registry.offer(thread_id, decision)')
    commit_index = source.index('_registry.commit(thread_id, decision)')

    assert normalize_index < offer_index
    assert normalize_index < commit_index


def test_durable_auth_interrupt_consumes_legacy_callback_cache():
    source = (ROOT / 'methods' / 'agent_common.py').read_text()
    pause_builder = source[
        source.index('def build_mcp_auth_pause_result('):
        source.index('def build_mcp_auth_required_result(')
    ]
    custom_event = source[
        source.index('if name == "parallel_hitl_interrupt"'):
        source.index('event_key = f"agent_{name}"')
    ]

    assert 'mcp_auth_durable_interrupt_seen' in pause_builder
    assert 'parallel_hitl_run_state' in pause_builder
    assert 'return None' in pause_builder
    assert 'if auth_entries:' in custom_event
    assert 'self.mcp_auth_pause_payload = None' in custom_event
    assert 'self.mcp_auth_pause_message = None' in custom_event
    assert 'self.mcp_auth_durable_interrupt_seen = True' in custom_event
    assert 'self.parallel_hitl_run_state[' in custom_event
    common = custom_event[
        custom_event.index('common = {'):
        custom_event.index('auth_entries = [')
    ]
    assert '"root_thread_id": data.get("root_thread_id")' in common
    assert '"thread_id": data.get("thread_id")' not in common
    assert 'response_metadata={**item, **common}' in custom_event
    assert custom_event.index('self.mcp_auth_pause_payload = None') < custom_event.index(
        'type=EventTypes.mcp_authorization_required'
    )

    callback_factory = (ROOT / 'utils' / 'agent_execution_common.py').read_text()
    assert 'parallel_hitl_run_state: Dict[str, Any] = {}' in callback_factory
    assert 'elitea_callback.parallel_hitl_run_state = parallel_hitl_run_state' in callback_factory
    assert 'elitea_custom_callback.parallel_hitl_run_state = parallel_hitl_run_state' in callback_factory

    pause_predicate = source[
        source.index('def _is_mcp_auth_paused('):
        source.index('def emit_subagent_invocation_chip(')
    ]
    assert 'self.parallel_hitl_run_state.get(' in pause_predicate
    assert '"mcp_auth_durable_interrupt_seen", False' in pause_predicate
    assert pause_predicate.index('return False') < pause_predicate.index(
        'return self.mcp_auth_pause_payload is not None'
    )


def test_resolved_auth_is_persisted_as_completed_tool_trace():
    source = (ROOT / 'methods' / 'agent_common.py').read_text()
    custom_event = source[source.index('class EliteACustomCallback'):]
    assert 'def _persist_mcp_auth_decision(' in custom_event
    assert 'if name == "mcp_auth_decision"' in custom_event
    assert 'for key in HIERARCHY_METADATA_KEYS' in custom_event
    assert 'self._persist_mcp_auth_decision(data, metadata)' in custom_event
    assert 'EventTypes.agent_tool_end' in custom_event
    assert 'EVENTNODE_PARTIAL_RESPONSE_NAME' in custom_event
    assert 'decorate_tool_call_for_persistence' in custom_event
