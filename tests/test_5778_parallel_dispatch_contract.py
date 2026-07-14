"""Focused tests for the explicit depth-3 hybrid dispatch contract."""

import importlib.util
import pathlib
import sys
import types

from utils.parallel_dispatch_contract import (
    durable_dispatch_allowed,
    normalize_hitl_pause,
)


def test_durable_dispatch_only_for_top_level_agent():
    assert durable_dispatch_allowed(True, False, 'react') is True
    assert durable_dispatch_allowed(True, True, 'react') is False
    assert durable_dispatch_allowed(True, False, 'pipeline') is False
    assert durable_dispatch_allowed(False, False, 'react') is False


def test_plural_only_pause_derives_compatible_singular_without_losing_list():
    plural = [{'tool_call_id': 'one'}, {'tool_call_id': 'two'}]
    singular, normalized = normalize_hitl_pause(None, plural)
    assert singular == plural[0]
    assert normalized == plural


def test_singular_pause_derives_complete_one_item_list():
    singular = {'tool_call_id': 'one'}
    normalized_singular, plural = normalize_hitl_pause(singular, None)
    assert normalized_singular == singular
    assert plural == [singular]


def test_outer_durable_child_path_prefixes_inner_leaf_path():
    pylon = types.ModuleType('pylon')
    pylon_core = types.ModuleType('pylon.core')
    pylon_tools = types.ModuleType('pylon.core.tools')
    pylon_tools.log = types.SimpleNamespace(error=lambda *_a, **_k: None, debug=lambda *_a, **_k: None)
    sys.modules.update({
        'pylon': pylon, 'pylon.core': pylon_core, 'pylon.core.tools': pylon_tools,
    })
    spec = importlib.util.spec_from_file_location(
        'indexer_node_interface',
        pathlib.Path(__file__).resolve().parents[1] / 'utils' / 'node_interface.py',
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    interface = module.NodeEventInterface.__new__(module.NodeEventInterface)
    interface.event_metadata_overlay = {
        'parent_agent_path': [
            {'name': 'B', 'call_id': 'call-b', 'sibling_ordinal': 2},
        ],
    }
    kwargs = {'response_metadata': {'metadata': {'parent_agent_path': [
        {'name': 'C', 'call_id': 'call-c', 'sibling_ordinal': 1},
    ]}}}

    interface._apply_event_metadata_overlay(kwargs)

    assert [
        item['call_id']
        for item in kwargs['response_metadata']['metadata']['parent_agent_path']
    ] == ['call-b', 'call-c']


def test_durable_outer_path_replaces_replayed_producer_self_hop(monkeypatch):
    import importlib.util
    import pathlib
    import sys
    import types

    pylon = types.ModuleType('pylon')
    pylon_core = types.ModuleType('pylon.core')
    pylon_tools = types.ModuleType('pylon.core.tools')
    pylon_tools.log = types.SimpleNamespace(debug=lambda *a, **k: None, error=lambda *a, **k: None)
    sys.modules.update({
        'pylon': pylon, 'pylon.core': pylon_core, 'pylon.core.tools': pylon_tools,
    })
    spec = importlib.util.spec_from_file_location(
        'indexer_node_interface_replay',
        pathlib.Path(__file__).resolve().parents[1] / 'utils' / 'node_interface.py',
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    interface = module.NodeEventInterface.__new__(module.NodeEventInterface)
    interface.event_metadata_overlay = {
        'parent_agent_path': [
            {'name': 'B', 'call_id': 'stable-b', 'sibling_ordinal': 2},
        ],
    }
    kwargs = {'response_metadata': {'metadata': {'parent_agent_path': [
        {'name': 'B', 'call_id': 'replayed-b'},
        {'name': 'C', 'call_id': 'call-c'},
    ]}}}

    interface._apply_event_metadata_overlay(kwargs)

    assert kwargs['response_metadata']['metadata']['parent_agent_path'] == [
        {'name': 'B', 'call_id': 'stable-b', 'sibling_ordinal': 2},
        {'name': 'C', 'call_id': 'call-c'},
    ]


def test_persisted_tools_and_thinking_steps_use_the_live_lineage_contract():
    pylon = types.ModuleType('pylon')
    pylon_core = types.ModuleType('pylon.core')
    pylon_tools = types.ModuleType('pylon.core.tools')
    pylon_tools.log = types.SimpleNamespace(debug=lambda *a, **k: None, error=lambda *a, **k: None)
    sys.modules.update({
        'pylon': pylon, 'pylon.core': pylon_core, 'pylon.core.tools': pylon_tools,
    })
    spec = importlib.util.spec_from_file_location(
        'indexer_node_interface_persistence',
        pathlib.Path(__file__).resolve().parents[1] / 'utils' / 'node_interface.py',
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    interface = module.NodeEventInterface.__new__(module.NodeEventInterface)
    interface.event_metadata_overlay = {
        'parent_agent_name': 'B',
        'child_thread_id': 'child-b2',
        'parent_agent_call_id': 'stable-b',
        'parent_agent_path': [
            {'name': 'B', 'call_id': 'stable-b', 'sibling_ordinal': 2},
        ],
    }
    producer = {
        'parent_agent_name': 'C',
        'parent_agent_call_id': 'call-c',
        'parent_agent_path': [
            {'name': 'B', 'call_id': 'replayed-b'},
            {'name': 'C', 'call_id': 'call-c'},
        ],
    }

    tool = interface.decorate_tool_call_for_persistence({
        'metadata': producer,
        'tool_meta': {'metadata': producer},
    })
    thinking = interface.decorate_thinking_step_for_persistence(dict(producer))

    expected = [
        {'name': 'B', 'call_id': 'stable-b', 'sibling_ordinal': 2},
        {'name': 'C', 'call_id': 'call-c'},
    ]
    assert tool['metadata']['parent_agent_path'] == expected
    assert tool['tool_meta']['metadata']['parent_agent_path'] == expected
    assert thinking['parent_agent_path'] == expected
    assert tool['metadata']['child_thread_id'] == 'child-b2'


def test_execution_generation_is_serialized_and_carried_to_child_and_reconcile():
    pylon = types.ModuleType('pylon')
    pylon_core = types.ModuleType('pylon.core')
    pylon_tools = types.ModuleType('pylon.core.tools')
    pylon_tools.log = types.SimpleNamespace(
        debug=lambda *a, **k: None, error=lambda *a, **k: None,
    )
    sys.modules.update({
        'pylon': pylon, 'pylon.core': pylon_core, 'pylon.core.tools': pylon_tools,
    })
    plugin_root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        'indexer_node_interface_generation',
        plugin_root / 'utils' / 'node_interface.py',
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    serialized = module.NodeEvent(
        type=module.EventTypes.agent_start,
        execution_generation='generation-1',
    ).model_dump(mode='json')
    source = (plugin_root / 'utils' / 'agent_execution_common.py').read_text()

    assert serialized['execution_generation'] == 'generation-1'
    assert "'execution_generation': parent_kwargs.get('execution_generation')" in source
    assert "'execution_generation'," in source
