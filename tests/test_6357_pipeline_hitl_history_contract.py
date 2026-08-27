import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_pipeline_hitl_history_contract_is_forwarded_to_core():
    source = (ROOT / 'utils' / 'agent_execution_common.py').read_text()
    hitl_emit = source[
        source.index("if hitl_interrupt:"):
        source.index("# Emit pipeline_finish")
    ]

    assert "'interaction_type': hitl_interrupt.get('interaction_type')" in hitl_emit
    assert "'history_contract_version': hitl_interrupt.get('history_contract_version')" in hitl_emit
    assert "'interrupt_id': hitl_interrupt.get('interrupt_id')" in hitl_emit


def test_custom_event_allowlist_keeps_pipeline_hitl_history_fields():
    source = (ROOT / 'utils' / 'node_interface.py').read_text()
    hitl_fields = source[
        source.index('EventTypes.agent_hitl_interrupt.value: {'):
        source.index('EventTypes.parallel_hitl_interrupt.value: {')
    ]

    assert "'interaction_type', 'history_contract_version'" in hitl_fields
