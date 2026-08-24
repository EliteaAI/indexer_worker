"""Toolkit-family runs must round-trip execution_generation on every event (#6345).

elitea_core stamps an execution_generation on the assistant message row for every chat_predict,
including toolkit ones, and drops any streamed callback whose payload does not carry it back
(is_current_execution). The toolkit task built its NodeEventInterface without the generation, so
every partial_message the EliteACallback emits — the events carrying tool_calls/thinking_steps —
was discarded, leaving Run History with request params and a final result but no trace.

Run from this directory (`cd tests && python3 -m pytest test_6345_toolkit_trace_persistence.py`):
invoking pytest from the plugin root imports the plugin's module.py, which needs the pylon runtime
and fails collection -- the other test files here behave the same way.
"""

import importlib.util
import pathlib
import sys
import types


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_node_interface():
    """Load by path with a pylon stub: importing the package pulls in the pylon runtime."""
    pylon = types.ModuleType('pylon')
    pylon_core = types.ModuleType('pylon.core')
    pylon_tools = types.ModuleType('pylon.core.tools')
    pylon_tools.log = types.SimpleNamespace(
        error=lambda *_a, **_k: None,
        debug=lambda *_a, **_k: None,
        info=lambda *_a, **_k: None,
    )
    sys.modules.update({
        'pylon': pylon, 'pylon.core': pylon_core, 'pylon.core.tools': pylon_tools,
    })
    spec = importlib.util.spec_from_file_location(
        'indexer_node_interface_6345', ROOT / 'utils' / 'node_interface.py',
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _CapturingEventNode:
    def __init__(self):
        self.events = []

    def emit(self, name, payload):
        self.events.append((name, payload))


def _build_interface(module, execution_generation):
    event_node = _CapturingEventNode()
    interface = module.NodeEventInterface(
        event_node=event_node,
        node_event_name='indexer_event',
        stream_id='stream-1',
        message_id='message-1',
        sio_event='chat_predict',
        question_id='question-1',
        execution_generation=execution_generation,
    )
    return interface, event_node


def test_emitted_events_carry_the_generation_at_top_level():
    module = _load_node_interface()
    interface, event_node = _build_interface(module, 'gen-1')

    interface.emit(type=module.EventTypes.agent_start)

    _, payload = event_node.events[-1]
    assert payload['execution_generation'] == 'gen-1'


def test_callback_built_events_inherit_the_generation():
    """EliteACallback builds its partial_message by hand, spreading payload_additional_kwargs."""
    module = _load_node_interface()
    interface, _ = _build_interface(module, 'gen-1')

    event = module.NodeEvent(
        type=module.EventTypes.partial_message,
        stream_id=interface.stream_id,
        message_id=interface.message_id,
        response_metadata={'tool_calls': {'run-1': {}}, 'thinking_steps': []},
        content=None,
        **interface.payload_additional_kwargs,
    ).model_dump(mode='json')

    assert event['execution_generation'] == 'gen-1'


def test_missing_generation_reproduces_the_discarded_shape():
    module = _load_node_interface()
    interface, event_node = _build_interface(module, None)

    interface.emit(type=module.EventTypes.agent_start)

    _, payload = event_node.events[-1]
    assert payload['execution_generation'] is None


def test_toolkit_task_passes_the_generation_into_its_interface():
    source = (ROOT / 'methods' / 'indexer_test_toolkit.py').read_text()
    task = source[
        source.index('def _indexer_test_toolkit_tool_task('):
        source.index('def _indexer_test_mcp_connection_task(')
    ]
    interface_call = task[
        task.index('node_interface = NodeEventInterface('):
        task.index('node_interface.emit(')
    ]

    assert 'execution_generation=execution_generation' in interface_call
