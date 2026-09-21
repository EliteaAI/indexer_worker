"""Execute the owning payload builder with transport/usage owner seams stubbed."""
import ast
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict
import pytest

source = Path(__file__).resolve().parents[1]/'utils/agent_execution_common.py'
tree = ast.parse(source.read_text())
tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_child_launch_payloads']
namespace = {'Dict': Dict, 'Any': Any, 'PREDICT_RUN_ID_KWARGS_KEY': '_elitea_predict_run_id',
             'ROOT_ENTITY_KWARGS_KEY': 'root_entity', 'entity_from_application': lambda app: None}
exec(compile(tree, str(source), 'exec'), namespace)
build = namespace['build_child_launch_payloads']


def test_explicit_child_removes_parent_auto_and_keeps_own_effort():
    parent = {'llm': {'kwargs': {'model': None, 'selection': {'mode': 'auto'}, 'api_key': 'fixture-only'}}, 'conversation_id': 2}
    spec = {'input': {'task': 'child task'}, 'version_details': {'agent_type': 'agent', 'llm_settings': {'model_name': 'chosen', 'reasoning_effort': 'high'}}}
    result = build(parent, [spec])[0]['child_payload']
    assert result['llm']['kwargs']['model'] == 'chosen'
    assert 'selection' not in result['llm']['kwargs']
    assert result['llm']['kwargs']['reasoning_effort'] == 'high'
    assert parent['llm']['kwargs']['selection'] == {'mode': 'auto'}


@pytest.mark.parametrize('parent_type', ['agent', 'pipeline'])
def test_auto_child_under_fixed_parent_owns_fresh_task(parent_type):
    parent = {'llm': {'kwargs': {'model': 'parent-model'}}, 'application': {'version_details': {'agent_type': parent_type}}, 'routing_principal':{'project_id':7,'user_id':2}}
    original = {'application_id': 9, 'version_details': {'agent_type': 'agent', 'instructions':'', 'llm_settings': {'selection': {'mode': 'auto'}}}, 'input': {'task': 'delegated task'}}
    specs = [{**deepcopy(original), 'child_thread_id': ident} for ident in ['child1', 'child2']]
    result = build(parent, specs)
    for row in result:
        child = row['child_payload']
        assert child['llm']['kwargs']['model'] is None
        assert child['llm']['kwargs']['selection']['mode'] == 'auto'
        assert child['user_input'] == 'delegated task' and child['chat_history'] == []
        assert child['application']['version_details']['instructions']==''
        assert child['routing_principal']=={'project_id':7,'user_id':2}
    assert result[0]['child_payload']['thread_id'] != result[1]['child_payload']['thread_id']
    result[0]['child_payload']['llm']['kwargs']['selection']['mode'] = 'changed'
    assert result[1]['child_payload']['llm']['kwargs']['selection']['mode'] == 'auto'
    assert parent['llm']['kwargs'] == {'model': 'parent-model'}


def test_auto_pipeline_child_rejected():
    with pytest.raises(ValueError, match='Pipelines'):
        build({'llm': {'kwargs': {'model': 'parent'}}}, [{'version_details': {'agent_type': 'pipeline', 'llm_settings': {'selection': {'mode': 'auto'}}}}])


def test_legacy_embedded_child_under_auto_rejected_before_dispatch(monkeypatch):
    # SDK is supplied by sdk_plugin at runtime, not Worker unit-test requirements.
    # Assert that Worker raises that owner's exact exception; the real SDK type
    # and message are covered in the explicit cross-repository integration suite.
    import sys
    from types import ModuleType
    class ChildModelRequired(ValueError):
        pass
    exceptions = ModuleType('elitea_sdk.runtime.exceptions')
    exceptions.AutoRoutingChildModelRequired = ChildModelRequired
    monkeypatch.setitem(sys.modules, 'elitea_sdk.runtime.exceptions', exceptions)
    parent = {'llm': {'kwargs': {'selection': {'mode': 'auto'}}}}
    with pytest.raises(ChildModelRequired) as error:
        build(parent, [{'version_details': {'agent_type': 'agent', 'llm_settings': None}}])
    assert type(error.value) is ChildModelRequired
    assert error.value.args == ()


def test_fixed_parent_legacy_inheritance_preserves_run_and_model():
    parent = {'llm': {'kwargs': {'model': 'chosen', 'reasoning_effort': 'high'}},
              '_elitea_predict_run_id': 'trusted-run'}
    child = build(parent, [{'child_thread_id': 'separate-child',
                           'version_details': {'agent_type': 'agent', 'llm_settings': None}}])[0]['child_payload']
    assert child['llm']['kwargs']['model'] == 'chosen'
    assert child['llm']['kwargs']['reasoning_effort'] == 'high'
    assert child['_elitea_predict_run_id'] == 'trusted-run'
    assert child['thread_id'] == 'separate-child'


def test_worker_installs_signer_from_server_principal_not_request_headers(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    path=source
    tree=ast.parse(path.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='install_routing_context_signer')
    for entry in ('indexer_predict_agent.py','indexer_agent.py'):
        entry_tree=ast.parse((source.parents[1]/'methods'/entry).read_text())
        assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
                   and n.func.id=='install_routing_context_signer' for n in ast.walk(entry_tree))
    import sys
    from types import ModuleType
    module = ModuleType('arbiter.rpcnode'); module.RpcNode = Mock()
    monkeypatch.setitem(sys.modules, 'arbiter.rpcnode', module)
    node = module.RpcNode.return_value
    node.proxy.restricted_sign_routing_context.return_value = 'signed'
    event = object()
    client=SimpleNamespace();ns={}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
    ns['install_routing_context_signer'](client,event,{'project_id':7,'user_id':2})
    assert client._routing_context_signer(context={},tools=[],scope_id='scope',invocation_id='run')=='signed'
    module.RpcNode.assert_called_once_with(event, id_prefix="indexer_", proxy_timeout=10)
    node.start.assert_called_once()
    node.stop.assert_called_once()
    assert node.proxy.restricted_sign_routing_context.call_args.kwargs['user_id']==2
    node.proxy.restricted_sign_routing_context.side_effect = TimeoutError('unavailable')
    import pytest
    with pytest.raises(TimeoutError):
        client._routing_context_signer(context={},tools=[],scope_id='scope',invocation_id='run')
    assert node.stop.call_count == 2


def test_client_class_is_ready_before_agent_fork_admission():
    path=source.parents[1]/'module.py'
    tree=ast.parse(path.read_text())
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='init')
    imports=[n for n in ast.walk(fn) if isinstance(n,ast.ImportFrom) and n.module=='elitea_sdk.runtime.clients.client']
    starts=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and ast.unparse(n.func)=='self.agent_task_node.start']
    assert len(imports)==len(starts)==1 and imports[0].lineno<starts[0].lineno


def test_task_projection_only_marks_server_payload_and_preserves_hitl_edit():
    from types import SimpleNamespace
    from langchain_core.messages import HumanMessage
    for name in ('indexer_predict_agent.py','indexer_agent.py'):
        path=source.parents[1]/'methods'/name;tree=ast.parse(path.read_text())
        guard=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and 'elitea_routing_content' in ast.unparse(n))
        for edit in (False,True):
            message=HumanMessage(content='generation metadata. Hi')
            ns={'projection':{'task':[{'type':'text','text':'Hi'}]},'hitl_resume':edit,'user_message':message}
            exec(compile(ast.Module(body=[guard],type_ignores=[]),str(path),'exec'),ns)
            assert ('elitea_routing_content' in message.additional_kwargs) is not edit
            assert message.content=='generation metadata. Hi'


def test_both_worker_entrypoints_bind_the_trusted_run_to_graph_config():
    for entry in ('indexer_agent.py','indexer_predict_agent.py'):
        path=source.parents[1]/'methods'/entry;tree=ast.parse(path.read_text())
        value=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign)
                   and any(isinstance(t,ast.Name) and t.id=='invoke_config' for t in n.targets))
        configurable=next(v for k,v in zip(value.keys,value.values) if getattr(k,'value',None)=='configurable')
        expression=next(v for k,v in zip(configurable.keys,configurable.values)
                        if getattr(k,'value',None)=='elitea_routing_run_id')
        assert eval(compile(ast.Expression(expression),str(path),'eval'),
                    {'kwargs':{'_elitea_predict_run_id':'trusted-run'},'PREDICT_RUN_ID_KWARGS_KEY':'_elitea_predict_run_id'})=='trusted-run'


def history_builder():
    from typing import List, Optional, Union
    from langchain_core.messages import HumanMessage, SystemMessage
    ns = {'List':List, 'Optional':Optional, 'Union':Union, 'Dict':Dict, 'Any':Any,
          'HumanMessage':HumanMessage, 'SystemMessage':SystemMessage,
          'ATTACHMENT_SYSTEM_MESSAGE_TEMPLATE':'[ATTACHMENTS] {conversation_id}',
          'strip_stale_filepath_image_chunks':lambda x: None,
          'strip_image_chunks_from_assistant_messages':lambda x: None,
          'has_images_in_messages':lambda *a: False}
    for path, name in [(source.parent/'funcs.py','prepend_attachment_system_message'),
                       (source,'prepare_invoke_input')]:
        fn = next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
    return ns['prepare_invoke_input']


def test_history_projection_survives_real_message_conversion_without_generation_pruning():
    from langchain_core.messages import HumanMessage, SystemMessage, convert_to_messages
    prepare = history_builder()
    authored = '<runtime_context>authored text</runtime_context>'
    history = [{'role':'user','content':[{'type':'text','text':'<runtime_context>server owned</runtime_context>'},
                                       {'type':'text','text':authored}]},
               {'role':'assistant','content':'Ready to code.'},
               {'role':'user','content':[{'type':'text','text':'context-only row'}]}]
    projected = [{'role':'user','content':[{'type':'text','text':authored}]}, history[1],
                 {'role':'user','content':[]}]
    original=deepcopy(history)
    messages=convert_to_messages(prepare(history,HumanMessage(content='Go'),conversation_id='622',
        routing_projection={'history':projected})['messages'])
    assert isinstance(messages[0],SystemMessage) and messages[0].additional_kwargs['elitea_routing_content']==''
    assert messages[1].content==original[0]['content']
    assert messages[1].additional_kwargs['elitea_routing_content']==projected[0]['content']
    assert messages[3].additional_kwargs['elitea_routing_content']==[]
    assert history==original
    # Existing substantive System content is retained in routing when merged.
    merged=convert_to_messages(prepare([{'role':'system','content':'Authored operating constraint'}],
        HumanMessage(content='Hi'),conversation_id='622',routing_projection={})['messages'])[0]
    assert merged.additional_kwargs['elitea_routing_content']=='Authored operating constraint'
    assert '[ATTACHMENTS]' in merged.content and 'Authored operating constraint' in merged.content


def test_history_projection_rejects_misalignment_and_manual_path_is_unchanged():
    from langchain_core.messages import HumanMessage
    prepare=history_builder();history=[{'role':'user','content':'old task'}];task=HumanMessage(content='new task')
    for projected in ([], [{'role':'assistant','content':'wrong owner'}]):
        with pytest.raises(ValueError,match='Routing history'):
            prepare(history,task,routing_projection={'history':projected})
    manual=prepare(history,task,conversation_id='622')['messages']
    assert manual==[{'role':'system','content':'[ATTACHMENTS] 622'},*history,task]
    assert prepare([],task,conversation_id='622',routing_projection={'history':[]})['messages']==[task]


@pytest.mark.parametrize('partial', ['', 'Existing answer ending\n'])
def test_token_continue_wire_projection_preserves_alignment(partial):
    from langchain_core.messages import HumanMessage, convert_to_messages
    # Concrete Core output contract. The actual Core helper -> Worker roundtrip
    # remains covered by the explicit cross-repository integration suite.
    history = [{'role': 'assistant', 'content': 'older answer'},
               {'role': 'user', 'content': 'platform context. Original task'}]
    projected = [{'role': 'assistant'},
                 {'role': 'user', 'content': [{'type': 'text', 'text': 'Original task'}]}]
    if partial:
        history.append({'role': 'assistant', 'content': partial.rstrip('\n')})
        projected.append({'role': 'assistant'})
    original = deepcopy(history)
    messages = convert_to_messages(history_builder()(
        history, HumanMessage(content='Continue the answer'), conversation_id='622',
        routing_projection={'history': projected})['messages'])
    assert messages[2].content == 'platform context. Original task'
    assert messages[2].additional_kwargs['elitea_routing_content'] == projected[1]['content']
    if partial:
        assert messages[3].content == partial.rstrip('\n')
    assert history == original
