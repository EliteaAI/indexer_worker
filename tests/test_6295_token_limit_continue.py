"""Regression coverage for token-limit continuation events (#6295)."""

import ast
import pathlib
from types import SimpleNamespace
import typing


def _load_finish_reason_extractor():
    source = (pathlib.Path(__file__).resolve().parents[1] / 'utils' / 'funcs.py').read_text()
    namespace = {
        'Dict': typing.Dict,
        'Optional': typing.Optional,
        'LLMResult': object,
        'log': SimpleNamespace(debug=lambda *_args, **_kwargs: None),
    }
    for node in ast.parse(source).body:
        is_target_func = isinstance(node, ast.FunctionDef) and node.name == 'extract_finish_reason'
        is_target_const = isinstance(node, ast.Assign) and any(
            getattr(target, 'id', '') == 'LENGTH_STOP_REASONS' for target in node.targets
        )
        if is_target_func or is_target_const:
            exec(compile(ast.Module([node], []), '<finish-reason>', 'exec'), namespace)
    return namespace['extract_finish_reason']


extract_finish_reason = _load_finish_reason_extractor()


def _load_confirmation_classifier():
    source = (pathlib.Path(__file__).resolve().parents[1] / 'utils' / 'funcs.py').read_text()
    namespace = {
        'Any': typing.Any,
        'Dict': typing.Dict,
        'Optional': typing.Optional,
    }
    function = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and node.name == 'should_emit_output_limit_confirmation'
    )
    exec(compile(ast.Module([function], []), '<confirmation-classifier>', 'exec'), namespace)
    return namespace['should_emit_output_limit_confirmation']


should_emit_output_limit_confirmation = _load_confirmation_classifier()


def _response(*, generations=None, llm_output=None):
    return SimpleNamespace(generations=generations or [], llm_output=llm_output or {})


def test_chat_completions_length_reason_is_normalized():
    generation = SimpleNamespace(
        generation_info={'finish_reason': 'length'},
        message=SimpleNamespace(response_metadata={}),
    )

    assert extract_finish_reason(_response(generations=[[generation]])) == 'length'


def test_responses_api_incomplete_metadata_is_normalized():
    generation = SimpleNamespace(
        generation_info=None,
        message=SimpleNamespace(response_metadata={
            'status': 'incomplete',
            'incomplete_details': {'reason': 'max_output_tokens'},
        }),
    )

    assert extract_finish_reason(_response(generations=[[generation]])) == 'length'


def test_empty_generations_can_report_responses_api_token_exhaustion():
    response = _response(llm_output={
        'status': 'incomplete',
        'incomplete_details': {'reason': 'max_output_tokens'},
    })

    assert extract_finish_reason(response) == 'length'


def test_non_limit_incomplete_reason_is_not_misclassified():
    response = _response(llm_output={
        'status': 'incomplete',
        'incomplete_details': {'reason': 'content_filter'},
    })

    assert extract_finish_reason(response) == 'content_filter'


def test_callback_emits_continue_without_requiring_a_thinking_step():
    source = (
        pathlib.Path(__file__).resolve().parents[1] / 'methods' / 'agent_common.py'
    ).read_text()
    tree = ast.parse(source)
    on_llm_end = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == 'on_llm_end'
    )
    function_source = ast.get_source_segment(source, on_llm_end)
    tail = function_source[function_source.rindex('finish_reason = extract_finish_reason'):]

    assert 'if self.thinking_steps:' not in tail
    assert 'EventTypes.agent_requires_confirmation' in tail
    assert '"thread_id": self.thread_id' in tail
    assert 'last_step = new_thinking_step' in function_source


def test_nested_length_completion_does_not_emit_root_continue():
    source = (
        pathlib.Path(__file__).resolve().parents[1] / 'methods' / 'agent_common.py'
    ).read_text()
    tree = ast.parse(source)
    on_llm_end = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == 'on_llm_end'
    )
    function_source = ast.get_source_segment(source, on_llm_end)

    assert 'should_emit_output_limit_confirmation' in function_source
    assert 'hierarchy_metadata' in function_source
    assert not should_emit_output_limit_confirmation(
        'length',
        {'parent_agent_path': [{'name': 'General Purpose', 'call_id': 'call-1'}]},
    )
    assert should_emit_output_limit_confirmation('length', {})


def test_direct_pipeline_llm_node_does_not_emit_root_continue():
    assert not should_emit_output_limit_confirmation(
        'length',
        {'langgraph_node': 'LLM1'},
    )
