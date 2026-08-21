"""The audio backend switch (#6323): LiteLLM proxy vs the platform gateway.

The three voice methods used to hold an absolute URL for a LiteLLM proxy inside
this container. The platform removed LiteLLM, so those calls must be able to go
to elitea-llm-gateway instead. The switch defaults to OFF because a hybrid
deployment still runs the local proxy and audio WORKS there today.

These tests pin the three things that change together when it is turned on, and
the fact that NOTHING changes when it is not. Each one fails if the switch stops
flipping a behaviour, which is the whole risk: a URL that moves while the model
name or the project header does not produces a 404 or a bill on the wrong
project, and both look like a provider problem from the outside.

Loaded by source so the suite runs without the pylon runtime. Run from this
directory: `cd tests && python3 -m pytest test_6323_audio_backend_switch.py`.
"""

import pathlib
import sys
import types


def _load_backend(configured_base_url):
    """Import utils/audio_backend.py with pylon's imports stubbed out."""
    # `this.descriptor.config` is the plugin's own module config at runtime.
    descriptor = types.SimpleNamespace(
        config={'audio_llm_base_url': configured_base_url}
    )
    tools = types.ModuleType('tools')
    tools.this = types.SimpleNamespace(descriptor=descriptor)

    pylon = types.ModuleType('pylon')
    pylon_core = types.ModuleType('pylon.core')
    pylon_tools = types.ModuleType('pylon.core.tools')
    pylon_tools.log = types.SimpleNamespace(
        debug=lambda *a, **k: None, warning=lambda *a, **k: None,
    )

    saved = {name: sys.modules.get(name) for name in
             ('tools', 'pylon', 'pylon.core', 'pylon.core.tools')}
    sys.modules.update({
        'tools': tools, 'pylon': pylon,
        'pylon.core': pylon_core, 'pylon.core.tools': pylon_tools,
    })
    try:
        source = (pathlib.Path(__file__).resolve().parents[1]
                  / 'utils' / 'audio_backend.py').read_text()
        namespace = {'__name__': 'audio_backend_under_test'}
        exec(compile(source, 'audio_backend.py', 'exec'), namespace)  # noqa: S102
        return namespace
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


GATEWAY = 'https://dev.elitea.ai/llm/v1'


def test_unset_keeps_every_legacy_behaviour():
    """The default must be a no-op. A hybrid deployment depends on it."""
    b = _load_backend('')
    assert b['audio_base_url']() == 'http://127.0.0.1:8081/v1'
    assert b['uses_platform_gateway']() is False
    # LiteLLM's addressable name IS the project-prefixed group.
    assert b['audio_model_name'](42, 'whisper-1') == '42_whisper-1'
    # And no project header: LiteLLM resolves the project from the virtual key.
    assert b['audio_headers'](42, 'sk-key') == {'Authorization': 'Bearer sk-key'}
    assert b['audio_ws_url']('/realtime?model=m').startswith('ws://127.0.0.1:8081/v1/realtime')


def test_gateway_sends_the_bare_model_name():
    """The prefixed form names no configuration row, so it 404s."""
    b = _load_backend(GATEWAY)
    assert b['uses_platform_gateway']() is True
    assert b['audio_model_name'](42, 'whisper-1') == 'whisper-1'


def test_gateway_carries_the_project_selector():
    """Without it the edge bills the CALLER's personal project, not this one."""
    b = _load_backend(GATEWAY)
    headers = b['audio_headers'](42, 'tok')
    assert headers['X-Project-Id'] == '42'
    assert headers['Authorization'] == 'Bearer tok'


def test_gateway_websocket_scheme_follows_the_base_url():
    """Dialling ws:// at an https origin fails the upgrade."""
    b = _load_backend(GATEWAY)
    url = b['audio_ws_url']('/realtime?model=whisper-1&intent=transcription')
    assert url == 'wss://dev.elitea.ai/llm/v1/realtime?model=whisper-1&intent=transcription'

    plain = _load_backend('http://gateway.internal:8080/llm/v1')
    assert plain['audio_ws_url']('/realtime?model=m').startswith('ws://gateway.internal:8080/llm/v1/')


def test_a_trailing_slash_does_not_double_up():
    """An operator-authored value is as likely to carry one as not."""
    b = _load_backend(GATEWAY + '/')
    assert b['audio_base_url'] () == GATEWAY
    assert b['audio_ws_url']('/realtime?model=m') == \
        'wss://dev.elitea.ai/llm/v1/realtime?model=m'
