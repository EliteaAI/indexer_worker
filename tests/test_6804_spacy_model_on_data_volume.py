"""EL-6804: the Unstructured spaCy model must live on the writable data volume.

Unstructured's own lazy installer moves the model into the interpreter's
site-packages, which is read-only on hardened pods (readOnlyRootFilesystem).
The worker installs the model into ``spacy_data`` - from the configured bootstrap
bundle depot first (like the old NLTK bundle), else the pinned wheel - and puts that
directory on ``sys.path`` so
``spacy.load("en_core_web_sm")`` resolves it without ever touching site-packages.
"""

import hashlib
import importlib.util
import io
import tarfile
from pathlib import Path
import sys
import types
from unittest.mock import Mock
import zipfile

import pytest


PLUGIN_ROOT = Path(__file__).parents[1]
VERSION = "3.8.0"
URL = "https://example.invalid/en_core_web_sm-3.8.0-py3-none-any.whl"


def _model_members(version=VERSION):
    # Mirrors the real wheel: the package dir (with versioned data subdir) + dist-info.
    return {
        "en_core_web_sm/__init__.py": b"",
        f"en_core_web_sm/en_core_web_sm-{version}/config.cfg": b"[nlp]\n",
        f"en_core_web_sm-{version}.dist-info/METADATA": b"Name: en_core_web_sm\n",
    }


def _wheel_bytes(version=VERSION):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as wheel:
        for name, data in _model_members(version).items():
            wheel.writestr(name, data)
    return buffer.getvalue()


def _unpack_model(target, version=VERSION):
    for name, data in _model_members(version).items():
        path = Path(target) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def _tar_bytes(members):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


class _FakeBootstrap:
    """Mimics bootstrap get_bundle: no depot configured -> raises, like the real resolver check."""

    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error or RuntimeError("RepoResolver is not for supported depot")
        self.calls = []

    def get_bundle(self, name, **kwargs):
        self.calls.append((name, kwargs))
        if self.payload is None:
            raise self.error
        if not kwargs["install_needed"]():
            return
        with tarfile.open(fileobj=io.BytesIO(self.payload), mode="r:gz") as archive:
            archive.extractall(kwargs["extract_target"])


def _load(monkeypatch, wheel=None, bootstrap=None):
    pylon = types.ModuleType("pylon")
    pylon_core = types.ModuleType("pylon.core")
    pylon_tools = types.ModuleType("pylon.core.tools")
    pylon_tools.log = types.SimpleNamespace(
        info=Mock(), warning=Mock(), error=Mock(), exception=Mock(), debug=Mock(),
    )
    pylon_tools.module = types.SimpleNamespace(ModuleModel=object)
    monkeypatch.setitem(sys.modules, "pylon", pylon)
    monkeypatch.setitem(sys.modules, "pylon.core", pylon_core)
    monkeypatch.setitem(sys.modules, "pylon.core.tools", pylon_tools)
    monkeypatch.setitem(sys.modules, "arbiter", types.ModuleType("arbiter"))

    tools = types.ModuleType("tools")
    tools.worker_core = types.SimpleNamespace()
    bootstrap = _FakeBootstrap() if bootstrap is None else bootstrap
    tools.this = types.SimpleNamespace(
        for_module=lambda name: types.SimpleNamespace(module=bootstrap) if name == "bootstrap" else None,
    )
    monkeypatch.setitem(sys.modules, "tools", tools)

    wheel = _wheel_bytes() if wheel is None else wheel
    tokenize = types.ModuleType("unstructured.nlp.tokenize")
    tokenize._SPACY_MODEL_VERSION = VERSION
    tokenize._SPACY_MODEL_URL = URL
    tokenize._SPACY_MODEL_SHA256 = hashlib.sha256(_wheel_bytes()).hexdigest()
    monkeypatch.setitem(sys.modules, "unstructured", types.ModuleType("unstructured"))
    monkeypatch.setitem(sys.modules, "unstructured.nlp", types.ModuleType("unstructured.nlp"))
    monkeypatch.setitem(sys.modules, "unstructured.nlp.tokenize", tokenize)

    spec = importlib.util.spec_from_file_location("indexer_worker_module_el6804", PLUGIN_ROOT / "module.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)

    urlopen = Mock(side_effect=lambda *_a, **_k: io.BytesIO(wheel))
    monkeypatch.setattr(loaded.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(sys, "path", list(sys.path))
    return loaded, urlopen, pylon_tools.log


def test_model_is_downloaded_onto_data_volume_and_path(monkeypatch, tmp_path):
    worker, urlopen, _log = _load(monkeypatch)
    target = str(tmp_path / "spacy")

    worker._prepare_spacy_model(target)

    urlopen.assert_called_once()
    assert urlopen.call_args.args[0] == URL
    assert worker._has_spacy_model(target, VERSION)
    assert sys.path[0] == target
    # The temporary download dir is cleaned up; only the installed model remains.
    assert sorted(p.name for p in Path(target).iterdir()) == [
        "en_core_web_sm", "en_core_web_sm-3.8.0.dist-info",
    ]


def test_present_model_skips_download(monkeypatch, tmp_path):
    worker, urlopen, _log = _load(monkeypatch)
    target = tmp_path / "spacy"
    _unpack_model(target)

    worker._prepare_spacy_model(str(target))

    urlopen.assert_not_called()
    assert sys.path[0] == str(target)


def test_stale_model_version_is_replaced_without_touching_other_files(monkeypatch, tmp_path):
    # An Unstructured upgrade bumps the pin; the old dist-info must go so metadata stays unambiguous.
    worker, _urlopen, _log = _load(monkeypatch)
    target = tmp_path / "spacy"
    _unpack_model(target, "3.7.1")
    (target / "unrelated.txt").write_text("keep")

    worker._prepare_spacy_model(str(target))

    assert not (target / "en_core_web_sm-3.7.1.dist-info").exists()
    assert not (target / "en_core_web_sm" / "en_core_web_sm-3.7.1").exists()
    assert worker._has_spacy_model(str(target), VERSION)
    assert (target / "unrelated.txt").read_text() == "keep"


def test_tampered_wheel_is_rejected(monkeypatch, tmp_path):
    worker, _urlopen, _log = _load(monkeypatch, wheel=b"not the pinned wheel")
    target = str(tmp_path / "spacy")

    with pytest.raises(RuntimeError, match="Hash mismatch"):
        worker._prepare_spacy_model(target)

    assert not worker._has_spacy_model(target, VERSION)


def test_path_is_registered_even_when_download_fails(monkeypatch, tmp_path):
    # A model placed on the volume later (e.g. by an operator) must still be found by on-demand retries.
    worker, urlopen, _log = _load(monkeypatch)
    urlopen.side_effect = OSError("offline")
    target = str(tmp_path / "spacy")

    with pytest.raises(OSError):
        worker._prepare_spacy_model(target)

    assert sys.path[0] == target


def test_model_is_installed_from_configured_bundle(monkeypatch, tmp_path):
    bootstrap = _FakeBootstrap(payload=_tar_bytes(_model_members()))
    worker, urlopen, _log = _load(monkeypatch, bootstrap=bootstrap)
    target = tmp_path / "spacy"

    worker._prepare_spacy_model(str(target))

    urlopen.assert_not_called()
    name, kwargs = bootstrap.calls[0]
    assert name == "en_core_web_sm-3.8.0.tar.gz"
    assert kwargs["processing"] == "tar_extract"
    assert worker._has_spacy_model(str(target), VERSION)
    # Staging dir is gone; only the model entries were adopted.
    assert sorted(p.name for p in target.iterdir()) == ["en_core_web_sm", "en_core_web_sm-3.8.0.dist-info"]


def test_missing_bundle_falls_back_to_pinned_wheel(monkeypatch, tmp_path):
    # No depot configured / bundle not published: behaves exactly like the wheel-only path.
    bootstrap = _FakeBootstrap(error=RuntimeError("404 Not Found"))
    worker, urlopen, log = _load(monkeypatch, bootstrap=bootstrap)
    target = str(tmp_path / "spacy")

    worker._prepare_spacy_model(target)

    assert len(bootstrap.calls) == 1
    urlopen.assert_called_once()
    assert worker._has_spacy_model(target, VERSION)
    log.warning.assert_called_once()


def test_wrong_bundle_asset_is_discarded(monkeypatch, tmp_path):
    # The GitHub resolver can return another release asset (the deno sandbox) for any name.
    deno_like = _tar_bytes({"bin/deno": b"ELF", "package.json": b"{}"})
    worker, urlopen, _log = _load(monkeypatch, bootstrap=_FakeBootstrap(payload=deno_like))
    target = tmp_path / "spacy"

    worker._prepare_spacy_model(str(target))

    urlopen.assert_called_once()
    assert sorted(p.name for p in target.iterdir()) == ["en_core_web_sm", "en_core_web_sm-3.8.0.dist-info"]


def test_bundle_for_other_version_falls_back(monkeypatch, tmp_path):
    bootstrap = _FakeBootstrap(payload=_tar_bytes(_model_members("3.7.1")))
    worker, urlopen, _log = _load(monkeypatch, bootstrap=bootstrap)
    target = tmp_path / "spacy"

    worker._prepare_spacy_model(str(target))

    urlopen.assert_called_once()
    assert not (target / "en_core_web_sm-3.7.1.dist-info").exists()
    assert worker._has_spacy_model(str(target), VERSION)


def test_present_model_skips_bundle(monkeypatch, tmp_path):
    bootstrap = _FakeBootstrap(payload=_tar_bytes(_model_members()))
    worker, _urlopen, _log = _load(monkeypatch, bootstrap=bootstrap)
    target = tmp_path / "spacy"
    _unpack_model(target)

    worker._prepare_spacy_model(str(target))

    assert bootstrap.calls == []
