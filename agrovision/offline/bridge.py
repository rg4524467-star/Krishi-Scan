"""
ONNX Runtime Web bridge for the Pyodide/stlite offline PWA.

`onnx` and `onnxruntime` have no installable Pyodide wheels, so Python cannot
run the ONNX models in the browser.  Instead we vendor the onnxruntime-web
WASM runtime (`ort.min.js` + the SIMD-threaded JSEP loader/wasm) inside the
stlite VFS and drive it from Python through the pyodide `js` proxy:

* load `ort.min.js` into the worker global scope with a global `eval`
  (it is a strict-mode UMD, so we wrap it in a function that returns `ort`);
* point `ort.env.wasm` at the vendored wasm (blob URL + `wasmBinary` bytes) so
  the backend never fetches anything - fully offline on any host/sub-path;
* create `InferenceSession`s from model bytes and run them (both async),
  bridging Python's synchronous API to JS promises with a poll loop
  (`time.sleep` yields to the worker event loop in Pyodide).

Everything here is a no-op on native Python - gate callers on
`agrovision.util.runtime.is_pyodide()`.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

_ORT_DIR = "agrovision/assets/ort"
_ORT_JS = f"{_ORT_DIR}/ort.min.js"
_ORT_MJS = f"{_ORT_DIR}/ort-wasm-simd-threaded.jsep.mjs"
_ORT_WASM = f"{_ORT_DIR}/ort-wasm-simd-threaded.jsep.wasm"

_ort = None


def is_pyodide() -> bool:
    import sys

    return sys.platform == "emscripten" or "pyodide" in sys.modules


def _read_vfs(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def _blob_url(data: bytes, mime: str) -> str:
    import js

    u8 = js.Uint8Array.new(data)
    blob = js.Blob.new([u8], {"type": mime})
    return js.URL.createObjectURL(blob)


def _await(promise_factory) -> object:
    """Run an async JS call to completion from synchronous Python.

    Stores the promise plus result/error slots on the worker global scope,
    then polls with ``time.sleep`` (which yields to the JS event loop in
    Pyodide) until the promise settles.
    """
    import time

    import js

    js.globalThis.__agro_pending = promise_factory()
    js.globalThis.__agro_res = js.undefined
    js.globalThis.__agro_err = js.undefined
    js.globalThis.__agro_pending.then(
        lambda value: setattr(js.globalThis, "__agro_res", value),
        lambda error: setattr(js.globalThis, "__agro_err", error),
    )
    while js.globalThis.__agro_res is js.undefined and js.globalThis.__agro_err is js.undefined:
        time.sleep(0.02)
    js.globalThis.__agro_pending = js.undefined
    if js.globalThis.__agro_err is not js.undefined:
        err = js.globalThis.__agro_err
        msg = getattr(err, "message", None)
        raise RuntimeError(f"onnxruntime-web error: {msg or err}")
    return js.globalThis.__agro_res


def _load_ort():
    """Idempotently load ort into the worker global scope and configure wasm."""
    global _ort
    if _ort is not None:
        return _ort
    import js

    src = _read_vfs(_ORT_JS).decode("utf-8")
    # strict-mode UMD: `var ort` stays function-local, so wrap and return it.
    ort = js.globalThis.eval(
        "(function(){" + src + "; return typeof ort !== 'undefined' ? ort : undefined;})()"
    )
    if ort is js.undefined or ort is None:
        raise RuntimeError("Failed to load onnxruntime-web into the worker global scope")

    mjs_url = _blob_url(_read_vfs(_ORT_MJS), "text/javascript")
    wasm_url = _blob_url(_read_vfs(_ORT_WASM), "application/wasm")
    ort.env.wasm.numThreads = 1
    ort.env.wasm.wasmBinary = js.Uint8Array.new(_read_vfs(_ORT_WASM))
    ort.env.wasm.wasmPaths = {"mjs": mjs_url, "wasm": wasm_url}
    _ort = ort
    return ort


def _tensor_to_js(arr: np.ndarray):
    import js

    ort = _load_ort()
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    u8 = js.Uint8Array.new(arr.tobytes())
    return ort.Tensor.new("float32", u8.buffer, [int(d) for d in arr.shape])


def _tensor_to_np(tensor) -> np.ndarray:
    import js

    dims = list(tensor.dims)
    data = tensor.data
    buf = getattr(data, "buffer", None)
    if buf is not None:
        raw = bytes(memoryview(buf))
    else:
        raw = bytes(js.to_py(data))
    return np.frombuffer(raw, dtype=np.float32).reshape(dims)


class JSSession:
    """Python wrapper around a JS ``ort.InferenceSession`` (duck-typed to the
    interface onnx_backend.py expects: ``run()`` and ``input_names``)."""

    def __init__(self, session):
        self._session = session

    @property
    def input_names(self) -> List[str]:
        return list(self._session.inputNames)

    @property
    def output_names(self) -> List[str]:
        return list(self._session.outputNames)

    def run(self, output_names: Optional[Sequence[str]], feeds: Dict[str, np.ndarray]):
        names = list(output_names) if output_names else self.output_names
        feeds_js = {name: _tensor_to_js(arr) for name, arr in feeds.items()}
        result = _await(lambda: self._session.run(names, feeds_js))
        return [_tensor_to_np(getattr(result, name)) for name in names]

    def release(self) -> None:
        try:
            _await(lambda: self._session.release())
        except Exception:
            pass


def create_session(model_bytes: bytes, external_data: Optional[Sequence[dict]] = None) -> JSSession:
    """Create a session from raw ONNX model bytes (external data allowed)."""
    import js

    _load_ort()
    options: dict = {}
    if external_data:
        options["externalData"] = [
            {"path": ext["path"], "data": js.Uint8Array.new(ext["data"])} for ext in external_data
        ]
    js_session = _await(
        lambda: js.globalThis.ort.InferenceSession.create(
            js.Uint8Array.new(model_bytes), options
        )
    )
    return JSSession(js_session)


def create_session_from_file(
    model_path: str, external_data: Optional[Sequence[dict]] = None
) -> JSSession:
    return create_session(_read_vfs(model_path), external_data=external_data)