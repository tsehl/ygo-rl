"""Construit une extension de lecture seule avec les en-têtes et le compilateur de l'oracle local."""
from pathlib import Path
import ctypes
import fcntl
import functools
import hashlib
import json
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
CORE = HERE / 'ygopro-core'
OUT = HERE.parent / '.tools'


@functools.lru_cache(maxsize=1)
def load():
    OUT.mkdir(exist_ok=True)
    binary = OUT / 'libeffect-inspector.so'
    stamp = OUT / 'effect-inspector.json'
    compiler = os.environ.get('CXX', 'g++')
    inputs = [HERE / 'effect_inspector.cpp', CORE / ('bin/release/libocgcore.dylib' if sys.platform == 'darwin' else 'bin/release/libocgcore.so'), Path(__file__)]
    inputs += sorted(CORE.rglob('*.h')) + sorted(CORE.rglob('*.hpp'))
    signature = hashlib.sha256(compiler.encode() + b''.join(p.read_bytes() for p in inputs) +
                               subprocess.check_output([compiler, '--version'])).hexdigest()
    with (OUT / 'effect-inspector.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            cached = json.loads(stamp.read_text())
        except (OSError, ValueError):
            cached = {}
        if not binary.exists() or not isinstance(cached, dict) or cached.get('signature') != signature:
            temporary = binary.with_suffix('.tmp.so')
            subprocess.run([compiler, '-std=c++17', '-O2', ('-dynamiclib' if sys.platform == 'darwin' else '-shared'), '-fPIC', '-fno-rtti',
                            '-I'+str(CORE), '-I'+str(CORE/'lua/src'), str(inputs[0]), '-o', str(temporary)], check=True)
            os.replace(temporary, binary)
            stamp_tmp = stamp.with_suffix('.tmp')
            stamp_tmp.write_text(json.dumps(dict(signature=signature)))
            os.replace(stamp_tmp, stamp)
    lib = ctypes.CDLL(str(binary))
    lib.YGO_EffectEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint)]
    lib.YGO_EffectEvent.restype = ctypes.c_int
    return lib


def event(duel, index, expected_code):
    code, value = ctypes.c_uint(), ctypes.c_uint()
    if not load().YGO_EffectEvent(duel, index, ctypes.byref(code), ctypes.byref(value)):
        raise RuntimeError('identité d’effet indisponible côté oracle')
    if code.value != expected_code:
        raise RuntimeError(f'identité d’effet oracle incohérente : {code.value} / {expected_code}')
    return value.value


if __name__ == '__main__':
    load()
    print('Inspecteur d’effets oracle construit et chargé')
