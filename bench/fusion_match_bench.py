"""Coût du calcul exact avec substitut ; option --reference chemin/fusion.py pour comparaison.
Exécuter sans autres tests actifs pour mesurer. Ce microbench ne mesure pas l'entraînement.
"""
import argparse,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'engine'))
from fusion import completions
p=argparse.ArgumentParser()
p.add_argument('--reference');p.add_argument('--b',type=int,default=32)
p.add_argument('--n',type=int,default=110);p.add_argument('--calls',type=int,default=50)
a=p.parse_args();rng=np.random.default_rng(0)
args=(jnp.array(rng.random((a.b,4,a.n))<.25),jnp.array(rng.random((a.b,a.n))<.08),
      jnp.array(rng.random((a.b,a.n))<.3),jnp.array(rng.integers(16,size=a.b),dtype=jnp.int32),
      jnp.tile(jnp.arange(a.n)[None],(a.b,1)))
fns={'optimized':jax.jit(jax.vmap(completions))}
if a.reference:
    spec=importlib.util.spec_from_file_location('fusion_reference',a.reference)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    fns['reference']=jax.jit(jax.vmap(old.completions))
outputs={name:fn(*args) for name,fn in fns.items()}
for out in outputs.values():jax.block_until_ready(out)
if 'reference' in outputs:
    for x,y in zip(outputs['optimized'],outputs['reference']):np.testing.assert_array_equal(np.asarray(x),np.asarray(y))
measurements={name:[] for name in fns}
for repeat in range(5):
    order=list(fns) if repeat%2==0 else list(reversed(fns))
    for name in order:
        t=time.perf_counter()
        for _ in range(a.calls):out=fns[name](*args)
        jax.block_until_ready(out)
        measurements[name].append((time.perf_counter()-t)*1000/a.calls)
for name,values in measurements.items():
    print(json.dumps(dict(name=name,ms_median=float(np.median(values)),samples_ms=values,b=a.b,n=a.n,
                          backend=jax.default_backend(),jax=jax.__version__)))
