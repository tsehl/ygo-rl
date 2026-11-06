"""Rejoue intégralement les cinq parties auparavant non concluantes, decks originaux."""
import json
from pathlib import Path
from unittest.mock import patch
import jax
from oracle_bank import build
import compare,core,build_inspector

cases=json.loads((Path(__file__).parent/'fixtures/ambiguous_effects.json').read_text())
codes=[c for g in cases for decks in (g['decks'],g['extras']) for deck in decks for c in deck]
db,bank,cmap=build(codes)
for g in cases:
    with patch.object(build_inspector,'event',wraps=build_inspector.event) as query:
        r=compare.play(db,bank,cmap,g['seed'],jax.jit(core.step),jax.jit(core.legal),mode='tr',given=(g['decks'],g['extras']))
        assert r['ok'],(g['seed'],r.get('err'))
        assert query.call_count>0 and r['decisions']>g['previous_decisions'],(g['seed'],r['decisions'],query.call_count)
        print(f"graine {g['seed']} : identique, {r['decisions']} décisions, {query.call_count} identités lues côté oracle",flush=True)
