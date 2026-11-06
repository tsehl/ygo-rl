"""Compare Hall + substitut à une recherche exhaustive indépendante, petits états."""
from pathlib import Path
import itertools
import sys
import unittest
import numpy as np
import jax
import jax.numpy as jnp
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'engine'))
from fusion import completions


def brute(cand, subs, field, eligible, rest, excluded, use_sub, need_field):
    slots = [j for j in range(4) if rest >> j & 1]
    for cards in itertools.permutations([i for i in range(cand.shape[1]) if i != excluded], len(slots)):
        wrong = [(j,i) for j,i in zip(slots,cards) if not cand[j,i]]
        if len(wrong) > int(use_sub): continue
        if any(not (eligible >> j & 1 and subs[i]) for j,i in wrong): continue
        if need_field and not any(field[i] for i in cards): continue
        return True
    return False


class MatchingTests(unittest.TestCase):
    def test_random_graphs_against_exhaustive_matching(self):
        rng = np.random.default_rng(1)
        evaluate = jax.jit(completions)
        for trial in range(20):
            cand = rng.random((4,6)) < .4
            subs, field = rng.random(6) < .4, rng.random(6) < .4
            eligible = int(rng.integers(16))
            excluded = np.arange(-1,6,dtype=np.int32)
            result = tuple(np.asarray(x) for x in evaluate(jnp.array(cand),jnp.array(subs),jnp.array(field),jnp.int32(eligible),jnp.array(excluded)))
            for rest in range(16):
                for column,x in enumerate(excluded):
                    for table,(use_sub,need_field) in enumerate(((False,False),(True,False),(False,True),(True,True))):
                        self.assertEqual(bool(result[table][rest,column]),brute(cand,subs,field,eligible,rest,x,use_sub,need_field),
                                         (trial,rest,x,use_sub,need_field))


    def test_partial_selections_against_exhaustive_assignments(self):
        import core, cards
        from collections import namedtuple
        codes = [100,101,102,999]
        effects = {102:[cards.cont(core.Z_HAND | core.Z_MZONE | core.Z_GRAVE,core.MC_FUS_SUB,0)]}
        recipe = cards.FusionRecipe([cards.MAT(code=100),cards.MAT(code=101),cards.MAT(code=101)],True)
        bank = core.make_bank(np.array(codes),[1000]*4,[1000]*4,[4]*4,
                              ctype=[17,17,33,65],**cards.assemble(codes,cards=effects,fusion={999:recipe},synchro={},xyz={},link={},ritual={}))
        State = namedtuple('MiniState','cid zone ctrl owner seq st fl_c')
        state = State(jnp.array([0,1,2,2,0,1,3,0]),jnp.array([core.Z_HAND]*6+[core.Z_EXTRA,core.Z_DECK]),
                      jnp.zeros(8,jnp.int32),jnp.zeros(8,jnp.int32),jnp.array([0,1,2,3,4,5,0,0]),
                      jnp.zeros(8,jnp.int32),jnp.zeros(8,jnp.int32))
        pick = jax.jit(lambda s: core.fus_picks(bank,s,jnp.int32(6)))
        normal = np.array([[1,0,0,0,1,0],[0,1,0,0,0,1],[0,1,0,0,0,1]],bool)
        states=[];expected_states=[]
        for count in range(3):
            for chosen in itertools.combinations(range(6),count):
                st=np.zeros(8,np.int32);st[list(chosen)]=core.ST_SEL
                got=np.asarray(pick(state._replace(st=jnp.array(st))))
                expected=np.zeros(8,bool)
                for candidate in set(range(6))-set(chosen):
                    required=set(chosen)|{candidate}
                    for assignment in itertools.permutations(range(6),3):
                        if not required.issubset(assignment): continue
                        wrong=[i for j,i in enumerate(assignment) if not normal[j,i]]
                        if len(wrong)<=1 and all(i in (2,3) for i in wrong):
                            expected[candidate]=True;break
                np.testing.assert_array_equal(got,expected,err_msg=str(chosen))
                states.append(state._replace(st=jnp.array(st)));expected_states.append(expected)
        batch=jax.tree.map(lambda *xs:jnp.stack(xs),*states)
        batched=jax.jit(jax.vmap(lambda s:core.fus_picks(bank,s,jnp.int32(6))))(batch)
        np.testing.assert_array_equal(np.asarray(batched),np.stack(expected_states))


    def test_full_field_requires_a_material_that_frees_a_zone(self):
        import core,cards
        from collections import namedtuple
        codes=[100,101,102,103,999]
        effects={102:[cards.cont(core.Z_HAND|core.Z_MZONE,core.MC_FUS_SUB,0)]}
        recipe=cards.FusionRecipe([cards.MAT(code=100),cards.MAT(code=101)],True)
        bank=core.make_bank(np.array(codes),[1000]*5,[1000]*5,[4]*5,ctype=[17,17,33,17,65],
                            **cards.assemble(codes,cards=effects,fusion={999:recipe},ritual={},synchro={},xyz={},link={}))
        State=namedtuple('FullState','cid zone ctrl owner seq st fl_c')
        s=State(jnp.array([0,1,2,0,1,2,4]+[3]*6),jnp.array([core.Z_HAND]*6+[core.Z_EXTRA]+[core.Z_MZONE]*6),
                jnp.zeros(13,jnp.int32),jnp.zeros(13,jnp.int32),jnp.array([0,1,2,3,4,5,0,0,1,2,3,4,5]),
                jnp.zeros(13,jnp.int32),jnp.zeros(13,jnp.int32))
        available=jax.jit(lambda state:core.fusion_ok(bank,state)[0][6])
        self.assertFalse(bool(available(s)))
        self.assertTrue(bool(available(s._replace(cid=s.cid.at[7].set(1)))))


if __name__ == '__main__': unittest.main()
