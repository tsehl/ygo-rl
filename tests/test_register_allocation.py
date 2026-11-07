"""Groupes virtuels : liveness, dépendances, mutations et frontières de chaîne."""
import sys
import random
import unittest
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'translate')]
import translate as T
C, core = T.C, T.core


class RegisterAllocationTests(unittest.TestCase):
    def lower(self, cost=(), target=(), resolve=()):
        return T.Translator.allocate(C.act(cost=cost,target=target,resolve=resolve))

    def test_many_temporary_groups_use_one_register(self):
        program=[]
        for r in range(7):
            program.extend([T.Translator.vfilter(r,C.F(me=core.Z_HAND,code=r%4+1)),C.REQ(r,1),C.DRAW(0,1)])
        # Deux phases pour respecter les dimensions des programmes.
        effect=self.lower(cost=program[:6],target=program[6:12],resolve=program[12:])
        self.assertEqual({ins[1] for pr in effect[4:7] for ins in pr if ins[0] in (core.OP_FILTER,core.OP_REQ)},{0})

    def test_overlapping_groups_are_preserved(self):
        cost=[T.Translator.vfilter(r,C.F(me=core.Z_HAND,code=r+1)) for r in range(4)]
        resolve=[C.MOVE(r) for r in range(4)]
        e=self.lower(cost=cost,resolve=resolve)
        self.assertEqual(len({ins[1] for ins in e[4]}),4)
        self.assertEqual([ins[1] for ins in e[4]],[ins[1] for ins in e[6]])
        with self.assertRaisesRegex(T.Unsupported,'registres vivants'):
            self.lower(cost=cost+[T.Translator.vfilter(4,C.F(me=core.Z_DECK))],
                       resolve=resolve+[C.MOVE(4)])

    def test_high_virtual_index_and_inplace_filter(self):
        e=self.lower(resolve=[C.SELF(25),T.Translator.vfilter(25,C.F(me=core.Z_MZONE),within=25,tgt=True),C.MOVE(25)])
        self.assertEqual(e[6][1][3],9)  # intersection physique r0 et ciblage
        self.assertEqual(e[6][-1][1],0)

    def test_repeated_query_shared_until_mutation_or_chain_response(self):
        f=C.F(me=core.Z_HAND)
        query=lambda r:T.Translator.vfilter(r,f)
        e=self.lower(cost=[query(0),C.REQ(0,1),query(1),C.REQ(1,1)],
                     target=[query(2),C.CHOOSE(3,2,1),C.MOVE(3),query(4),C.REQ(4,1)],
                     resolve=[query(5),C.MOVE(5)])
        self.assertEqual([sum(ins[0]==core.OP_FILTER for ins in pr) for pr in e[4:7]],[1,1,1])

    def test_unused_query_is_removed_and_count_operand_remapped(self):
        e=self.lower(resolve=[C.SELF(17),T.Translator.vfilter(23,C.F(me=core.Z_HAND)),C.DAMAGE(0,100,per=23)])
        self.assertEqual(len(e[6]),2)
        self.assertEqual(e[6][-1],C.DAMAGE(0,100,per=0))

    def test_destructive_result_remains_available(self):
        e=self.lower(resolve=[T.Translator.vfilter(12,C.F(me=core.Z_MZONE)),C.DESTROY(12),C.DAMAGE(0,100,per=12)])
        self.assertEqual(e[6][1],C.DESTROY(0))
        self.assertEqual(e[6][2],C.DAMAGE(0,100,per=0))

    def test_mutation_reads_the_original_slot_even_when_a_lower_slot_is_free(self):
        e=self.lower(resolve=[T.Translator.vfilter(10,C.F(me=core.Z_HAND)),
                              T.Translator.vfilter(11,C.F(me=core.Z_MZONE)),
                              C.REQ(10,1),C.DESTROY(11),C.DAMAGE(0,100,per=11)])
        self.assertEqual(e[6][1][1],1)
        self.assertEqual(e[6][3],C.DESTROY(1))
        self.assertEqual(e[6][4],C.DAMAGE(0,100,per=1))

    def test_independent_queries_do_not_share_a_mutated_group(self):
        f=C.F(me=core.Z_MZONE)
        e=self.lower(resolve=[T.Translator.vfilter(10,f),T.Translator.vfilter(11,f),
                              C.DESTROY(11),C.DAMAGE(0,100,per=10)])
        self.assertEqual(len(e[6]),4)
        self.assertNotEqual(e[6][0][1],e[6][1][1])

    def test_optimized_programs_preserve_group_snapshots_and_observable_results(self):
        # Petit interpréteur de référence indépendant de l'allocateur ; les groupes sont
        # des ensembles persistants, la destruction retire seulement les cartes fragiles.
        def execute(programs, initial, vulnerable):
            zones=list(initial); regs={}; damage=0; events=[]
            for phase,program in enumerate(programs):
                if phase==core.PG_RESOLVE:
                    zones[0]=core.Z_GRAVE  # une réponse peut déplacer une carte
                for op,a,b,c in program:
                    if op==core.OP_FILTER:
                        within,target=c if isinstance(c,tuple) else ((c&7)-1 if c&7 else None,bool(c&8))
                        group={i for i,z in enumerate(zones) if z&b[0]}
                        if within is not None:group&=regs[within]
                        regs[a]=group
                    elif op==core.OP_SELF:regs[a]={0}
                    elif op==core.OP_CHOOSE:
                        regs[a]=set(sorted(regs[b])[:(c>>4)&15])
                        events.append(('choose',tuple(sorted(regs[a]))))
                    elif op==core.OP_MOVE:
                        group=regs[a] & vulnerable if c==1 else regs[a]
                        if c==1:regs[a]=set(group)
                        for i in group:zones[i]=b
                        events.append(('move',tuple(sorted(group)),b))
                    elif op==core.OP_DAMAGE:damage+=b*(len(regs[c-1]) if c else 1)
                    elif op==core.OP_REQ:
                        valid=len(regs[a])<=b if c else len(regs[a])>=b
                        if not valid:break
                    else:raise AssertionError(op)
            return zones,damage,events
        rng=random.Random(8713)
        for trial in range(200):
            initial=[rng.choice((core.Z_HAND,core.Z_MZONE,core.Z_GRAVE)) for _ in range(12)]
            fragile={i for i in range(12) if rng.randrange(2)}
            cost=[T.Translator.vfilter(10,C.F(me=core.Z_HAND)),C.REQ(10,0)]
            target=[T.Translator.vfilter(11,C.F(me=core.Z_HAND))]
            resolution=[T.Translator.vfilter(20,C.F(me=core.Z_MZONE)),
                        T.Translator.vfilter(21,C.F(me=core.Z_MZONE)),C.DESTROY(21),
                        C.DAMAGE(0,100,per=20),C.DAMAGE(0,100,per=21),
                        T.Translator.vfilter(22,C.F(me=core.Z_HAND)),C.CHOOSE(23,22,1),C.MOVE(23),
                        T.Translator.vfilter(24,C.F(me=core.Z_GRAVE),within=20),C.DAMAGE(0,100,per=24)]
            optimized=self.lower(cost=cost,target=target,resolve=resolution)
            self.assertEqual(execute((cost,target,resolution),initial,fragile),
                             execute(optimized[4:7],initial,fragile),trial)

if __name__=='__main__':unittest.main()
