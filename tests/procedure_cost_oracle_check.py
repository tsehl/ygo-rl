"""Paiement des procédures sans chaîne, sur des cartes officielles."""
import jax
from oracle_bank import build
import core,compare
CRUISER,THRESHOLD,MACHINE,CYBERSE,ELF=1586457,31944175,70095154,36211150,44287299
db,bank,cmap=build([CRUISER,THRESHOLD,MACHINE,CYBERSE,ELF])
step,legal=jax.jit(core.step),jax.jit(core.legal)
for card,material in ((CRUISER,MACHINE),(THRESHOLD,CYBERSE)):
    count=0
    for seed in (400,401,402,403):
        def chooser(dec,indices):
            for kind in ('spsummon','summon','activate','attack','to_bp'):
                opts=[i for i in indices if dec.options[i][0]==kind]
                if opts:return opts[0]
            return indices[0]
        result=compare.play(db,bank,cmap,seed,step,legal,mode='tr',given=([[card,material]*20,[ELF]*40],[[],[]]),chooser=chooser)
        assert result['ok'],(card,seed,result.get('err'))
        count+=sum(t['option'][0]=='spsummon' and t['option'][1]==card and t['option'][3]=='hand' for t in result['trace'])
        assert not any(e['code']==card and e['stage']=='activated' for e in result['coverage'])
    assert count>0,(card,count)
    print(f'{card}: 4/4 identiques, {count} procédures avec paiement',flush=True)
