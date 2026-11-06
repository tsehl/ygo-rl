"""Scénarios réels de procédures de main, suivis contre l'oracle jusqu'à la fin."""
import jax
from oracle_bank import build
import core,compare

CYBER,JUNK,HACK,PHOTON,SCHWARZ,DM,ELF,REBORN=70095154,56410040,97466712,65367484,6930746,46986414,44287299,83764718
MOON,MYTHIC=336601,92841002
db,bank,cmap=build([CYBER,JUNK,HACK,PHOTON,SCHWARZ,DM,ELF,REBORN,MOON,MYTHIC])
step,legal=jax.jit(core.step),jax.jit(core.legal)
cases=[('Cyber Dragon',CYBER,[CYBER]*40,[ELF]*40,True),
       ('Junk Forward',JUNK,[JUNK]*40,[ELF]*40,True),
       ('Hack Worm',HACK,[HACK]*40,[ELF]*40,True),
       ('Photon Thrasher',PHOTON,[PHOTON,REBORN]*20,[ELF]*40,True),
       ('Schwarzschild, ennemi faible',SCHWARZ,[SCHWARZ]*40,[ELF]*40,False),
       ('Schwarzschild, ennemi fort',SCHWARZ,[SCHWARZ]*40,[JUNK,DM]*20,True),
       ('Mythic Water, limite par nom',MYTHIC,[JUNK,MYTHIC]*20,[ELF]*40,True),
       ('Moon Gardna, clones regroupés',MOON,[MOON,ELF]*20,[CYBER,JUNK]*20,True)]
for name,card,deck0,deck1,expected in cases:
    uses=0
    for seed in (300,301):
        def chooser(dec,indices):
            for kind in ('spsummon','summon','activate','attack','to_bp'):
                opts=[i for i in indices if dec.options[i][0]==kind]
                if opts:return opts[0]
            return indices[0]
        result=compare.play(db,bank,cmap,seed,step,legal,mode='tr',given=([deck0,deck1],[[],[]]),chooser=chooser)
        assert result['ok'],(name,seed,result.get('err'))
        uses+=sum(t['option'][0]=='spsummon' and t['option'][1]==card and t['option'][3]=='hand' for t in result['trace'])
        # Une invocation inhérente ne doit pas être comptabilisée comme activation en chaîne.
        if card not in (MOON,PHOTON):
            assert not any(e['code']==card and e['stage']=='activated' for e in result['coverage'])
    assert (uses>0)==expected,(name,uses)
    print(f'{name} : 2/2 identiques, {uses} procédures de main exercées',flush=True)
