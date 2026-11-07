"""Politique ciblée : un défenseur, attaque seulement après l'enregistrement de la protection."""
import core,compare


def chooser_for(card, protected_attacks):
    def chooser(dec,indices):
        state=compare.LAST_STATE
        own=(state.zone==core.Z_MZONE)&(state.ctrl==0)
        registered=bool((state.tm_m & (state.tm_code==core.MC_INDES_BATTLE)[:,None]).any())
        if dec.player==0:
            if int(state.tp)==1:
                for i in indices:
                    if dec.options[i][0]=='chain':return i
            for i in indices:
                if dec.options[i][0]=='sset':return i
            for i in indices:
                option=dec.options[i]
                if option[0]=='repos' and option[1]==card:
                    slot=compare.slot(state,option[2],option[3],option[4])
                    if int(state.pos[slot])==core.P_FU_ATK:return i
            if not bool(own.any()):
                for i in indices:
                    if dec.options[i][0]=='summon':return i
        else:
            for kind in (('summon','attack','to_bp') if registered else ('summon',)):
                for i in indices:
                    if dec.options[i][0]==kind:
                        if kind=='attack' and int(own.sum())==1 and bool((own&(state.ib_c<core.BIG)).any()):
                            protected_attacks[0]+=1
                        return i
        for i in indices:
            if dec.options[i][0]=='to_ep':return i
        return indices[0]
    return chooser
