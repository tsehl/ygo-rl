"""Benchmark sur les vraies cartes (jamais vues à l'entraînement) : banque et decks de référence.

La banque se construit depuis la base de l'oracle (`python train/realcards.py` → train/realbank.npz), puis se recharge
sans l'oracle (Colab). Elle est complétée aux formes des banques procédurales (C = 480 cartes, tprog à 2 × PT) : mêmes
formes, donc pas de nouvelle compilation des fonctions du rollout.

Quatre decks de 40 cartes + Extra Deck de 15, construits à la main avec les cartes que le moteur sait jouer :
contrôle (magies / pièges génériques), synchro / Xyz / Link, rituel / fusion, pendule. Les cartes fictives de test
(Test Link) n'y sont jamais."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", d) for d in ("engine", "gen", "oracle")]
PATH = os.path.join(HERE, "realbank.npz")
C_PAD = 480  # cartes d'une banque procédurale (ppo.py : --main 400 --extra 80)
# champs de core.Bank dont l'axe 1 est l'effet (K_NONE = 0 : un effet nul est vide)
E_FIELDS = ("ekind", "prog", "tprog", "filt", "etrig", "eopt", "erng", "emod", "edmg", "einline", "erefund")

LINKS = [("Knightmare Phoenix", 1), ("Knightmare Cerberus", 1), ("LANphorhynchus", 1),
         ("Gaia Saber, the Lightning Shadow", 1), ("Traffic Ghost", 1), ("Berserker of the Tenyi", 1)]
DECKS = {
    "controle": dict(main=[
        ("Sangan", 1), ("Witch of the Black Forest", 1), ("Mystic Tomato", 2), ("Exiled Force", 1), ("D.D. Crow", 1),
        ("Effect Veiler", 1), ("Command Knight", 2), ("Cannon Soldier", 1), ("Lord of D.", 1),
        ("Obnoxious Celtic Guard", 2), ("Man-Eater Bug", 1), ("Dark Magician Girl", 1), ("Kazejin", 1),
        ("Arsenal Bug", 2), ("Winged Rhynos", 1), ("Big-Tusked Mammoth", 1),
        ("Pot of Greed", 1), ("Graceful Charity", 1), ("Raigeki", 1), ("Dark Hole", 1), ("Monster Reborn", 1),
        ("Harpie's Feather Duster", 1), ("Mystical Space Typhoon", 2), ("Upstart Goblin", 1), ("Shrink", 1),
        ("Forbidden Chalice", 1),
        ("Mirror Force", 1), ("Torrential Tribute", 1), ("Trap Hole", 2), ("Sakuretsu Armor", 2),
        ("Magic Jammer", 1), ("Seven Tools of the Bandit", 1), ("Jar of Greed", 1)],
        extra=LINKS + [("Gem-Knight Pearl", 3), ("Gagagigo the Risen", 3), ("Number 17: Leviathan Dragon", 3)]),
    "synchro": dict(main=[
        ("Tune Warrior", 3), ("Genex Controller", 2), ("Ally Mind", 1), ("Water Spirit", 1), ("Flamvell Guard", 2),
        ("Command Knight", 2), ("Obnoxious Celtic Guard", 2), ("Lord of D.", 1), ("Arsenal Bug", 2),
        ("Mystic Tomato", 2), ("Big-Tusked Mammoth", 1), ("Ally of Justice Quarantine", 2),
        ("Elemental HERO Wildheart", 2), ("Ghost Ogre & Snow Rabbit", 1),
        ("Pot of Greed", 1), ("Monster Reborn", 1), ("E - Emergency Call", 2), ("Rush Recklessly", 2),
        ("Banner of Courage", 1), ("Mystical Space Typhoon", 2), ("Upstart Goblin", 2),
        ("Mirror Force", 1), ("Trap Hole", 2), ("Reverse Glasses", 1), ("Just Desserts", 1)],
        extra=[("Gaia Knight, the Force of Earth", 2), ("Scrap Archfiend", 2), ("Naturia Leodrake", 1),
               ("Gem-Knight Pearl", 2), ("Gagagigo the Risen", 1), ("Number 17: Leviathan Dragon", 1)] + LINKS),
    "rituel_fusion": dict(main=[
        ("Black Luster Soldier", 2), ("Black Luster Ritual", 2), ("Demise, King of Armageddon", 2),
        ("End of the World", 2), ("Polymerization", 3), ("Flame Manipulator", 2), ("Masaki the Legendary Swordsman", 2),
        ("Gaia The Fierce Knight", 2), ("Curse of Dragon", 2), ("Blue-Eyes White Dragon", 3),
        ("Sangan", 1), ("Witch of the Black Forest", 2), ("Mystic Tomato", 2), ("Man-Eater Bug", 1),
        ("Pot of Greed", 1), ("Graceful Charity", 1), ("Monster Reborn", 1), ("Dark Hole", 1), ("Upstart Goblin", 2),
        ("Gravity Bind", 1), ("Castle Walls", 1), ("Mirror Force", 1), ("Dian Keto the Cure Master", 1),
        ("Mystical Space Typhoon", 1), ("Trap Hole", 1)],
        extra=[("Flame Swordsman", 3), ("Gaia the Dragon Champion", 3), ("Blue-Eyes Ultimate Dragon", 3)] + LINKS),
    "pendule": dict(main=[
        ("Flash Knight", 3), ("Mandragon", 2), ("Fire Opal Head", 3), ("Gallugallud the Endless Enigma", 2),
        ("Dragong", 3), ("Lord of D.", 1), ("Mystic Tomato", 1), ("Bladefly", 1), ("Deepsea Macrotrema", 1),
        ("The Little Swordsman of Aile", 2), ("Command Knight", 2), ("Winged Rhynos", 2), ("Exiled Force", 1),
        ("Hinotama", 2), ("Banner of Courage", 1), ("Rush Recklessly", 2), ("Shrink", 1), ("Monster Reborn", 1),
        ("Pot of Greed", 1), ("Fissure", 1), ("Back-Up Rider", 1),
        ("Skill Drain", 1), ("Sakuretsu Armor", 2), ("Trap Hole", 1), ("Just Desserts", 1), ("Castle Walls", 1)],
        extra=LINKS + [("Gem-Knight Pearl", 3), ("Gagagigo the Risen", 3), ("Number 17: Leviathan Dragon", 3)]),
}


def _expand(lst):
    return [name for name, k in lst for _ in range(k)]


def build(path=PATH):
    """Banque des cartes des decks (base de l'oracle), complétée aux formes procédurales, et decks → npz."""
    import driver
    import compare
    import cardgen
    db = driver.CardDB()
    by_name = {}
    for c, nm in db.names.items():
        if c in db.rows and db.rows[c][1] == 0:   # pas les versions alternatives (alias)
            by_name.setdefault(nm, c)
    missing = sorted({n for d in DECKS.values() for n, _ in d["main"] + d["extra"]} - set(by_name))
    assert not missing, f"cartes introuvables : {missing}"
    for nm, d in DECKS.items():
        assert len(_expand(d["main"])) == 40, (nm, len(_expand(d["main"])))
        assert len(_expand(d["extra"])) == 15, (nm, len(_expand(d["extra"])))
    codes = sorted({by_name[n] for d in DECKS.values() for n in _expand(d["main"]) + _expand(d["extra"])})
    bank, cmap = compare.make_bank(db, codes)
    out = {}
    for k in bank._fields:
        x = np.asarray(getattr(bank, k))
        if k == "tprog":
            x = np.pad(x, ((0, 0), (0, 0), (0, 2 * cardgen.PT - x.shape[2]), (0, 0)))
        if k in ("einline", "erefund") and x.shape[1] == 0:   # mêmes formes que cardgen.to_bank
            x = np.zeros((x.shape[0], np.asarray(bank.ekind).shape[1]), x.dtype)
        assert x.shape[0] == len(codes), (k, x.shape)
        # complétée par des copies de la première carte (jamais dans un deck)
        out["bank_" + k] = np.concatenate([x, np.repeat(x[:1], C_PAD - len(codes), 0)])
    names = sorted(DECKS)
    out["deck_names"] = np.array(names)
    out["card_names"] = np.array([db.names[c] for c in codes] + [""] * (C_PAD - len(codes)))
    out["main"] = np.array([[cmap[by_name[n]] for n in _expand(DECKS[d]["main"])] for d in names], np.int32)
    out["extra"] = np.array([[cmap[by_name[n]] for n in _expand(DECKS[d]["extra"])] for d in names], np.int32)
    np.savez_compressed(path, **out)
    print(f"{path} : {len(codes)} cartes (complétées à {C_PAD}), decks {names}")


def load(path=PATH):
    """→ (core.Bank, noms des decks, main [D, 40], extra [D, 15]) ; noms des cartes : card_names(path)."""
    import jax.numpy as jnp
    import core
    z = np.load(path)
    c = z["bank_code"].shape[0]
    # champs optionnels ajoutés après la création du fichier : largeur 0 (chemin éliminé au traçage)
    get = lambda k: z["bank_" + k] if "bank_" + k in z else np.zeros((c, 0), np.int32)

    def fit(k, x):  # fichier créé avec moins d'effets par carte (core.E) : effets vides ajoutés
        if k in E_FIELDS and 0 < x.shape[1] < core.E:
            x = np.pad(x, [(0, 0), (0, core.E - x.shape[1])] + [(0, 0)] * (x.ndim - 2))
        return x
    bank = core.Bank(**{k: jnp.asarray(fit(k, get(k))) for k in core.Bank._fields})
    return bank, list(z["deck_names"]), z["main"], z["extra"]


def card_names(path=PATH):
    """Nom de chaque carte de la banque ("" pour le remplissage)."""
    return list(np.load(path)["card_names"])


if __name__ == "__main__":
    build()
