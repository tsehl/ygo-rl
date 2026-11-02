"""Pilote Python (ctypes) d'ygopro-core, utilisé comme oracle de vérification.

Le duel avance jusqu'à une décision ; on l'expose sous forme normalisée (Decision : liste d'options
lisibles) et on répond par l'index de l'option choisie. L'état est lu par OCG_DuelQueryLocation.
Usage rapide : python driver.py  (joue une partie aléatoire entre deux decks de monstres normaux)
"""
import ctypes as C
import itertools
import os
import random
import re
import sqlite3
import struct
import sys
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "ygopro-core/bin/release", "libocgcore.dylib" if sys.platform == "darwin" else "libocgcore.so")
DATA = os.path.join(HERE, "../data")
SCRIPTS = [os.path.join(DATA, "CardScripts"), os.path.join(DATA, "CardScripts/official"), os.path.join(HERE, "test_scripts")]
CDB = os.path.join(DATA, "BabelCDB/cards.cdb")
# Cartes fictives, pour tester contre l'oracle une règle qu'aucune vraie carte simple ne met en jeu (scripts dans
# test_scripts/ ; la règle reste celle d'ygopro-core et des procédures Lua). Jamais dans l'entraînement.
# code : ((alias, setcode, type, atk, def ou flèches, level, race, attribute), nom)
TEST_CARDS = {
    # Extra Link (volet 5e) : Link sans effet, 2 monstres, Cyberse LIGHT
    990000001: ((0, 0, 0x4000001, 1500, 0x140, 2, 0x1000000, 0x10), "Test Link TL-TR"),  # haut-gauche, haut-droite
    990000002: ((0, 0, 0x4000001, 1400, 0x0A0, 2, 0x1000000, 0x10), "Test Link T-R"),    # haut, droite
    990000003: ((0, 0, 0x4000001, 1400, 0x088, 2, 0x1000000, 0x10), "Test Link L-T"),    # gauche, haut
}


def _constants():
    """Toutes les constantes #define numériques d'ocgapi_constants.h (expressions | évaluées)."""
    out = {}
    txt = open(os.path.join(HERE, "ygopro-core/ocgapi_constants.h")).read()
    for name, expr in re.findall(r"#define\s+(\w+)\s+(.+)", txt):
        expr = expr.split("//")[0].strip()
        try:
            out[name] = eval(re.sub(r"\b([A-Z_][A-Z0-9_]*)\b", lambda m: str(out.get(m.group(1), "None")), expr))
        except Exception:
            pass
    return out


K = _constants()
LOCS = {K["LOCATION_DECK"]: "deck", K["LOCATION_HAND"]: "hand", K["LOCATION_MZONE"]: "mzone",
        K["LOCATION_SZONE"]: "szone", K["LOCATION_GRAVE"]: "grave", K["LOCATION_REMOVED"]: "removed",
        K["LOCATION_EXTRA"]: "extra"}


# ---------------------------------------------------------------- structures C

class CardData(C.Structure):
    _fields_ = [("code", C.c_uint32), ("alias", C.c_uint32), ("setcodes", C.POINTER(C.c_uint16)),
                ("type", C.c_uint32), ("level", C.c_uint32), ("attribute", C.c_uint32), ("race", C.c_uint64),
                ("attack", C.c_int32), ("defense", C.c_int32), ("lscale", C.c_uint32), ("rscale", C.c_uint32),
                ("link_marker", C.c_uint32)]


class Player(C.Structure):
    _fields_ = [("startingLP", C.c_uint32), ("startingDrawCount", C.c_uint32), ("drawCountPerTurn", C.c_uint32)]


DataReader = C.CFUNCTYPE(None, C.c_void_p, C.c_uint32, C.POINTER(CardData))
DataReaderDone = C.CFUNCTYPE(None, C.c_void_p, C.POINTER(CardData))
ScriptReader = C.CFUNCTYPE(C.c_int, C.c_void_p, C.c_void_p, C.c_char_p)
LogHandler = C.CFUNCTYPE(None, C.c_void_p, C.c_char_p, C.c_int)


class DuelOptions(C.Structure):
    _fields_ = [("seed", C.c_uint64 * 4), ("flags", C.c_uint64), ("team1", Player), ("team2", Player),
                ("cardReader", DataReader), ("payload1", C.c_void_p), ("scriptReader", ScriptReader),
                ("payload2", C.c_void_p), ("logHandler", LogHandler), ("payload3", C.c_void_p),
                ("cardReaderDone", DataReaderDone), ("payload4", C.c_void_p), ("enableUnsafeLibraries", C.c_uint8)]


class NewCardInfo(C.Structure):
    _fields_ = [("team", C.c_uint8), ("duelist", C.c_uint8), ("code", C.c_uint32), ("con", C.c_uint8),
                ("loc", C.c_uint32), ("seq", C.c_uint32), ("pos", C.c_uint32)]


class QueryInfo(C.Structure):
    _fields_ = [("flags", C.c_uint32), ("con", C.c_uint8), ("loc", C.c_uint32), ("seq", C.c_uint32),
                ("overlay_seq", C.c_uint32)]


lib = C.CDLL(LIB)
lib.OCG_CreateDuel.argtypes = [C.POINTER(C.c_void_p), C.POINTER(DuelOptions)]
lib.OCG_DestroyDuel.argtypes = [C.c_void_p]
lib.OCG_DuelNewCard.argtypes = [C.c_void_p, C.POINTER(NewCardInfo)]
lib.OCG_StartDuel.argtypes = [C.c_void_p]
lib.OCG_DuelProcess.argtypes = [C.c_void_p]
lib.OCG_DuelGetMessage.argtypes = [C.c_void_p, C.POINTER(C.c_uint32)]
lib.OCG_DuelGetMessage.restype = C.c_void_p
lib.OCG_DuelSetResponse.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32]
lib.OCG_LoadScript.argtypes = [C.c_void_p, C.c_char_p, C.c_uint32, C.c_char_p]
lib.OCG_DuelQueryLocation.argtypes = [C.c_void_p, C.POINTER(C.c_uint32), C.POINTER(QueryInfo)]
lib.OCG_DuelQueryLocation.restype = C.c_void_p


# ---------------------------------------------------------------- base de cartes

class CardDB:
    def __init__(self, path=CDB):
        self.rows = {}
        con = sqlite3.connect(path)
        for r in con.execute("select id, alias, setcode, type, atk, def, level, race, attribute from datas"):
            self.rows[r[0]] = r
        self.names = dict(con.execute("select id, name from texts"))
        for code, (row, name) in TEST_CARDS.items():
            assert code not in self.rows
            self.rows[code], self.names[code] = (code,) + row, name

    def add(self, rows):
        """Ajoute des cartes (même format que TEST_CARDS) : cartes procédurales du jalon 6 (gen/tolua.py cdb_rows),
        dont les scripts sont dans un dossier ajouté en tête de SCRIPTS."""
        for code, (row, name) in rows.items():
            self.rows[code], self.names[code] = (code,) + tuple(int(x) for x in row), name

    def fill(self, code, out):
        r = self.rows.get(code)
        if r is None:
            return
        cid, alias, setcode, typ, atk, df, level, race, attr = r
        out.code, out.alias, out.type, out.attack, out.attribute, out.race = cid, alias, typ, atk, attr, race
        out.level = level & 0xFF
        out.lscale, out.rscale = (level >> 24) & 0xFF, (level >> 16) & 0xFF
        if typ & K["TYPE_LINK"]:
            out.defense, out.link_marker = 0, df
        else:
            out.defense, out.link_marker = df, 0
        codes = [(setcode >> (16 * i)) & 0xFFFF for i in range(4)]
        arr = (C.c_uint16 * 5)(*[c for c in codes if c], 0)
        self._keep = getattr(self, "_keep", []) + [arr]  # garder le tableau vivant pendant la lecture
        out.setcodes = C.cast(arr, C.POINTER(C.c_uint16))

    def normal_monsters(self):
        want = K["TYPE_MONSTER"] | K["TYPE_NORMAL"]
        return sorted(c for c, r in self.rows.items()
                      if r[3] == want and r[1] == 0 and 0 < (r[6] & 0xFF) <= 12)


# ---------------------------------------------------------------- lecture binaire

class Reader:
    def __init__(self, buf):
        self.b, self.p = buf, 0

    def take(self, fmt):
        v = struct.unpack_from("<" + fmt, self.b, self.p)
        self.p += struct.calcsize("<" + fmt)
        return v[0] if len(v) == 1 else v

    def loc_info(self):  # controler u8, location u8, sequence u32, position u32
        return self.take("BBII")


@dataclass
class Decision:
    msg: int
    player: int
    options: list            # descriptions lisibles, une par réponse possible
    responses: list          # bytes à envoyer pour chaque option
    raw: dict = field(default_factory=dict)


def _i32(*vals):
    return struct.pack("<" + "i" * len(vals), *vals)


def _cards_response(idxs):
    return struct.pack("<iI" + "I" * len(idxs), 0, len(idxs), *idxs)


# ---------------------------------------------------------------- duel

class Duel:
    def __init__(self, db, deck0, deck1, seed=0, flags=None, lp=8000, extra0=(), extra1=()):
        self.db = db
        self._cb = (DataReader(lambda p, code, out: db.fill(code, out.contents)),
                    DataReaderDone(lambda p, d: None),
                    ScriptReader(self._read_script),
                    LogHandler(lambda p, s, t: self.log.append(s.decode(errors="replace"))))
        self.log = []
        opt = DuelOptions()
        for i in range(4):
            opt.seed[i] = (seed * 0x9E3779B97F4A7C15 + i * 0xBF58476D1CE4E5B9 + 1) & (2 ** 64 - 1)
        opt.flags = K["DUEL_MODE_MR5"] | K["DUEL_PSEUDO_SHUFFLE"] if flags is None else flags
        opt.team1 = Player(lp, 5, 1)
        opt.team2 = Player(lp, 5, 1)
        opt.cardReader, opt.scriptReader, opt.logHandler, opt.cardReaderDone = (
            self._cb[0], self._cb[2], self._cb[3], self._cb[1])
        self.ptr = C.c_void_p()
        status = lib.OCG_CreateDuel(C.byref(self.ptr), C.byref(opt))
        if status != 0:
            raise RuntimeError(f"OCG_CreateDuel: {status}")
        for name in ("constant.lua", "utility.lua"):
            self._read_script(None, self.ptr, name.encode())
        for team, deck, extra in ((0, deck0, extra0), (1, deck1, extra1)):
            for code in deck:  # PSEUDO_SHUFFLE : le dernier ajouté est le dessus du deck
                lib.OCG_DuelNewCard(self.ptr, C.byref(NewCardInfo(team, 0, code, team, K["LOCATION_DECK"], 0,
                                                                  K["POS_FACEDOWN_DEFENSE"])))
            for code in extra:
                lib.OCG_DuelNewCard(self.ptr, C.byref(NewCardInfo(team, 0, code, team, K["LOCATION_EXTRA"], 0,
                                                                  K["POS_FACEDOWN_DEFENSE"])))
        self.lp = [lp, lp]
        self.turn, self.turn_player, self.phase = 0, 0, 0
        self.winner = None
        self.events = []
        lib.OCG_StartDuel(self.ptr)

    def _read_script(self, payload, duel, name):
        name = name.decode()
        for d in SCRIPTS:
            path = os.path.join(d, name)
            if os.path.exists(path):
                data = open(path, "rb").read()
                return lib.OCG_LoadScript(duel, data, len(data), name.encode())
        return 0

    def close(self):
        if self.ptr:
            lib.OCG_DestroyDuel(self.ptr)
            self.ptr = None

    # -------------------------------------------------- boucle

    def next_decision(self):
        """Avance jusqu'à la prochaine décision ; renvoie None si le duel est fini."""
        while True:
            status = lib.OCG_DuelProcess(self.ptr)
            n = C.c_uint32()
            p = lib.OCG_DuelGetMessage(self.ptr, C.byref(n))
            buf = C.string_at(p, n.value)
            dec = None
            off = 0
            while off < len(buf):
                (size,) = struct.unpack_from("<I", buf, off)
                msg = buf[off + 4:off + 4 + size]
                off += 4 + size
                d = self._handle(msg)
                if d is not None:
                    dec = d
            if dec is not None:
                return dec
            if self.winner is not None or status == K.get("OCG_DUEL_STATUS_END", 0):
                return None

    def respond(self, dec, i):
        r = dec.responses[i]
        lib.OCG_DuelSetResponse(self.ptr, r, len(r))

    # -------------------------------------------------- messages

    def _handle(self, msg):
        r = Reader(msg)
        t = r.take("B")
        if t == K["MSG_RETRY"]:
            raise RuntimeError("MSG_RETRY : réponse invalide envoyée à l'oracle")
        if t == K["MSG_WIN"]:
            self.winner = r.take("B")
            self.events.append(("win", self.winner))
        elif t == K["MSG_NEW_TURN"]:
            self.turn += 1
            self.turn_player = r.take("B")
        elif t == K["MSG_NEW_PHASE"]:
            self.phase = r.take("H")
        elif t == K["MSG_DAMAGE"] or t == K["MSG_PAY_LPCOST"]:
            pl, amt = r.take("BI")
            self.lp[pl] = max(0, self.lp[pl] - amt)
        elif t == K["MSG_RECOVER"]:
            pl, amt = r.take("BI")
            self.lp[pl] += amt
        elif t == K["MSG_LPUPDATE"]:
            pl, v = r.take("BI")
            self.lp[pl] = v
        elif t == K["MSG_SHUFFLE_HAND"]:
            self.events.append(("shuffle_hand", r.take("B")))
        elif t in SELECT_PARSERS:
            return SELECT_PARSERS[t](t, r)
        elif 10 <= t < 30 or t in (K["MSG_ANNOUNCE_RACE"], K["MSG_ANNOUNCE_ATTRIB"], K["MSG_ANNOUNCE_CARD"],
                                   K["MSG_ANNOUNCE_NUMBER"], K["MSG_ROCK_PAPER_SCISSORS"]):
            raise NotImplementedError(f"message de décision {t} pas encore géré par le pilote")
        return None

    # -------------------------------------------------- état

    QFLAGS = (K["QUERY_CODE"] | K["QUERY_POSITION"] | K["QUERY_ATTACK"] | K["QUERY_DEFENSE"]
              | K["QUERY_LEVEL"] | K["QUERY_OWNER"] | K["QUERY_OVERLAY_CARD"])

    def query(self, con, loc):
        q = QueryInfo(self.QFLAGS, con, loc, 0, 0)
        n = C.c_uint32()
        p = lib.OCG_DuelQueryLocation(self.ptr, C.byref(n), C.byref(q))
        buf = C.string_at(p, n.value)
        r = Reader(buf)
        r.take("I")
        out = []
        while r.p < len(buf):
            size = r.take("H")
            if size == 0:
                out.append(None)
                continue
            card = {}
            while True:
                flag = r.take("I")
                if flag == K["QUERY_END"]:
                    break
                raw = r.b[r.p:r.p + size - 4]
                r.p += size - 4
                name = {K["QUERY_CODE"]: "code", K["QUERY_POSITION"]: "pos", K["QUERY_ATTACK"]: "atk",
                        K["QUERY_DEFENSE"]: "def", K["QUERY_LEVEL"]: "level", K["QUERY_OWNER"]: "owner"}.get(flag)
                if name:
                    card[name] = int.from_bytes(raw, "little", signed=True)
                elif flag == K["QUERY_OVERLAY_CARD"]:  # matériaux Xyz attachés : nombre puis codes
                    k = int.from_bytes(raw[:4], "little")
                    card["ov"] = tuple(int.from_bytes(raw[4 + 4 * i:8 + 4 * i], "little") for i in range(k))
                size = r.take("H")
            out.append(card)
        return out

    def snapshot(self):
        """État comparable : par joueur et zone, liste ordonnée de cartes (code, position, atk, def)."""
        s = dict(lp=tuple(self.lp), turn=self.turn, turn_player=self.turn_player, phase=self.phase)
        for pl in (0, 1):
            for loc, name in LOCS.items():
                cards = self.query(pl, loc)
                s[(pl, name)] = [None if c is None else (c["code"], c["pos"], c["atk"], c["def"]) + ((c["ov"],) if c.get("ov") else ())
                                 for c in cards]
        return s


# ---------------------------------------------------------------- décodage des décisions

def _p_idle(t, r):
    pl = r.take("B")
    opts, resp = [], []
    for kind, idx, seqfmt in (("summon", 0, "I"), ("spsummon", 1, "I"), ("repos", 2, "B"), ("mset", 3, "I"),
                              ("sset", 4, "I")):
        for i in range(r.take("I")):
            code, con, loc = r.take("IBB")
            seq = r.take(seqfmt)
            opts.append((kind, code, con, LOCS.get(loc, loc), seq))
            resp.append(_i32(idx | (i << 16)))
    for i in range(r.take("I")):
        code, con, loc, seq, desc, mode = r.take("IBBIQB")
        opts.append(("activate", code, con, LOCS.get(loc, loc), seq, desc))
        resp.append(_i32(5 | (i << 16)))
    to_bp, to_ep, shuffle = r.take("BBB")
    if to_bp:
        opts.append(("to_bp",)), resp.append(_i32(6))
    if to_ep:
        opts.append(("to_ep",)), resp.append(_i32(7))
    return Decision(t, pl, opts, resp)


def _p_battle(t, r):
    pl = r.take("B")
    opts, resp = [], []
    for i in range(r.take("I")):
        code, con, loc, seq, desc, mode = r.take("IBBIQB")
        opts.append(("activate", code, con, LOCS.get(loc, loc), seq, desc))
        resp.append(_i32(0 | (i << 16)))
    for i in range(r.take("I")):
        code, con, loc, seq, direct = r.take("IBBBB")
        opts.append(("attack", code, con, LOCS.get(loc, loc), seq, direct))
        resp.append(_i32(1 | (i << 16)))
    to_m2, to_ep = r.take("BB")
    if to_m2:
        opts.append(("to_m2",)), resp.append(_i32(2))
    if to_ep:
        opts.append(("to_ep",)), resp.append(_i32(3))
    return Decision(t, pl, opts, resp)


def _p_yesno(t, r):
    pl = r.take("B")
    if t == K["MSG_SELECT_EFFECTYN"]:  # activer un effet déclenché optionnel : présenté comme une fenêtre de chaîne
        code = r.take("I")
        con, loc, seq, pos = r.loc_info()
        desc = r.take("Q")
        return Decision(t, pl, [("pass",), ("chain", code, con, LOCS.get(loc, loc), seq, desc)], [_i32(0), _i32(1)])
    r.take("Q")
    return Decision(t, pl, [("no",), ("yes",)], [_i32(0), _i32(1)])


def _p_option(t, r):
    pl, n = r.take("BB")
    descs = [r.take("Q") for _ in range(n)]
    return Decision(t, pl, [("option", d) for d in descs], [_i32(i) for i in range(n)])


def _card_list(r, n, tribute=False):
    out = []
    for _ in range(n):
        if tribute:
            code, con, loc, seq, rel = r.take("IBBIB")
            out.append((code, con, LOCS.get(loc, loc), seq, rel))
        else:
            code = r.take("I")
            con, loc, seq, pos = r.loc_info()
            if loc & K["LOCATION_OVERLAY"]:  # matériau Xyz : seq = case du Xyz, pos = rang du matériau
                out.append((code, con, "overlay", (seq, pos)))
            else:
                out.append((code, con, LOCS.get(loc, loc), seq))
    return out


def _p_select_card(t, r):
    """Sélection de min..max cartes : on énumère les sous-ensembles (petits en pratique)."""
    tribute = t == K["MSG_SELECT_TRIBUTE"]
    pl, cancel = r.take("BB")
    mn, mx, n = r.take("III")
    cards = _card_list(r, n, tribute)
    opts, resp = [], []
    for k in range(max(mn, 1 if not cancel else 0), mx + 1):
        for comb in itertools.combinations(range(n), k):
            if tribute and sum(cards[i][4] for i in comb) < mn:
                continue
            if tribute and k > mn and any(sum(cards[i][4] for i in comb if i != j) >= mn for j in comb):
                continue  # libérations superflues : l'oracle les accepte, on les ignore pour garder l'énumération petite
            opts.append(("cards", tuple(cards[i] for i in comb)))
            resp.append(_cards_response(comb))
    if cancel:
        opts.append(("cancel",)), resp.append(_i32(-1))
    return Decision(t, pl, opts, resp, dict(cards=cards, min=mn, max=mx))


def _p_select_sum(t, r):
    """Sélection par somme (rituel, synchro...) : une seule réponse, le sous-ensemble entier. Mode « supérieur » (max
    nul dans ygopro) : somme >= acc et sans carte superflue (somme - plus petite < acc) ; sinon somme exacte
    (select_sum_check1). On énumère les sous-ensembles valides des cartes proposées (les cartes imposées sont en plus)."""
    pl, greater = r.take("BB")
    acc, mn, mx = r.take("III")

    def lst():
        out = []
        for _ in range(r.take("I")):
            code = r.take("I")
            con, loc, seq, pos = r.loc_info()
            out.append(((code, con, LOCS.get(loc, loc), seq), r.take("I")))
        return out
    must, cards = lst(), lst()
    if mx < mn:
        mx = mn

    def ok(params):
        if greater:
            lo = [min(o & 0xFFFF, o >> 16) if (o >> 16) else o & 0xFFFF for o in params]
            hi = [max(o & 0xFFFF, o >> 16) for o in params]
            return sum(hi) >= acc and sum(lo) - min(lo) < acc

        def rec(i, a):
            if i == len(params):
                return a == 0
            o1, o2 = params[i] & 0xFFFF, params[i] >> 16
            return (a >= o1 and rec(i + 1, a - o1)) or (o2 > 0 and a >= o2 and rec(i + 1, a - o2))
        return rec(0, acc)
    opts, resp = [], []
    for k in range(0 if greater else mn, (len(cards) if greater else mx) + 1):
        for comb in itertools.combinations(range(len(cards)), k):
            if ok([m[1] for m in must] + [cards[i][1] for i in comb]):
                opts.append(("cards", tuple(cards[i][0] for i in comb)))
                resp.append(_cards_response(comb))
    return Decision(t, pl, opts, resp, dict(acc=acc, must=must))


def _p_unselect(t, r):
    """Sélection carte par carte : choisir une carte (ou en retirer une déjà choisie), ou terminer."""
    pl, finishable, cancel = r.take("BBB")
    mn, mx = r.take("II")
    sel = _card_list(r, r.take("I"))
    unsel = _card_list(r, r.take("I"))
    opts = [("pick", c) for c in sel] + [("unpick", c) for c in unsel]
    resp = [struct.pack("<ii", 1, i) for i in range(len(opts))]
    if finishable or cancel:  # même réponse -1 : « terminer » si le minimum est atteint, sinon « annuler »
        opts.append(("finish",) if finishable else ("cancel",)), resp.append(_i32(-1))
    return Decision(t, pl, opts, resp, dict(min=mn, max=mx))


def _p_chain(t, r):
    pl, spe, forced = r.take("BBB")
    r.take("II")
    opts, resp = [], []
    for i in range(r.take("I")):
        code = r.take("I")
        con, loc, seq, pos = r.loc_info()
        desc, mode = r.take("QB")
        opts.append(("chain", code, con, LOCS.get(loc, loc), seq, desc))
        resp.append(_i32(i))
    if not forced:
        opts.append(("pass",)), resp.append(_i32(-1))
    return Decision(t, pl, opts, resp)


def _p_place(t, r):
    pl, count, flag = r.take("BBI")
    if count != 1:
        raise NotImplementedError("SELECT_PLACE avec count > 1")
    opts, resp = [], []
    for side, base in ((pl, 0), (1 - pl, 16)):
        for loc, shift, nz in ((K["LOCATION_MZONE"], 0, 7), (K["LOCATION_SZONE"], 8, 8)):
            for s in range(nz):
                if not (flag >> (base + shift + s)) & 1:
                    opts.append(("place", side, LOCS[loc], s))
                    resp.append(struct.pack("<BBB", side, loc, s))
    return Decision(t, pl, opts, resp)


def _p_position(t, r):
    pl, code, positions = r.take("BIB")
    ps = [p for p in (1, 2, 4, 8) if positions & p]
    return Decision(t, pl, [("pos", p) for p in ps], [_i32(p) for p in ps])


SELECT_PARSERS = {
    K["MSG_SELECT_IDLECMD"]: _p_idle, K["MSG_SELECT_BATTLECMD"]: _p_battle,
    K["MSG_SELECT_YESNO"]: _p_yesno, K["MSG_SELECT_EFFECTYN"]: _p_yesno, K["MSG_SELECT_OPTION"]: _p_option,
    K["MSG_SELECT_CARD"]: _p_select_card, K["MSG_SELECT_TRIBUTE"]: _p_select_card,
    K["MSG_SELECT_CHAIN"]: _p_chain, K["MSG_SELECT_PLACE"]: _p_place, K["MSG_SELECT_DISFIELD"]: _p_place,
    K["MSG_SELECT_POSITION"]: _p_position, K["MSG_SELECT_UNSELECT_CARD"]: _p_unselect, K["MSG_SELECT_SUM"]: _p_select_sum,
}


# ---------------------------------------------------------------- démo

def random_duel(db, seed=0, max_decisions=3000, verbose=False):
    rng = random.Random(seed)
    pool = db.normal_monsters()
    decks = [[rng.choice(pool) for _ in range(40)] for _ in range(2)]
    duel = Duel(db, decks[0], decks[1], seed=seed)
    trace = []
    for _ in range(max_decisions):
        dec = duel.next_decision()
        if dec is None:
            break
        i = rng.randrange(len(dec.options))
        trace.append((dec.msg, dec.player, dec.options[i]))
        if verbose:
            print(duel.turn, hex(duel.phase), dec.player, dec.options[i], duel.lp)
        duel.respond(dec, i)
    result = dict(winner=duel.winner, turns=duel.turn, lp=tuple(duel.lp), decisions=len(trace))
    duel.close()
    return result, trace


if __name__ == "__main__":
    import collections
    import time
    db = CardDB()
    print(len(db.normal_monsters()), "monstres normaux")
    t0 = time.perf_counter()
    kinds = collections.Counter()
    for s in range(20):
        res, trace = random_duel(db, seed=s)
        kinds.update(o[0] for _, _, o in trace)
        print(s, res)
    print(f"{time.perf_counter() - t0:.2f}s", dict(kinds))
