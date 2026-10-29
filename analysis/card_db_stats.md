# Yu-Gi-Oh card database statistics

Source: `/Users/thomas/Documents/collab/ygo/data/BabelCDB/cards.cdb` (table `datas`, texts from `texts`).

## Filter

| Metric | Value |
|---|---|
| Rows in `datas` | 14759 |
| Excluded (alias != 0, alternate artworks) | 438 |
| Cards after filter | 14321 |

## Top-level kind

| Kind | Count | Share |
|---|---|---|
| Monster | 9224 | 64.4% |
| Spell | 2838 | 19.8% |
| Trap | 2059 | 14.4% |
| other | 200 | 1.4% |

Tokens (TYPE_TOKEN) are counted in `other`: 200. Multi-kind anomalies: 0.

## Monsters

| Deck | Count | Share |
|---|---|---|
| Main | 7067 | 76.6% |
| Extra (Fusion/Synchro/Xyz/Link) | 2157 | 23.4% |

### Subtype flags (every TYPE_ bit; zero rows omitted)

| Flag | Count | Share of monsters |
|---|---|---|
| NORMAL | 722 | 7.8% |
| EFFECT | 8417 | 91.3% |
| FUSION | 568 | 6.2% |
| RITUAL | 148 | 1.6% |
| SPIRIT | 42 | 0.5% |
| UNION | 40 | 0.4% |
| GEMINI | 45 | 0.5% |
| TUNER | 602 | 6.5% |
| SYNCHRO | 533 | 5.8% |
| FLIP | 212 | 2.3% |
| TOON | 18 | 0.2% |
| XYZ | 593 | 6.4% |
| PENDULUM | 389 | 4.2% |
| SPSUMMON | 344 | 3.7% |
| LINK | 463 | 5.0% |

### Level / rank (main deck, level & 0xFF)

| Level/Rank | Count |
|---|---|
| 1 | 675 |
| 2 | 697 |
| 3 | 1179 |
| 4 | 2336 |
| 5 | 510 |
| 6 | 479 |
| 7 | 369 |
| 8 | 522 |
| 9 | 81 |
| 10 | 172 |
| 11 | 21 |
| 12 | 26 |

### Link rating (Link monsters)

| Link rating | Count |
|---|---|
| 1 | 62 |
| 2 | 222 |
| 3 | 110 |
| 4 | 54 |
| 5 | 13 |
| 6 | 2 |

### Pendulum left scale

| Scale | Count |
|---|---|
| 0 | 28 |
| 1 | 81 |
| 2 | 39 |
| 3 | 39 |
| 4 | 37 |
| 5 | 38 |
| 6 | 16 |
| 7 | 28 |
| 8 | 48 |
| 9 | 13 |
| 10 | 13 |
| 11 | 4 |
| 12 | 3 |
| 13 | 2 |

### ATK buckets (monsters)

| ATK | Count |
|---|---|
| 0-499 | 1481 |
| 500-999 | 1100 |
| 1000-1499 | 1525 |
| 1500-1999 | 1946 |
| 2000-2499 | 1369 |
| 2500-2999 | 1065 |
| 3000+ | 656 |
| ? | 82 |
| negative_other | 0 |

### Race (monsters; a card may have one race)

| Race | Count |
|---|---|
| AQUA | 289 |
| BEAST | 430 |
| BEASTWARRIOR | 264 |
| CREATORGOD | 1 |
| CYBERSE | 293 |
| DINOSAUR | 150 |
| DIVINE | 5 |
| DRAGON | 806 |
| FAIRY | 577 |
| FIEND | 903 |
| FISH | 147 |
| ILLUSION | 55 |
| INSECT | 267 |
| MACHINE | 1055 |
| PLANT | 257 |
| PSYCHIC | 210 |
| PYRO | 161 |
| REPTILE | 193 |
| ROCK | 269 |
| SEASERPENT | 98 |
| SPELLCASTER | 794 |
| THUNDER | 153 |
| WARRIOR | 1149 |
| WINGEDBEAST | 332 |
| WYRM | 98 |
| ZOMBIE | 268 |

### Attribute (monsters)

| Attribute | Count |
|---|---|
| DARK | 2619 |
| DIVINE | 6 |
| EARTH | 2073 |
| FIRE | 815 |
| LIGHT | 1928 |
| WATER | 929 |
| WIND | 854 |

## Spells

| Subtype | Count | Share of spells |
|---|---|---|
| Normal | 1069 | 37.7% |
| QUICKPLAY | 565 | 19.9% |
| CONTINUOUS | 511 | 18.0% |
| EQUIP | 282 | 9.9% |
| FIELD | 328 | 11.6% |
| RITUAL | 83 | 2.9% |

## Traps

| Subtype | Count | Share of traps |
|---|---|---|
| Normal | 1320 | 64.1% |
| CONTINUOUS | 563 | 27.3% |
| COUNTER | 176 | 8.5% |

## Setcodes (archetypes)

| Distinct setcodes on one card | Cards |
|---|---|
| 0 | 6351 |
| 1 | 7317 |
| 2 | 598 |
| 3 | 48 |
| 4 | 7 |

Max distinct setcodes on one card: 4. Distinct archetype codes (nonzero 16-bit chunks): 616.

## Effect text length (characters)

| Group | Count | Min | Median | p90 | Max |
|---|---|---|---|---|---|
| Effect monsters | 8417 | 18 | 364 | 543 | 1112 |
| Spells | 2838 | 13 | 291.5 | 499 | 713 |
| Traps | 2059 | 12 | 245 | 475 | 695 |

p90 uses nearest-rank.

## Data quality notes

- Composite constant lines skipped: 3
- Unmapped type bits seen: none
- Unknown race bits seen: none
- Unknown attribute bits seen: none
- Rows with no description text: none
