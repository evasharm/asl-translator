## Iteration 10: caption cleanup, LLM (anthropic: claude-opus-5-5)

| Version | eva CER | isha CER | All CER | Word acc | Over-correction | Made worse | Stream CER |
|---|---|---|---|---|---|---|---|
| raw caption | 49.5% | 40.2% | 44.8% | 10% | - | - | 36.7% |
| dictionary fix | 52.7% | 37.7% | 45.2% | 30% | 0% | 30% | 38.8% |
| LLM (anthropic: claude-opus-5-5) | 43.8% | 40.2% | 42.0% | 25% | 0% | 30% | 28.6% |

Over-correction is measured on the 2 of 20 words that were already right before the fix.
LLM latency: 4.90 s per call on average (median 3.61 s, 22 calls).

| Person | Word | Raw | Dictionary | LLM |
|---|---|---|---|---|
| eva | HELLO | `HECOCO` | `COCO` | `ROCOCO` |
| eva | BOOK | `BOK` | `BOOK` | `BOOK` |
| eva | PENN | `GETN` | `GET` | `GET` |
| eva | CAT | `CATS` | `CATS` | `CATS` |
| eva | DOG | `DDOQ` | `DOO` | `DOG` |
| eva | NAME | `ONAME` | `NAME` | `NAME` |
| eva | WATER | `BWATTE` | `BATTLE` | `BUTTE` |
| eva | FIVE | `FIVE` | `FIVE` | `FIVE` |
| eva | TAXI | `STAOI` | `STAGE` | `STAG` |
| eva | GOOD | `OROQODVD` | `ORTHODOX` | `GROOVED` |
| isha | HELLO | `HEL` | `HE` | `HELL` |
| isha | BOOK | `BOV` | `BOY` | `BOW` |
| isha | PENN | `QEN` | `MEN` | `PEN` |
| isha | CAT | `CAT` | `CAT` | `CAT` |
| isha | DOG | `POV` | `POV` | `POV` |
| isha | NAME | `NATM` | `NATO` | `NATS` |
| isha | WATER | `WATR` | `WATER` | `WAR` |
| isha | FIVE | `FIVM` | `FIVE` | `FIRM` |
| isha | TAXI | `TAI` | `TAI` | `TAI` |
| isha | GOOD | `VOX` | `BOX` | `VOX` |

Streams (all 10 words in one caption):

- eva raw: `HECOCO BOK GETN CATS DDOQQ NAME BWATTE FIVE STAOI OROQODVD`
  dictionary: `COCO BOOK GET CATS DOOR NAME BATTLE FIVE STAGE ORTHODOX`  LLM: `HECTOR BOOK GETS CATS DOG NAME BATTLE FIVE STAGE GROOVED`
- isha raw: `HEL BOV QEN CAT POV NATM WATR FIVM TAI VOX`
  dictionary: `HE BOY MEN CAT POV NATO WATER FIVE TAI BOX`  LLM: `HELL BOW PEN CAT POW NAME WATER FIVE TAIL VOX`
