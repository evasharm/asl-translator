## Iteration 10: caption cleanup, LLM (ollama: llama3.2)

| Version | eva CER | isha CER | All CER | Word acc | Over-correction | Made worse | Stream CER |
|---|---|---|---|---|---|---|---|
| raw caption | 49.5% | 40.2% | 44.8% | 10% | - | - | 36.7% |
| dictionary fix | 52.7% | 37.7% | 45.2% | 30% | 0% | 30% | 38.8% |
| LLM (ollama: llama3.2) | 55.3% | 85.8% | 70.6% | 10% | 100% | 30% | 39.8% |

Over-correction is measured on the 2 of 20 words that were already right before the fix.
LLM latency: 0.30 s per call on average (median 0.11 s, 22 calls).

| Person | Word | Raw | Dictionary | LLM |
|---|---|---|---|---|
| eva | HELLO | `HECOCO` | `COCO` | `HEKOKO` |
| eva | BOOK | `BOK` | `BOOK` | `BOCK` |
| eva | PENN | `GETN` | `GET` | `GENT` |
| eva | CAT | `CATS` | `CATS` | `COTS` |
| eva | DOG | `DDOQ` | `DOO` | `DOOQ` |
| eva | NAME | `ONAME` | `NAME` | `ONAME` |
| eva | WATER | `BWATTE` | `BATTLE` | `BATTERY` |
| eva | FIVE | `FIVE` | `FIVE` | `FIVES` |
| eva | TAXI | `STAOI` | `STAGE` | `STAOI` |
| eva | GOOD | `OROQODVD` | `ORTHODOX` | `OROGODVD` |
| isha | HELLO | `HEL` | `HE` | `HELLO` |
| isha | BOOK | `BOV` | `BOY` | `BOV` |
| isha | PENN | `QEN` | `MEN` | `QEN` |
| isha | CAT | `CAT` | `CAT` | `COT` |
| isha | DOG | `POV` | `POV` | `POVS` |
| isha | NAME | `NATM` | `NATO` | `NATM` |
| isha | WATER | `WATR` | `WATER` | `WATER` |
| isha | FIVE | `FIVM` | `FIVE` | `FIVM CORRECTED TO FIVE` |
| isha | TAXI | `TAI` | `TAI` | `TAY` |
| isha | GOOD | `VOX` | `BOX` | `VOK` |

Streams (all 10 words in one caption):

- eva raw: `HECOCO BOK GETN CATS DDOQQ NAME BWATTE FIVE STAOI OROQODVD`
  dictionary: `COCO BOOK GET CATS DOOR NAME BATTLE FIVE STAGE ORTHODOX`  LLM: `HEGOCO BOOK GETN CATS DDOQQ NAME BOWATTE FIVE STAOI OROQODVD`
- isha raw: `HEL BOV QEN CAT POV NATM WATR FIVM TAI VOX`
  dictionary: `HE BOY MEN CAT POV NATO WATER FIVE TAI BOX`  LLM: `HEL BOY QEN CAT POV NATM WATER FIVE MATE TAI VOY`
