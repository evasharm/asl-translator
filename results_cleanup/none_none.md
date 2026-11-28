## Iteration 10: caption cleanup, dictionary baseline only

| Version | eva CER | isha CER | All CER | Word acc | Over-correction | Made worse | Stream CER |
|---|---|---|---|---|---|---|---|
| raw caption | 49.5% | 40.2% | 44.8% | 10% | - | - | 36.7% |
| dictionary fix | 52.7% | 37.7% | 45.2% | 30% | 0% | 30% | 38.8% |

Over-correction is measured on the 2 of 20 words that were already right before the fix.

| Person | Word | Raw | Dictionary |
|---|---|---|---|
| eva | HELLO | `HECOCO` | `COCO` |
| eva | BOOK | `BOK` | `BOOK` |
| eva | PENN | `GETN` | `GET` |
| eva | CAT | `CATS` | `CATS` |
| eva | DOG | `DDOQ` | `DOO` |
| eva | NAME | `ONAME` | `NAME` |
| eva | WATER | `BWATTE` | `BATTLE` |
| eva | FIVE | `FIVE` | `FIVE` |
| eva | TAXI | `STAOI` | `STAGE` |
| eva | GOOD | `OROQODVD` | `ORTHODOX` |
| isha | HELLO | `HEL` | `HE` |
| isha | BOOK | `BOV` | `BOY` |
| isha | PENN | `QEN` | `MEN` |
| isha | CAT | `CAT` | `CAT` |
| isha | DOG | `POV` | `POV` |
| isha | NAME | `NATM` | `NATO` |
| isha | WATER | `WATR` | `WATER` |
| isha | FIVE | `FIVM` | `FIVE` |
| isha | TAXI | `TAI` | `TAI` |
| isha | GOOD | `VOX` | `BOX` |

Streams (all 10 words in one caption):

- eva raw: `HECOCO BOK GETN CATS DDOQQ NAME BWATTE FIVE STAOI OROQODVD`
  dictionary: `COCO BOOK GET CATS DOOR NAME BATTLE FIVE STAGE ORTHODOX`
- isha raw: `HEL BOV QEN CAT POV NATM WATR FIVM TAI VOX`
  dictionary: `HE BOY MEN CAT POV NATO WATER FIVE TAI BOX`
