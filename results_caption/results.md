## Iteration 8: caption rules (two-stage landmarks)

Character error rate, lower is better. Parameters tuned on the other person.

| Rule | eva word CER | eva stream CER | isha word CER | isha stream CER | Mean word CER |
|---|---|---|---|---|---|
| 1 raw argmax | 721.7% | 538.8% | 356.5% | 283.7% | 539.1% |
| 2 majority vote | 143.8% | 93.9% | 102.7% | 71.4% | 123.2% |
| 3 confidence-weighted vote | 115.7% | 73.5% | 65.2% | 53.1% | 90.4% |
| 4 + hold rule | 49.5% | 49.0% | 40.2% | 51.0% | 44.8% |
| 5 + space on hand drop | 49.5% | 40.8% | 40.2% | 32.7% | 44.8% |

Parameters chosen (tuned on the other person):

- test eva: {'W': 4, 'c_min': 0.0, 'H': 10, 'd_min': 0.6, 'G': 2}
- test isha: {'W': 4, 'c_min': 0.0, 'H': 22, 'd_min': 0.6, 'G': 2}

Captions per word (rule 1 raw -> rule 5 final):

| Word | eva raw | eva final | isha raw | isha final |
|---|---|---|---|---|
| HELLO | `HKHXFOEFCFCOEOCOCOCXC...` | `HECOCO` | `HOEDLDCO` | `HEL` |
| BOOK | `FHCBCBCOCOCOCOMOMORK` | `BOK` | `WBCOCOCOCVKVWCH` | `BOV` |
| PENN | `HTHXOPGPGPGPGPGXAXEFO...` | `GETN` | `HGPGPGQGQPNOEOEOEODFR...` | `QEN` |
| CAT | `ECVOCOCOCOCOCVCROASAK...` | `CATS` | `HCHCOALXPT` | `CAT` |
| DOG | `LDVODCOCDRKHVCPOHGHPO...` | `DDOQ` | `PLDRPTPLDCODPHVH` | `POV` |
| NAME | `OGIODTKTNTRAFEFENME` | `ONAME` | `GHNRLADWTEFEMOAMAMAMA...` | `NATM` |
| WATER | `CGBWFAXTATATATDEDT` | `BWATTE` | `WBFOAKRWKTATLDEMEDODM...` | `WATR` |
| FIVE | `HFIEOVKDFE` | `FIVE` | `FIFRURUVBFEOMEO` | `FIVM` |
| TAXI | `HTSASASTATXAXTDODXADO...` | `STAOI` | `TLATODXDXHDHDOIM` | `TAI` |
| GOOD | `CPORORGROROROPOQRORPX...` | `OROQODVD` | `VHCOCODLPXL` | `VOX` |

Streams (all 10 clips joined, final rule):

- eva: `HECOCO BOK GETN CATS DDOQQ NAME BWATTE FIVE STAOI OROQODVD`
- isha: `HEL BOV QEN CAT POV NATM WATR FIVM TAI VOX`

## Iteration 9b: single pass vs two-stage on the word recordings

| Landmarks | eva word CER (final rule) | isha word CER (final rule) | Mean |
|---|---|---|---|
| single pass | 46.3% | 40.2% | 43.2% |
| two-stage | 49.5% | 40.2% | 44.8% |

Frames per person: eva 3263 (92% with a hand), isha 1776 (90% with a hand)
