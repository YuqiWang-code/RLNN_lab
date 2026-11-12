# Positioning: wall clock against independent-set size

100 validation graphs, search seed [1017], frozen backbone and frozen PAVE (horizon=1). Size = mean count of vertices in the returned independent set; wall = measured mean seconds per graph for one full solve.

## Budget curve

| method | clock | steps | size | wall s/graph |
| --- | --- | --- | --- | --- |
| KaMIS (ReduMIS) | 0.1 s limit | - | 20.0450 | 1.87 |
| KaMIS (ReduMIS) | 0.5 s limit | - | 20.0600 | 2.67 |
| KaMIS (ReduMIS) | 1 s limit | - | 20.0650 | 3.12 |
| KaMIS (ReduMIS) | 4 s limit | - | 20.0650 | 4.93 |
| RLNN captured | 0.5 s | 795 | 19.8850 | 0.49 |
| RLNN eager | 0.5 s | 121 | 19.6900 | 0.48 |
| RLNN captured | 1 s | 2348 | 20.0000 | 0.99 |
| RLNN eager | 1 s | 301 | 19.7750 | 0.98 |
| RLNN captured | 2.5 s | 7285 | 20.0400 | 2.49 |
| RLNN captured | 5 s | 15661 | 20.0500 | 4.98 |

## Engine ablation at equal clock

| clock | eager size | captured size | paired captured-eager [CI] | wins/ties/losses | steps ratio |
| --- | --- | --- | --- | --- | --- |
| 0.5 s | 19.6900 | 19.8850 | +0.1950 [+0.1250,+0.2700] | 21/79/0 | x6.75 |
| 1 s | 19.7750 | 20.0000 | +0.2250 [+0.1400,+0.3200] | 22/78/0 | x7.88 |

## At matched wall clock

KaMIS has a floor of about 1.9 s per graph even at a 0.1 s limit, so the small clocks have no KaMIS entry.

| wall s/graph | KaMIS size | RLNN captured size | paired captured-KaMIS [CI] | wins/ties/losses |
| --- | --- | --- | --- | --- |
| 1.87 (KaMIS 0.1 s) vs 2.49 (ours 2.5 s) | 20.0450 | 20.0400 | -0.0050 [-0.0450,+0.0350] | 2/94/4 |
| 2.67 (KaMIS 0.5 s) vs 2.49 (ours 2.5 s) | 20.0600 | 20.0400 | -0.0200 [-0.0600,+0.0200] | 2/93/5 |
| 3.12 (KaMIS 1 s) vs 2.49 (ours 2.5 s) | 20.0650 | 20.0400 | -0.0250 [-0.0700,+0.0150] | 2/93/5 |
| 4.93 (KaMIS 4 s) vs 4.98 (ours 5 s) | 20.0650 | 20.0500 | -0.0150 [-0.0500,+0.0200] | 1/96/3 |

