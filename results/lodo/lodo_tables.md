# Leave-one-dataset-out results (mean ± std over seeds)

### Held out: BoT  (trained on CICIDS, ToN, UNSW)

| Metric | Trained avg (CICIDS, ToN, UNSW) | BoT (held-out) |
|---|---|---|
| ROC-AUC | 0.8032 ± 0.0352 | 0.6532 ± 0.0000 |
| Recall | 0.7775 ± 0.0991 | 0.9632 ± 0.0000 |
| Precision | 0.6711 ± 0.3369 | 0.9996 ± 0.0000 |
| F1 | 0.6692 ± 0.2076 | 0.9811 ± 0.0000 |
| False Positive Rate | 0.1630 ± 0.1467 | 0.4995 ± 0.0000 |

### Held out: CICIDS  (trained on BoT, ToN, UNSW)

| Metric | Trained avg (BoT, ToN, UNSW) | CICIDS (held-out) |
|---|---|---|
| ROC-AUC | 0.6994 ± 0.0737 | 0.7689 ± 0.0000 |
| Recall | 0.7439 ± 0.1891 | 0.8939 ± 0.0000 |
| Precision | 0.7011 ± 0.3266 | 0.2809 ± 0.0000 |
| F1 | 0.7089 ± 0.2657 | 0.4275 ± 0.0000 |
| False Positive Rate | 0.3254 ± 0.2647 | 0.3097 ± 0.0000 |

### Held out: ToN  (trained on BoT, CICIDS, UNSW)

| Metric | Trained avg (BoT, CICIDS, UNSW) | ToN (held-out) |
|---|---|---|
| ROC-AUC | 0.8158 ± 0.0544 | 0.7672 ± 0.0000 |
| Recall | 0.7947 ± 0.0552 | 0.7049 ± 0.0000 |
| Precision | 0.7169 ± 0.2661 | 0.7776 ± 0.0000 |
| F1 | 0.7410 ± 0.1698 | 0.7395 ± 0.0000 |
| False Positive Rate | 0.1419 ± 0.1206 | 0.2255 ± 0.0000 |

### Held out: UNSW  (trained on BoT, CICIDS, ToN)

| Metric | Trained avg (BoT, CICIDS, ToN) | UNSW (held-out) |
|---|---|---|
| ROC-AUC | 0.8027 ± 0.1344 | 0.8186 ± 0.0000 |
| Recall | 0.8250 ± 0.1098 | 0.7514 ± 0.0000 |
| Precision | 0.8669 ± 0.1167 | 0.9474 ± 0.0000 |
| F1 | 0.8439 ± 0.1042 | 0.8381 ± 0.0000 |
| False Positive Rate | 0.2330 ± 0.2138 | 0.0041 ± 0.0000 |
