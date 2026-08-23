# PaperOS citation gold v1

The frozen fixture is derived independently from the supplied MinerU `content_list.json`
and Canonical `references.jsonl`. Formal validation must read the JSON and must not regenerate it.

Baseline:

| paper | spans | atomic targets |
|---|---:|---:|
| Isogeometric GPDE | 75 | 121 |
| Explicit Flows | 73 | 79 |
| DSG-Net | 102 | 155 |
| NISE | 39 | 66 |
| 4Deform | 63 | 98 |
| Volume Preserving | 38 | 41 |

Known non-citation square-bracket domains in MinerU text:
- Isogeometric: `[0, T]`
- DSG-Net: `[; ]`
- NISE internal cross-references: `[1, Sec. 3.5]`, `[3, Cor. 6.2]`
- Volume Preserving: `[0, 1]`

Known non-citation author-year-like publication text:
- `submitted to Eurographics Symposium on Geometry Processing (2025)`

Suggested repo paths:

```text
tests/validation/gold/citation_gold_v1.json
tests/validation/validate_citation_gold.py
```

Run:

```bash
PYTHONPATH=. python tests/validation/validate_citation_gold.py \
  --gold tests/validation/gold/citation_gold_v1.json \
  --run-dir data/validation/runs/chunk
```
