# Benchmark — semantic cache calibration (2026-11-02)

Embedding model: `text-embedding-3-small` — 191 texts,
50 paraphrase pairs, 50 look-alike pairs.
Current default: 0.97.

**Recommended default threshold: 0.95** (lowest threshold with zero false positives on this set).

## Reuse / false-positive curve

| Threshold | Reuse (paraphrases) | False positives (look-alikes) |
|---|---|---|
| 0.80 | 30% | 38% |
| 0.81 | 26% | 30% |
| 0.82 | 26% | 28% |
| 0.83 | 22% | 28% |
| 0.84 | 20% | 24% |
| 0.85 | 16% | 20% |
| 0.86 | 12% | 18% |
| 0.87 | 12% | 8% |
| 0.88 | 12% | 6% |
| 0.89 | 8% | 6% |
| 0.90 | 6% | 4% |
| 0.91 | 4% | 4% |
| 0.92 | 2% | 2% |
| 0.93 | 2% | 2% |
| 0.94 | 2% | 2% |
| 0.95 | 2% | 0% |
| 0.96 | 2% | 0% |
| 0.97 | 0% | 0% |
| 0.98 | 0% | 0% |
| 0.99 | 0% | 0% |

## Most dangerous look-alike pairs (highest similarity)

| Cosine | Question A | Question B |
|---|---|---|
| 0.9499 | Convert 10 miles to kilometers. | Convert 10 kilometers to miles. |
| 0.9173 | Is coffee good for health? | Is coffee bad for health? |
| 0.8903 | Comment activer la double authentification ? | Comment désactiver la double authentification ? |
| 0.8788 | Give me a vegetarian recipe for dinner. | Give me a vegan recipe for dinner. |
| 0.8675 | Êtes-vous ouverts le samedi ? | Êtes-vous ouverts le dimanche ? |

## Hardest paraphrases (lowest similarity)

| Cosine | Question A | Question B |
|---|---|---|
| 0.5452 | How long does delivery take? | How many days until my order arrives? |
| 0.5675 | Do you offer refunds? | Can I get my money back? |
| 0.6209 | Comment retourner un article ? | Comment puis-je renvoyer un produit acheté ? |
| 0.6258 | Can I change the size of my order? | Is it possible to exchange for a different size? |
| 0.6387 | How do I create an account? | How can I sign up? |
