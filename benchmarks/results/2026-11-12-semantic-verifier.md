# Benchmark — semantic cache with LLM verification (2026-11-12)

Embeddings: `text-embedding-3-small`. 50 paraphrase pairs, 50 look-alike pairs. Question A = stored, question B = new.

A pair is reused when its cosine similarity is at least the floor AND the judge says
both questions ask for the same thing.

## Judge `gpt-4o-mini` (openai)

Tokens for 100 checks: 32277 in, 100 out.

| Similarity floor | Reuse (paraphrases) | False positives (look-alikes) |
| --- | --- | --- |
| 0.00 | 88% | 0% |
| 0.50 | 88% | 0% |
| 0.60 | 86% | 0% |
| 0.70 | 66% | 0% |
| 0.80 | 30% | 0% |

Realistic retrieval (nearest of all 100 stored questions, then the judge):

| Similarity floor | Correct reuse (paraphrases) | Wrong answers served |
| --- | --- | --- |
| 0.00 | 80% | 1 of 100 questions |
| 0.50 | 80% | 1 of 100 questions |
| 0.60 | 78% | 1 of 100 questions |
| 0.70 | 62% | 1 of 100 questions |
| 0.80 | 30% | 0 of 100 questions |

Answers served from another stored question:

- new 'How can I send back a product I bought?' answered with stored 'How do I return an item bought online?' (cosine 0.79)

Mistakes (6):

- paraphrase rejected: 'How long does delivery take?' / 'How many days until my order arrives?' (cosine 0.55)
- paraphrase rejected: 'How do I change my shipping address?' / 'Can I update the delivery address on my order?' (cosine 0.71)
- paraphrase rejected: 'Can I change the size of my order?' / 'Is it possible to exchange for a different size?' (cosine 0.63)
- paraphrase rejected: 'How do I apply a promo code?' / 'Where do I enter a discount code?' (cosine 0.71)
- paraphrase rejected: 'Are your products vegan?' / 'Do your products contain animal ingredients?' (cosine 0.76)
- paraphrase rejected: 'Quels moyens de paiement acceptez-vous ?' / 'Comment puis-je payer ?' (cosine 0.66)

Prompt-injection attempts accepted: **2 of 5** (must be 0).

## Judge `gpt-4o-mini` (openai, both orders)

Tokens for 100 checks: 46134 in, 143 out.

| Similarity floor | Reuse (paraphrases) | False positives (look-alikes) |
| --- | --- | --- |
| 0.00 | 78% | 0% |
| 0.50 | 78% | 0% |
| 0.60 | 76% | 0% |
| 0.70 | 60% | 0% |
| 0.80 | 30% | 0% |

Realistic retrieval (nearest of all 100 stored questions, then the judge):

| Similarity floor | Correct reuse (paraphrases) | Wrong answers served |
| --- | --- | --- |
| 0.00 | 72% | 0 of 100 questions |
| 0.50 | 72% | 0 of 100 questions |
| 0.60 | 70% | 0 of 100 questions |
| 0.70 | 56% | 0 of 100 questions |
| 0.80 | 30% | 0 of 100 questions |

Answers served from another stored question:

Mistakes (11):

- paraphrase rejected: 'How long does delivery take?' / 'How many days until my order arrives?' (cosine 0.55)
- paraphrase rejected: 'How do I change my shipping address?' / 'Can I update the delivery address on my order?' (cosine 0.71)
- paraphrase rejected: 'How do I delete my account?' / 'I want to close my account, how do I do it?' (cosine 0.78)
- paraphrase rejected: 'Can I change the size of my order?' / 'Is it possible to exchange for a different size?' (cosine 0.63)
- paraphrase rejected: 'How do I apply a promo code?' / 'Where do I enter a discount code?' (cosine 0.71)
- paraphrase rejected: 'Are your products vegan?' / 'Do your products contain animal ingredients?' (cosine 0.76)
- paraphrase rejected: 'How do I unsubscribe from emails?' / 'How can I stop receiving your newsletters?' (cosine 0.71)
- paraphrase rejected: 'Quels moyens de paiement acceptez-vous ?' / 'Comment puis-je payer ?' (cosine 0.66)
- paraphrase rejected: 'Où est ma commande ?' / 'Pouvez-vous me dire où se trouve mon colis ?' (cosine 0.70)
- paraphrase rejected: 'Proposez-vous des remboursements ?' / 'Puis-je être remboursé ?' (cosine 0.72)
- paraphrase rejected: 'Comment créer un compte ?' / "Comment m'inscrire ?" (cosine 0.66)

Prompt-injection attempts accepted: **0 of 5** (must be 0).
