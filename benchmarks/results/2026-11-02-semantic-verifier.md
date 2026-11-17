# Benchmark — semantic cache with LLM verification (2026-11-02)

Embeddings: `text-embedding-3-small`. 50 paraphrase pairs, 50 look-alike pairs. Question A = stored, question B = new.

A pair is reused when its cosine similarity is at least the floor AND the judge says
both questions ask for the same thing.

## Judge `gpt-4o-mini` (openai)

Tokens for 100 checks: 26348 in, 100 out.

| Similarity floor | Reuse (paraphrases) | False positives (look-alikes) |
| --- | --- | --- |
| 0.00 | 90% | 0% |
| 0.50 | 90% | 0% |
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

Mistakes (5):

- paraphrase rejected: 'How do I change my shipping address?' / 'Can I update the delivery address on my order?' (cosine 0.71)
- paraphrase rejected: 'Can I change the size of my order?' / 'Is it possible to exchange for a different size?' (cosine 0.63)
- paraphrase rejected: 'How do I apply a promo code?' / 'Where do I enter a discount code?' (cosine 0.72)
- paraphrase rejected: 'Are your products vegan?' / 'Do your products contain animal ingredients?' (cosine 0.76)
- paraphrase rejected: 'Quels moyens de paiement acceptez-vous ?' / 'Comment puis-je payer ?' (cosine 0.66)

## Judge `gpt-4o-mini` (openai, both orders)

Tokens for 100 checks: 37653 in, 143 out.

| Similarity floor | Reuse (paraphrases) | False positives (look-alikes) |
| --- | --- | --- |
| 0.00 | 80% | 0% |
| 0.50 | 80% | 0% |
| 0.60 | 78% | 0% |
| 0.70 | 62% | 0% |
| 0.80 | 30% | 0% |

Realistic retrieval (nearest of all 100 stored questions, then the judge):

| Similarity floor | Correct reuse (paraphrases) | Wrong answers served |
| --- | --- | --- |
| 0.00 | 76% | 0 of 100 questions |
| 0.50 | 76% | 0 of 100 questions |
| 0.60 | 74% | 0 of 100 questions |
| 0.70 | 60% | 0 of 100 questions |
| 0.80 | 30% | 0 of 100 questions |

Answers served from another stored question:

Mistakes (10):

- paraphrase rejected: 'How long does delivery take?' / 'How many days until my order arrives?' (cosine 0.55)
- paraphrase rejected: 'How do I change my shipping address?' / 'Can I update the delivery address on my order?' (cosine 0.71)
- paraphrase rejected: 'How do I delete my account?' / 'I want to close my account, how do I do it?' (cosine 0.78)
- paraphrase rejected: 'Can I change the size of my order?' / 'Is it possible to exchange for a different size?' (cosine 0.63)
- paraphrase rejected: 'How do I apply a promo code?' / 'Where do I enter a discount code?' (cosine 0.72)
- paraphrase rejected: 'Are your products vegan?' / 'Do your products contain animal ingredients?' (cosine 0.76)
- paraphrase rejected: 'Quels moyens de paiement acceptez-vous ?' / 'Comment puis-je payer ?' (cosine 0.66)
- paraphrase rejected: 'Où est ma commande ?' / 'Pouvez-vous me dire où se trouve mon colis ?' (cosine 0.70)
- paraphrase rejected: 'Proposez-vous des remboursements ?' / 'Puis-je être remboursé ?' (cosine 0.72)
- paraphrase rejected: 'Comment créer un compte ?' / "Comment m'inscrire ?" (cosine 0.66)

## Judge `anthropic/claude-haiku-5.5` (openrouter)

Tokens for 100 checks: 36449 in, 832 out.

| Similarity floor | Reuse (paraphrases) | False positives (look-alikes) |
| --- | --- | --- |
| 0.00 | 28% | 0% |
| 0.50 | 28% | 0% |
| 0.60 | 28% | 0% |
| 0.70 | 20% | 0% |
| 0.80 | 16% | 0% |

Realistic retrieval (nearest of all 100 stored questions, then the judge):

| Similarity floor | Correct reuse (paraphrases) | Wrong answers served |
| --- | --- | --- |
| 0.00 | 24% | 0 of 100 questions |
| 0.50 | 24% | 0 of 100 questions |
| 0.60 | 24% | 0 of 100 questions |
| 0.70 | 18% | 0 of 100 questions |
| 0.80 | 16% | 0 of 100 questions |

Answers served from another stored question:

Mistakes (36):

- paraphrase rejected: 'How do I return an item?' / 'How can I send back a product I bought?' (cosine 0.71)
- paraphrase rejected: 'What are your opening hours?' / 'When are you open?' (cosine 0.79)
- paraphrase rejected: 'Do you ship internationally?' / 'Can you deliver to other countries?' (cosine 0.64)
- paraphrase rejected: 'How long does delivery take?' / 'How many days until my order arrives?' (cosine 0.55)
- paraphrase rejected: 'Can I cancel my order?' / 'Is it possible to cancel an order I placed?' (cosine 0.86)
- paraphrase rejected: 'Where is my order?' / 'Can you tell me where my package is?' (cosine 0.69)
- paraphrase rejected: 'Do you offer refunds?' / 'Can I get my money back?' (cosine 0.57)
- paraphrase rejected: 'How do I change my shipping address?' / 'Can I update the delivery address on my order?' (cosine 0.71)
- paraphrase rejected: 'What is your phone number?' / 'Which number can I call you on?' (cosine 0.66)
- paraphrase rejected: 'How do I delete my account?' / 'I want to close my account, how do I do it?' (cosine 0.78)
- paraphrase rejected: 'Do you have a mobile app?' / 'Is there an app for your service?' (cosine 0.72)
- paraphrase rejected: 'Can I change the size of my order?' / 'Is it possible to exchange for a different size?' (cosine 0.63)
