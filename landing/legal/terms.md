---
title: "Terms of Service"
description: "The rules for using CAELUM Cloud: plans, payment, acceptable use, data and liability."
---

**Version 1.0 · Last updated: November 16, 2026**

These terms govern your use of **CAELUM Cloud** (the hosted dashboard, shared cache and related
services at `app.caelum-ai.com` and `api.caelum-ai.com`), provided by **Issouf KANAZOE**
("CAELUM", "we", "us"). If a company takes over CAELUM, it will take over these terms and we
will tell you by email.

The open-source CAELUM SDK is a separate product, licensed under the Apache License 2.0. You can
use it without CAELUM Cloud and without accepting these terms.

By creating an account or using CAELUM Cloud, you accept these terms. If you use it for a
business, you accept them on behalf of that business.

## 1. The service

CAELUM Cloud adds, to the CAELUM SDK running in your application:

- a dashboard of your AI requests, cache hits, estimated costs and savings;
- a cache of AI answers shared by all your servers;
- on the Starter plan: cost per end customer, budget alerts by email, a repetition counter
  shared by all your servers, and 90 days of history.

**Your AI calls never go through CAELUM Cloud.** Your application calls the AI provider
(OpenAI, Anthropic, Google, Mistral...) directly with your own API key, which you never give to
us. You remain bound by each provider's own terms and are billed by them directly. CAELUM
never resells AI tokens.

The SDK is designed so that your application never needs CAELUM Cloud to work: if the Cloud is
slow, unavailable, or your plan has ended, the SDK continues with its local cache.

## 2. Your account

- You must give accurate information and keep your sign-in method secure.
- **CAELUM API keys** (`sk-caelum-...`) give access to your account's data: keep them secret,
  and revoke any key you think has leaked from the dashboard.
- You are responsible for everything done with your account and keys.
- You must be at least 18 years old and able to enter into a contract.

## 3. Plans and payment

| Plan | Price | Included |
| --- | --- | --- |
| Free | $0 | 10,000 questions per month, shared cache, dashboard with 7 days of history |
| Starter | **$29 (17,500 FCFA) per 30 days** | 1,000,000 questions per month, shared cache, cost per end customer, budget alerts, shared repetition counter, 90 days of history |

A **question** is one request your application sends through the CAELUM SDK with a CAELUM key,
answered from the cache or not.

- **Starter is prepaid**: each payment gives 30 days of Starter. **There is no automatic
  renewal and no automatic charge.** We remind you by email before the end of the period. If
  you do not renew, your account returns to Free and your application keeps working.
- Payments are processed by our payment provider (GeniusPay) in West African CFA francs
  (XOF). The amount in dollars is an indication; your bank or mobile money operator may apply
  its own exchange rate and fees.
- An early renewal adds 30 days after the current paid period.
- **Refunds:** within **7 days** of a payment, you can ask for a full refund by email if fewer
  than 1,000 questions were counted on your account since that payment. Otherwise, payments
  are not refundable, except where the law requires it or as stated in section 9.
- Over the monthly question allowance of your plan, the shared cache pauses until the next
  month; usage statistics are still recorded and your application keeps working.
- We may change prices for future periods. A change never affects a period you have already
  paid, and we announce it at least 30 days in advance.

## 4. Cached answers are your choice

- Caching reuses an answer that was given before. **You decide** whether to turn the cache on,
  which settings to use (for example how long answers are kept and how similar two questions
  must be), and for which requests.
- A reused answer may be out of date or not fit a new question. CAELUM includes safeguards
  (automatic exclusions of personal or time-bound questions, verification of similar
  questions, conversations never cached), but they work on a **best-effort** basis and cannot
  catch every case. **You remain responsible for the answers your application gives to your
  own users**, and for turning the cache off where a wrong or outdated answer could cause harm
  (for example medical, legal or financial advice, or prices that change).
- **Savings are not guaranteed.** They depend on how often your questions repeat and on your
  AI provider's prices. Cost and savings figures are **estimates**; your provider's invoice is
  the reference.

## 5. Acceptable use

You agree not to:

- use CAELUM Cloud to break the law or the terms of your AI providers;
- store in the shared cache content you are not allowed to process;
- try to access another customer's data, test or bypass our security without written
  permission, or overload the service on purpose;
- resell CAELUM Cloud itself, or use it to build a competing hosted service, without our
  written agreement.

We may suspend an account that seriously breaks these rules, after a warning when the
situation allows it.

You agree to compensate us for the costs and damages we suffer because of claims by third
parties that result from your breach of these terms or from the content your application
sends, to the extent permitted by law.

## 6. Your data

- **You keep all rights to your data**, including the cached answers and the usage data your
  application sends.
- You give us only the rights needed to run the service for you: store, process and display
  that data in your account.
- You are responsible for the content your application sends, including any personal data of
  your own users. Use internal ids, not names or emails, as customer labels. See our
  [Privacy Policy](/privacy).

## 7. Availability and support

- We do our best to keep CAELUM Cloud available, but **no availability guarantee (SLA) is
  included in the Free and Starter plans**. Maintenance or incidents may happen.
- Support is provided by email at **contact@caelum-ai.com**, on a best-effort basis.
- We are not responsible for delays or failures caused by events beyond our reasonable control
  (for example outages of our hosting, payment or email providers, network failures, natural
  disasters, or decisions of public authorities).

## 8. Liability

- CAELUM Cloud is provided **"as is"** and **"as available"**, without any warranty other
  than those that cannot be excluded by law. The Free plan is free and provided without any
  warranty.
- To the extent permitted by law, we are not liable for indirect losses (lost profit, lost
  data, loss of business or reputation, business interruption), nor for answers given by AI
  providers or reused from your cache.
- To the extent permitted by law, our total liability for all claims is limited to the amount
  you paid us in the 3 months before the event that caused the claim.
- Nothing in these terms limits liability that cannot be limited by law.

## 9. Ending the service

- You can stop using CAELUM Cloud at any time, delete your API keys from the dashboard, and ask
  us to delete your account by email.
- We may close an account with at least 30 days' notice, or at once in case of serious breach
  of these terms. If we close the whole service, we give at least 30 days' notice and refund
  the unused part of any paid period.

## 10. Changes and applicable law

- We may update these terms. We announce important changes by email at least 30 days in
  advance. Continuing to use the service after that date means you accept the new terms; if
  you do not, you can stop using it and ask for the refund of the unused part of a paid period.
- These terms are governed by the laws of **Burkina Faso**. Before any court action, we will
  try to settle disagreements amicably by email. Failing that, disputes go to the competent
  courts of **Ouagadougou**, unless mandatory consumer law gives you other rights.

**Contact:** contact@caelum-ai.com
