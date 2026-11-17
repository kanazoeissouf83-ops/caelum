---
title: "Privacy Policy"
description: "What CAELUM Cloud collects, why, where it is stored and for how long."
---

**Version 1.0 · Last updated: November 16, 2026**

This policy explains how **Issouf KANAZOE** ("CAELUM", "we") handles personal data in
**CAELUM Cloud**. If a company takes over CAELUM, it will take over this role and we will tell
you by email.

**The open-source CAELUM SDK alone sends nothing to us.** Data reaches CAELUM Cloud only if you
configure a CAELUM key (`caelum_key=` or `CAELUM_API_KEY`). The SDK's local cache and its
per-request journal stay on your own machine.

## 1. What we collect

| Data | Why | Where it comes from |
| --- | --- | --- |
| **Account:** email, name, sign-in identifiers | Create and secure your account | You, via our sign-in provider (Clerk) |
| **Billing:** name, phone number, payment reference, amount, date | Take payments, send receipts | You, via our payment provider (GeniusPay). **We never see your card or mobile money details.** |
| **Usage metrics:** model name, token counts, estimated cost and savings, cache hit or miss, latency, the customer label you choose | Your dashboard, plan limits, budget alerts | The CAELUM SDK in your application |
| **Shared cache:** the AI answers your application stores in it, under a hashed key and a hash of the end customer | Reuse answers across your servers | The CAELUM SDK, only when the cache is on |
| **Shared repetition counter** (Starter): a hash of each question, a counter and dates. **No question text.** | Cache an answer once it has been asked on any of your servers | The CAELUM SDK |
| **Emails sent** (type, date) | Avoid duplicate emails | Our systems |
| **Technical logs** (IP address, errors) | Security, debugging | Our hosting provider |

What we **never** collect: your AI provider API keys, and the prompts in usage metrics. Prompts
can appear in cached answers only if your provider's answer contains them. Questions that the
SDK detects as containing personal or time-bound data (phone numbers, emails, order numbers,
amounts, dates...) are never cached nor sent to the repetition counter.

## 2. Your role and ours

- For your **account and billing data**, we decide how it is used (we are the "controller").
- For the **content your application sends** (cached answers, customer labels), we act on your
  behalf only to provide the service (we are a "processor"). You decide what your application
  sends; do not put names, emails or other personal data of your users in customer labels.

## 3. Where data is stored, and who helps us

| Provider | Role | Location |
| --- | --- | --- |
| Cloudflare | Hosting of the website, dashboard, API, database and shared cache | Global network |
| Clerk | Sign-in and accounts | United States |
| GeniusPay | Payments | Côte d'Ivoire (West Africa) |
| Resend | Sending emails | Ireland (EU) / United States |

These providers only process data to provide their service to us. Some of them are outside
your country; we choose established providers that protect data in transit and at rest.

## 4. How long we keep data

| Data | Kept |
| --- | --- |
| Shared cache entries | Until they expire: **24 hours by default, 30 days at most** (or earlier if you clear them) |
| Shared repetition counter | Until it expires, with the lifetime of your cached answers (30 days at most) |
| Usage metrics | While your account exists |
| Account data | While your account exists; deleted within 30 days after you ask us to delete it |
| Billing records | 10 years, as required by accounting law |

## 5. Security

- CAELUM keys are stored **only as a hash**: our database never contains usable keys.
- Each account's cache and statistics are **isolated** from every other account; access to
  another account's data is refused even to someone who knows its customer labels.
- Data is encrypted in transit (HTTPS). Access to production systems is restricted to the
  founder.
- If a security incident affects your personal data, we will inform you without undue delay
  once we are aware of it.

## 6. Your rights

You can ask to **access, correct, export or delete** your personal data, or object to a
processing, by writing to **contact@caelum-ai.com**. We answer within 30 days. You can delete
API keys and clear your cached answers at any time from your code and dashboard, and stop
sending data by removing your CAELUM key. You can also complain to your data protection
authority (for example the CIL in Burkina Faso, the Data Protection Commission in Ghana, or
your authority in the European Union).

## 7. Cookies

The dashboard uses only cookies needed for sign-in and security. The public website does not use
advertising or analytics cookies.

## 8. Changes

We will announce important changes by email at least 30 days in advance.

**Contact:** contact@caelum-ai.com
