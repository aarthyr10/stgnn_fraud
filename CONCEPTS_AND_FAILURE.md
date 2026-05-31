# Concepts & Failure Analysis — Prior-Tracking STGNN

A study companion to the demo. Part 1 explains, in plain words, *why the
headline targets were missed*. Part 2 explains *every concept* in the
pipeline so the whole flow makes sense end to end.

---

# Part 1 · Why it fell short

## The one-sentence version

The correction layer (our contribution) was fine; the **encoder feeding
it produced almost the same output every week**, so there was nothing
for the correction to work with — and a strong correction on top of weak
signal can't manufacture accuracy that isn't there.

## The scorecard, honestly

| Promise | Target | Got | Met? |
|---|---|---|---|
| (a) network accurate | F1 ≈ 0.69 | ~0.49–0.54 | No |
| (b) batch correction after shutdown | F1 0.08–0.20 | ~0.077 | Borderline |
| (c) live correction after shutdown | F1 ≥ 0.18 | ~0.06–0.07 | No |
| (d) live tracker follows true rate | ρ ≥ 0.70 | best 0.61, median ~0.07 | No |
| (e) correction works on any model | RF+ ρ > 0 | +0.18 | **Yes** |
| (ref) baseline matches literature | F1 ≈ 0.82 | 0.83 | **Yes** |

## The failure, as a chain of cause and effect

**Root cause — the encoder is nearly a constant function over time.**
The diagnostic showed the GCN-GRU's raw "chance this is illicit" output
sitting at essentially the *same value* (~19%) at every test week. If the
model says the same thing every week, it isn't really using time at all.

That single fact causes every downstream miss:

1. **Why F1 capped at ~0.50.** A classifier that outputs near-identical
   scores can't cleanly separate fraud from normal. ~0.50 is roughly the
   ceiling you hit when the model has weak discriminative signal on a
   ~10%-fraud, heavily-imbalanced dataset. (Tellingly, the plain GCN
   *without* the GRU scored slightly higher, ~0.52 — meaning the GRU was
   removing signal, not adding it.)

2. **Why the prior tracker had nothing to re-rank.** Our "thermostat"
   works by *re-ordering* suspects as the fraud rate changes. But if every
   week's scores are flat and identical, multiplying them all by a
   changing number shifts everyone equally — the *ranking* barely moves.
   The diagnostic confirmed this: changing the tracker's knobs (α, β, EM
   iterations) barely changed the metrics, because the knobs operate on
   output that has no week-to-week structure to begin with.

3. **Why post-shutdown F1 was near zero in early runs.** The GRU was
   *miscalibrated*: its effective training fraud rate read as ~19% when
   the truth was ~10% — biased about 2× too high. The Saerens correction
   multiplies each score by (new rate ÷ training rate). With the training
   rate inflated, that ratio pushes illicit scores *down*, so after
   "correction" almost nothing crossed the decision threshold → almost no
   fraud flagged after the shutdown.

4. **Why the correlation (ρ) swung wildly between runs.** When the
   encoder gives no real signal, the tracker's estimate of the weekly
   fraud rate is basically noise. Noise correlates with the truth by luck:
   one seed got +0.61, another got −0.75, and the median across 10 seeds
   was ~0.07 (i.e., no reliable relationship). High variance like that is
   itself a symptom of "there's no signal underneath."

## Why the two passing results matter most

The two promises that held are the ones that actually validate the
*idea*:

- **The Random Forest baseline hit F1 0.83**, matching the published
  ~0.82. That proves the experimental setup, data handling, and metrics
  are correct — the rig is sound.
- **The tracker bolted onto the Random Forest produced a positive
  correlation (ρ ≈ +0.18).** The Random Forest is a genuinely strong,
  reasonably calibrated classifier whose scores *do* vary week to week —
  so the tracker had real signal to track, and it tracked it in the right
  direction. Crucially, the tracker was never designed for the Random
  Forest, which is the whole point of prediction (e): the correction is
  **architecture-agnostic**. It works on top of *any* classifier that
  produces decent, varying scores.

So the contribution is directionally validated. The limiting factor is
upstream of it.

## The deeper reason the encoder went flat

Three things compounded:

- **Freezing.** The plan was to train the GCN once, freeze it, and let the
  GRU learn on top of its fixed outputs. But a frozen GCN can't adapt to
  the temporal task, and a GRU reading fixed summaries had little new to
  learn — so it collapsed toward a constant.
- **The strict time-split.** Elliptic is split so the model trains on
  early weeks and is tested on strictly later weeks it has never seen
  ("strict inductive"). That's the honest, hard way to evaluate, but it
  leaves limited week-to-week signal the model can latch onto.
- **Imbalance and missing labels.** Only ~10% of *labelled* nodes are
  illicit, and most nodes are unlabelled. Thin, sparse supervision makes
  it hard to learn a sharp decision boundary.

## What was tried, and what would actually fix it

Tried: **unfreezing** the GCN and training it end-to-end with the GRU
(the "joint" mode). It helped a little but didn't close the gap, because
the data-split ceiling is real, not just a training choice.

The clear next steps (all about the *encoder*, not the tracker):

- **Confidence calibration** — a one-parameter temperature scaling so the
  GRU's 19% reads as the true 10%; this alone would unblock the
  post-shutdown collapse.
- **More temporal capacity** — bigger/deeper GRU, or more training.
- **A stronger temporal encoder** — the codebase already contains a
  temporal-attention model (TGAT); swapping it in keeps the same
  Saerens-correction principle while feeding it better signal.

The headline: *fix the signal, not the thermostat.*

---

# Part 2 · Every concept in the flow

Read top to bottom; each idea builds on the last.

## The data

- **Elliptic Bitcoin dataset** — a real, public dataset of ~200k Bitcoin
  transactions. Each transaction is described by **166 numbers**
  (features), is connected to other transactions it sent money to/from,
  and is labelled **illicit**, **licit**, or — for most of them —
  **unknown**.
- **Node / edge / graph** — a *node* is one transaction (a dot); an *edge*
  is a money flow between two transactions (a line); the whole thing is a
  *graph* (a web of dots and lines). Graphs matter here because fraud is
  *relational* — a transaction looks suspicious partly because of *who it
  deals with*.
- **Timesteps** — the data is sliced into **49 weekly snapshots**. The
  same kinds of nodes appear week after week, so we can watch behaviour
  *change over time*.
- **Class imbalance** — only ~10% of labelled nodes are illicit. This is
  why we don't measure plain "accuracy" (a model that says "all licit"
  would score 90% and catch zero fraud).
- **Strict temporal / inductive split** — we train on early weeks and test
  on strictly *later* weeks the model has never seen. Realistic and hard:
  it forbids the model from "memorising" the future.
- **t = 43 "shutdown"** — a point in the timeline where the mix of fraud
  changes sharply (modelled on a real-world marketplace takedown). The
  weeks after it are the stress test.

## The models

- **Embedding** — a compact list of numbers that summarises a node after
  the model has "understood" it (e.g., 64 numbers capturing its role in
  the graph). Similar nodes get similar embeddings.
- **GCN (Graph Convolutional Network)** — a model that builds each node's
  embedding by **mixing in information from its neighbours**, then their
  neighbours, and so on ("message passing"). Intuition: judge an account
  partly by the company it keeps.
- **GRU (Gated Recurrent Unit)** — a model for **sequences** that keeps a
  small running **memory** ("hidden state") and updates it each step using
  little "gates" that decide what to keep and what to forget. Here it
  reads a node's embedding week after week to capture *how it changes over
  time*.
- **GCN → GRU hybrid (the STGNN)** — "spatio-temporal": the GCN handles
  *space* (the graph structure) and the GRU handles *time* (the weekly
  sequence). Spatial first, temporal second.
- **Freezing / transfer** — train the GCN once, **save its weights to a
  file**, and reuse those fixed embeddings. This gives the project its
  "save a model, load it into another program" property — and, as Part 1
  explains, is also part of why the encoder went flat.
- **Random Forest** — a classic, non-graph model: it builds hundreds of
  small decision trees on the raw 166 features and lets them vote. Strong
  and reliable on tabular data — our reference baseline (~0.82 F1 in the
  literature).

## The prior-correction idea (the contribution)

- **Prior / base rate** — the fraction of transactions that are illicit
  *right now*. A model learns the *training* base rate and silently
  assumes it never changes.
- **Prior shift / label shift / non-stationarity** — when that base rate
  *does* change over time (exactly what happens around the shutdown). A
  model frozen at the old rate makes systematically wrong calls.
- **Posterior** — the model's output for one node: "given everything I
  see, the chance this is illicit is X%."
- **Bayes' rule, in one line** — posterior ∝ evidence × prior. If only the
  *prior* changed, you can fix the posterior by rescaling it — **no
  retraining needed**.
- **Saerens-EM correction** — the textbook recipe that does exactly that.
  **EM (Expectation-Maximisation)** is an iterative "guess, refine, guess
  again" loop: guess the new base rate, use it to re-weight every node's
  score, recompute the base rate from those scores, repeat until it
  settles. The re-weighting factor is (new rate ÷ training rate) per
  class.
- **α and β (the Beta prior knobs)** — gentle guardrails on that estimate
  so EM doesn't chase noise to an absurd value. They encode a soft belief
  like "fraud is usually around 10%, don't stray too far without strong
  evidence."
- **The three conditions** —
  - **C1 (none)** — no correction; the static baseline.
  - **C2 (batch)** — estimate the new base rate *once* over the whole test
    period and correct everything by it.
  - **C3 (online per-timestep)** — **our contribution**: re-estimate the
    base rate *fresh every week* and correct that week's scores with it,
    so the model keeps pace with the drift in real time.
- **Calibration** — whether a model's stated probabilities match reality
  (of all the nodes it calls "70% illicit," are ~70% actually illicit?).
  Our GRU was *miscalibrated* (read ~19% vs true ~10%), which broke the
  correction math. Fix = temperature scaling.
- **Decision threshold** — the cutoff that turns a probability into a
  yes/no ("flag if illicit-probability > T"). Moving T trades catching
  more fraud against raising more false alarms.

## The metrics (how we grade it)

- **Precision** — of the nodes we flagged, how many were really fraud
  (low precision = crying wolf).
- **Recall** — of all real fraud, how much we caught (low recall =
  misses).
- **F1** — one number balancing precision and recall (their harmonic
  mean). 1.0 is perfect; the right summary when classes are imbalanced.
- **PR-AUC (Precision–Recall Area Under Curve)** — measures *ranking*
  quality across all thresholds: if you investigated suspects top-down,
  how many would be real? Robust under imbalance.
- **Recall @ 5% FPR** — of all real fraud, how much we catch while keeping
  false alarms to 5%. Mirrors a real analyst's limited time budget.
- **Spearman ρ (rho)** — a **rank** correlation: does our *estimated*
  weekly fraud rate rise and fall in step with the *true* one? +1 = in
  step, 0 = unrelated, −1 = backwards. This is the direct score for "is
  the tracker tracking?"
- **Pearson vs Spearman** — Pearson measures straight-line correlation;
  Spearman only cares about *order*, so it's robust to scale and outliers
  — appropriate for "do they move together?"

## The experimental rigour

- **Seed** — the starting point for the random number generator. Training
  involves randomness, so different seeds give slightly different models.
- **Seed sweep / bootstrap** — run the whole thing across many seeds (10
  here) and report the **median** with a **confidence interval**, instead
  of cherry-picking one lucky run. Honest science.
- **Variance as a diagnostic** — when results swing wildly across seeds
  (ours: ρ from −0.75 to +0.61), that *instability itself* is evidence the
  model has little real signal — a stable result would land in a tight
  band.

## How it all connects (the flow in one breath)

Bitcoin transactions form a **graph** that changes **weekly**. A **GCN**
turns each transaction into an **embedding** from its neighbourhood; a
**GRU** reads those embeddings over time. The model outputs a
**posterior** (chance of fraud), assuming a fixed **prior**. Because the
real fraud rate **drifts**, we add a **Saerens correction** that
re-estimates the prior — once (**C2**) or **every week (C3, our
contribution)** — and rescales the scores, **no retraining**. We grade
with **F1 / PR-AUC / recall** for accuracy and **Spearman ρ** for
tracking, across **10 seeds**. Result: the correction is sound and
**architecture-agnostic** (it works on the **Random Forest**), but the
**GCN-GRU encoder** produced near-constant weekly output, capping
accuracy — so the fix lies in a **stronger/calibrated encoder**, not the
correction layer.
