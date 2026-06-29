# Research History — From Raw Waves to 82% Cross-Subject

*An eight-week log of what failed, what worked, and why. Companion to the [README](README.md), which documents the final system; this document records how it was found.*

---

This is not a list of what worked. It is the record of how a model that classifies
emotions from brain signals was *found* — including the six weeks it spent stuck at a
wall, the three times the data quietly lied, and the moment the model was caught
cheating.

I am writing it down for one reason: the final architecture (a graph network with a
learnable input gate, a frequency-band recalibrator, and a per-subject bias term) looks
arbitrary if you only see the endpoint. Every piece of it is the scar tissue of a
specific failure. Read in order, the choices stop looking clever and start looking
*forced* — which is the only kind of design decision worth trusting.

The whole journey is one number climbing in three jumps:

| Stage | What changed | Accuracy | Reading |
| :-- | :-- | :--: | :-- |
| **Raw waveforms** | 1D-CNN over raw EEG | **~35%** | barely above guessing |
| **DE features** | hand-built spectral features | **~67%** | a hard ceiling, hit from several directions |
| **The final model** | input gating + band attention + subject bias | **~82%** | and on a *harder* test than the rows above |

A caveat that actually makes the last jump bigger than it looks: the first two rows are
*within-subject* (train on a person's early sessions, test on a later one); the final row
is *cross-subject* — Leave-One-Subject-Out, scored on people the model has never seen. The
35→67 climb and the 67→82 climb are the two jumps this document explains; the long,
productive stall between them is most of the story.

---

## Phase 0 — The Honest 35%
**Nov 30 – Dec 6, 2025** · *Primary record: [`docs/reports/PROJECT_SUMMARY.md`](docs/reports/PROJECT_SUMMARY.md)*

> *The question: can a graph network read emotion straight off the raw EEG waveform?*

The first design was the obvious one. Treat the 62 electrodes as a graph wired by
physical distance (k-nearest-neighbours, k=5), run a small 1D-CNN over each channel's
raw time-series to extract temporal features, let graph convolutions mix information
between brain regions, and pool to a label. Clean, end-to-end, no hand-engineering.

Most of week one was not machine learning. It was three bugs, and each one taught me
something about the shape of the problem.

- **The out-of-memory wall.** The graph tensors blew past a 6 GB RTX 3060 at the default
  batch size. Forced down to 32. Mundane — but it set a constraint that shadowed the
  whole project: this is a *memory-bound* problem, and batch size is not free.
- **The Silent GCN.** The model ran, trained, and learned almost nothing. The cause was
  subtle and worth stating precisely: the `edge_index` described *one* 62-node graph, but
  a batch stacks 32 of them into 1,984 nodes. PyTorch Geometric saw nodes 62–1983 as
  *isolated* — no edges — so graph convolution did nothing for 31 of every 32 samples.
  The fix was to replicate the edge structure across the batch with an offset:
  ```python
  offsets = (torch.arange(curr_batch_size) * 62).view(-1, 1, 1)
  edge_index = (base_edge_index.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
  ```
  A graph that silently isn't a graph produces a model that silently isn't learning. The
  loss curve looked *fine*. Only the accuracy gave it away.
- **The collapse.** Once it actually trained, it collapsed to predicting two of three
  classes, ignoring *Positive* entirely. Two causes stacked: weight decay (`1e-3`) was
  crushing weights to zero, and the learning rate (`0.005`) was too hot. Dialled both
  down (`5e-4` LR, `1e-4` decay) and added class weights to force attention onto the
  starved class.

With the pipeline finally honest, the verdict arrived: **~36% on the held-out session.**
Three points above chance. The model learned the training sessions and fell off a cliff
on session three.

Before blaming the architecture, I ran one clean experiment to rule out the simplest
alternative explanation — *noise*. I split the raw signal into a standard band (1–49 Hz)
and an isolated gamma band (50–75 Hz), on the theory that the emotional signal might be
hiding in high frequencies the broadband model was drowning. The standard band
reproduced 36%. The gamma band did something more interesting: it **mode-collapsed** —
it discovered it could game the class weights by predicting *Positive* for almost
everything (recall > 0.90 on that class) and gave up on finding real structure. It found
a statistical shortcut instead of a signal.

**What this phase taught me.** The pipeline was correct; the *input representation* was
the bottleneck. Raw EEG waveforms are non-stationary across days — the waveform for
"happy" on Monday genuinely does not look like "happy" on Wednesday — and a shallow CNN
will always prefer to memorise session-specific noise over learning a day-invariant
concept. You cannot regularise your way out of the wrong input. The honest 35% was the
most useful result of the month, because it killed a whole branch of the search tree.

The pivot it forced: stop feeding the model raw waves. Feed it **Differential Entropy** —
a per-band spectral feature that is far more stable across sessions.

---

## Phase 1 — The 67% Wall
**Dec 6 – Dec 14, 2025** · *Primary records: [`GCN_DE_1s_Report.md`](docs/reports/GCN_DE_1s_Report.md), [`DGCNN_Report.md`](docs/reports/DGCNN_Report.md), [`SUBJECT_DEPENDENT_TRAINING.md`](docs/reports/SUBJECT_DEPENDENT_TRAINING.md)*

> *The question: with better features, how high can a within-subject model go?*

Switching to Differential Entropy across the five canonical bands (Delta, Theta, Alpha,
Beta, Gamma) did exactly what theory promised — accuracy jumped into the mid-60s. And
then it stopped. Hard. At roughly **67%**, and it would not move no matter what I threw
at it. The story of this phase is *three runs at the same wall — two genuine attacks and a
control experiment that proved why they bounced.*

**Attack 1 — tell the model which bands to trust (Fisher scores).** Not every frequency
band carries emotion equally for every person, so I computed Fisher scores per band per
subject and scaled the inputs accordingly. This surfaced the first two ways the *data
itself lies*:

- **The Delta Trap.** For Subject 2, the score crowned Delta (1–3 Hz) the most
  "discriminative" band — and Delta is exactly where eye-blink artifacts live. The
  subject had blinked more during negative clips in training, so the model dutifully
  learned to *detect blinking*. It evaporated on the test session, where the blink
  pattern was different.
- **The Theta Trap.** For Subjects 4 and 12, the winner was Theta (4–7 Hz), the signature
  of *drowsiness*. The model learned "tired vs. alert," not "sad vs. happy." Fatigue
  drifts between sessions; so did the predictions.

A discriminative feature is not the same as a *causal* one. Fisher scores can't tell the
difference between a brain state and a face muscle. That distinction became the spine of
everything that followed.

**Attack 2 — let the model learn the graph (DGCNN).** If a fixed physical graph is the
limitation, learn the connectivity instead. I built it in three escalating forms: a
single global learnable adjacency matrix; then a true input-dependent dynamic graph that
re-wires per sample via self-attention ($A = \mathrm{softmax}(QK^\top/\sqrt{d})$); then
that plus learnable band attention. All three landed at **~67%**, with the *identical*
error pattern — the same handful of subjects failing in the same way.

When three architectures of increasing power converge to the same number, the number is
not telling you about the architecture. It is telling you about the data.

**The control — prove the wall is real (the random split).** I ran one deliberately
"cheating" experiment: shuffle all the one-second windows together, ignoring session
boundaries, and split randomly. If the task were learnable the normal way, this should be
*easier*. Instead it crashed to **~53%** — near chance. That number is a smoking gun. It
means a "sad" signal in session 1 is statistically a *different distribution* from a
"sad" signal in session 3; mixing them doesn't add data, it adds contradiction.

**What this phase taught me.** The ceiling was never spatial (GCN vs. DGCNN) or spectral
(which bands). It was **distributional** — severe non-stationarity across sessions, what
the domain-adaptation literature calls covariate shift, and what I started calling
*negative transfer*: training on sessions 1–2 actively *hurt* session 3, because the
model spent its capacity learning the style of the training days. The architecture search
was over. The data investigation had to begin.

---

## Phase 2 — Reading the Patients
**Dec 14 – Dec 25, 2025** · *Primary records: [`FEATURE_ENGINEERING_STRATEGY.md`](docs/reports/FEATURE_ENGINEERING_STRATEGY.md), [`FEATURE_ENGINEERING_STRATEGY_V2.md`](docs/reports/FEATURE_ENGINEERING_STRATEGY_V2.md), [`EASY_SUBJECTS_ANALYSIS.md`](docs/reports/EASY_SUBJECTS_ANALYSIS.md)*

> *The question: who exactly is the model failing on, and what do their brains look like?*

I stopped tuning models and started reading data — plotting per-channel amplitude
heatmaps, band-power bars, and per-channel variance for the subjects the model loved and
the ones it couldn't touch. The dataset turned out not to be homogeneous at all. It was
four kinds of person wearing the same cap.

- **The Internalizer (Subject 14).** Pure neural signal. In the gamma band, the
  *Positive* class floats visibly above the others across the entire scalp — I started
  calling it **the Green Halo**. The model reads these subjects effortlessly.
- **The Externalizer (Subject 15).** A higher scorer than the Internalizer — but for a
  dangerous reason. A single channel, FC5, shows a massive variance spike *only* for
  negative clips. The subject is **frowning**: that's a jaw/EMG artifact, not a brain
  state. The model exploits it as a near-perfect predictor. Brilliant accuracy, built on
  a muscle.
- **The Hybrid (Subject 06).** Strong neural signal *and* a tell — a variance spike at
  AF4 (a wink/squint) for positive clips. Robust, because if the artifact fails the brain
  signal still carries it.
- **The Stone (Subjects 2 and 12).** The hard ones. In gamma and beta, all three
  emotions overlap into a single glued band — no separation at all. These are the
  subjects every architecture had been failing on. But buried in the *Delta* band there
  was a faint signal: the Neutral class sat lower, because these subjects were physically
  *still* during neutral clips and fidgeted during emotional ones.

Underneath the archetypes was a layer of plain hardware pathology, shared across
subjects: **sinkholes** (Cz, CPz dropping to near-zero — dead vertex contacts that, as a
zero-valued node, drag down every neighbour during graph smoothing) and **screaming
channels** (F7, T7, FC5 throwing huge variance — sometimes loose-contact garbage, as in
Subject 2, and sometimes *real* muscle signal, as in Subject 15's frown). The cruel part:
the *same* symptom — a giant variance spike — was noise on one subject and gold on
another. You could not blindly clean it.

This diagnosis drove three concrete, *justified* engineering decisions — each one a
direct answer to something I had seen in the plots, not a guess:

1. **Add variance as a feature (5 → 10 channels).** Mean DE captures the Green Halo;
   rolling *variance* captures the Frown and the Fidget. Stacking both stops Subject 15
   from looking like Subject 2.
2. **Robust scaling (median/IQR over mean/std).** So that Subject 2's screaming F7 can't
   squash every other channel to zero during normalisation.
3. **Lateral interpolation for dead vertices.** Rebuild Cz from its left/right neighbours
   only (`Cz = (C1 + C2)/2`), pointedly *excluding* CPz because it's dead too.

**What this phase taught me.** Two things, and the second outranks the first. (1) Feature
engineering should be *forensic* — every added feature should answer a pathology you can
point to in a plot. (2) More importantly: half my "good" subjects were good for the wrong
reason. The model wasn't always reading emotion; sometimes it was reading a face muscle,
or simply reading *who the person was*. That suspicion needed a real test. It got one.

---

## Phase 3 — The Model Was Cheating
**Jan 3 – Jan 9, 2026** · *Primary records: [`DATA_CENTRIC_APPROACH_REPORT.md`](docs/reports/DATA_CENTRIC_APPROACH_REPORT.md), [`REPRODUCIBILITY_AND_TRIAL_ANALYSIS.md`](docs/reports/REPRODUCIBILITY_AND_TRIAL_ANALYSIS.md), [`DYNAMIC LEARNING.md`](docs/reports/DYNAMIC%20LEARNING.md)*

> *The question: when the model is right, is it right for the right reason?*

This is the phase where the project grew up. Three findings, each one a small
demolition.

**The binary diagnostic: gamma was a crutch.** I stripped the task down to its hardest
core — *Negative vs. Neutral only*, dropping the easy high-energy Positive class — and
then ablated the gamma band. Without gamma, the hard task ran at 66%. *With* gamma, 68%.
Two points. The headline 3-class accuracy had been almost entirely the Positive class's
gamma signature doing the work; on the genuinely hard discrimination, gamma was nearly
useless. A lot of apparent skill was one easy class in a trench coat.

**The Subject Identity Trap: the model was classifying the *person*, not the *video*.**
I switched from epoch-level accuracy to *per-trial* heatmaps — accuracy for each of the
15 film clips, per subject — and the pathology was unmistakable. Solid horizontal bars of
one colour: Subject 4 predicted *Negative* for **every single clip**; Subject 10 predicted
*Positive* for every clip, regardless of what they were watching. The model had learned
each person's *resting energy fingerprint* — Subject 4 runs low-energy, so "low energy =
negative," forever — and was reciting identity instead of reading emotion. On a graph
that should change with the stimulus, the prediction never moved.

**The Rising Loss Paradox confirmed it.** All through these runs, validation accuracy sat
flat while validation *loss climbed*. That combination is a signature, and once you see
the identity trap it decodes cleanly: early in training the model guesses a subject's
fingerprint-label at 50% confidence; late in training it recognises the fingerprint and
asserts the *same wrong label* at 99% confidence. Accuracy doesn't change — the answer was
already wrong — but cross-entropy punishes confident wrongness savagely, so loss
explodes. The model wasn't getting more confused. It was getting more *arrogant*.

There was also a moment of honest humility in here worth recording: I tried to reproduce
an earlier run ("Attempt 18") that had hit a tantalising 76%, rebuilt what I believed was
its exact configuration — and landed at **67.26%**. I never fully recovered the missing
9%. With no fixed random seed and imperfect provenance, that run is, scientifically, lost.
I wrote it down as lost rather than quietly claiming the 76%. (The same no-seed honesty is
why the README reports every headline as a mean with spread, not a bit-exact digit.)

**What this phase taught me.** The enemy finally had a name, and it was not *emotion is
hard*. The enemy was **the confound** — the model's standing temptation to predict the
subject's identity (or their face muscles, or their fatigue) instead of their brain state,
because the confound is easier and, in-distribution, almost as accurate. Every architecture
so far had quietly surrendered to it. The next phase had to make surrender *expensive*.

---

## Phase 4 — The Silencer
**Jan 9 – Jan 24, 2026** · *Primary records: [`DYNAMIC LEARNING.md`](docs/reports/DYNAMIC%20LEARNING.md), [`ADAPTIVE_GRAPH_INPUT_LAYER.md`](docs/reports/ADAPTIVE_GRAPH_INPUT_LAYER.md)*

> *The question: can the model be made to silence its own unreliable inputs?*

The diagnosis pointed at a clear target: the model needed to learn, *per electrode and
per band*, which inputs to trust — to mute the dead sinkholes and the screaming artifacts
on its own, instead of being hand-fed cleaned data. This is the **Adaptive Graph Input
Layer (AGLI)**: a learnable affine gate on every (channel, band) input,

$$y = x \cdot \gamma + \beta,$$

acting as a learnable pre-amplifier. Drive a noisy channel's gain $\gamma \to 0$ and it
goes silent before it can poison its neighbours in the graph convolution. To *make* the
model use this — to push it toward silence rather than letting every channel stay loud —
I put a much heavier L2 penalty on the gain parameters specifically (weight decay `1e-2`
on $\gamma$ versus `1e-3` elsewhere; it is still wired exactly this way in
[`build_optimizer`](src/eeg_gnn/train.py)). Sparsity as a default; a channel has to *earn*
its volume.

It did not work the first time, and the reason it failed is my favourite bug in the whole
project.

**The LayerNorm that undid the silence.** AGLI would correctly learn to crush a noise
spike from 1.0 down to 0.01 — and then the `LayerNorm` immediately downstream would
re-normalise that 0.01 right back up to ~1.0. Normalisation is *designed* to restore
scale, so it faithfully resurrected exactly the noise AGLI had just buried. Two components,
each correct in isolation, exactly cancelling — AGLI muting each noise spike into a LayerNorm that amplified it
straight back. Removing that normalisation from the gated path was what finally let the
suppression *stick*.

Two more pieces clicked into place in the same stretch:

- **The SE-Block, as a spectral equaliser.** A squeeze-and-excitation block that learns a
  per-band importance vector and recalibrates the five frequency bands. An early version
  (Attempt 45) failed in a telling way — its global-average "squeeze" was itself corrupted
  by the single-channel artifact spikes, so the noise was literally shouting down the
  summary statistic the block relied on. It only worked *after* AGLI was muting those
  channels upstream. Order mattered: silence first, then equalise. (The specific adaptation
  for EEG frequency bands here is my own modification, cited in the thesis references.)
- **The Subject Bias.** A small learned per-subject term. The reasoning is almost
  paradoxical: by giving the model an *explicit, cheap* channel for "who is this person,"
  you stop it from smuggling identity into the emotional features. Hand it the confound on
  a plate and it stops stealing it. The same instinct, generalised, is how clinical models
  are made to condition on site instead of laundering it through the signal.

Across this stretch of attempts — the assembled configuration consolidating around
**Attempt 59**: one-second windows, subject- and session-specific normalisation, AGLI +
SE-Block, 10 mean/variance features — the wall that had held since Phase 1 finally broke,
and runs climbed into the low 80s. **Attempt 60** folded in Focal Loss (to stop the easy
Positive class from dominating the gradient) and the adaptive Subject Bias, and confirmed
it held. After six weeks, the ceiling was gone —
and every brick that broke it traced to a specific failure I could name.

**What this phase taught me.** Architecture *can* beat the confound — but only by
attacking it directly: silence the unreliable inputs (AGLI), recalibrate what's left
(SE), and give identity its own honest outlet so it stops contaminating the signal
(Subject Bias). And watch your normalisation layers — they will cheerfully undo your best
ideas with a perfectly straight face.

---

## Phase 5 — Three Roads to the Same Place
**Jan 24 – Feb 22, 2026** · *Primary record: see the headline tables in [`README.md`](README.md) §4*

> *The question: does the win depend on one architecture, or is it the method that won?*

With a model that finally worked, the last job was to find out *why* — and to stop
trusting the easier within-subject numbers. I moved everything to the proper benchmark:
**Leave-One-Subject-Out (LOSO)** — train on 14 people, test on a 15th the model has never
seen, fifteen times over. This is the honest analogue of deploying to a new person (and,
by extension, a new hospital): no calibration, no peeking. Then I ran the *same* training
recipe through three different graph paradigms, to separate the method from the model.

- **GCN_DE** — static graph from physical electrode distance. The strongest result and
  the most *principled* one: **~82% LOSO**.
- **Adaptive DGCNN** — learns its own graph. The most *stable* across folds (lowest
  variance), but it needs high data density to keep its learned topology from
  hallucinating, and it never beat the static prior.
- **GraphSAGE** — inductive local neighbourhood sampling. The **worst cross-subject
  transfer** of the three (the largest train→test gap), precisely because local
  aggregation overfits to neighbourhood quirks that don't survive the jump to a new
  brain.

The spread is the punchline of the entire project. The *fixed physical graph won* — not
despite being the simplest, but *because* of it. A graph wired by anatomy is a strong,
unlearnable prior; it cannot be bent to fit one subject's idiosyncratic noise, so it
*regularises by refusing to overfit*. Every failure in this log, all eight weeks of them,
was ultimately the same failure — the model overfitting to the subject, the session, the
artifact, the confound — and the architecture that generalised best was the one with the
least freedom to do it. The full final tables (means, standard deviations, per-fold
breakdowns) live in the [README](README.md); the point here is the *shape* of the result,
not the digits.

---

## What I would do next (the honest open problems)

A research log that ends in triumph is lying. Three real limitations remain, and I'd rather
name them than let a careful reader find them:

- **Some subjects are genuinely unreachable.** Subjects 2 and 12 resisted every
  intervention — interpolation, robust scaling, dynamic graphs, attention. The forensic
  read is that the emotional signal isn't faint, it's *absent* (poor contact, disengaged
  alpha-blocking). Attention can decide *where* to look, but it cannot conjure a signal
  that was never recorded. Knowing when to stop engineering against missing data is itself
  a result.
- **The non-stationarity was managed, not solved.** The real fix for session/subject drift
  is explicit domain adaptation (DANN-style adversarial alignment, or test-time
  adaptation). The current model leans on a strong prior and honest normalisation to *blunt*
  the shift; it does not *close* it. That's the obvious next chapter.
- **Reproducibility is statistical, not bit-exact.** There is no fixed random seed anywhere
  in the training code (verified by search). Every headline number is a mean with run-to-run
  variance — which is why the lost 9% of "Attempt 18" stayed lost, and why I report spread
  instead of pretending to a precision I don't have.

---

## Why this generalises beyond EEG

Strip the neuroscience away and this is a study of one problem: **a model that keeps
learning the confound instead of the signal, and the discipline required to stop it.** The
session drift is temporal distribution shift. The cross-subject test is external
validation on an unseen source. The Subject Identity Trap — a model scoring well by
silently encoding *which subject* rather than *what state* — is the exact failure mode that
makes a clinical model trained at one hospital collapse at the next, where it has quietly
learned the site instead of the disease. The countermeasures here have direct analogues:
AGLI is per-source channel reliability weighting (sensor harmonisation); the Subject Bias
is explicit site-conditioning so the confound stops contaminating the features; LOSO is
leave-one-site-out validation. The dataset is small and the domain is niche, but the habit
of mind — *suspect every good number until you've proven it isn't the confound* — is the
part that transfers. The [README](README.md) draws this clinical mapping out in full.

---

*Eight weeks. Three jumps. One enemy, wearing a different mask each phase. The model that
won was the one I gave the least room to cheat.*
