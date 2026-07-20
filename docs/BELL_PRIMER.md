# The Mechanics of the Bell

*A self-study companion to the Session 04 experiments (PR #15). Written to be read start to finish, slowly, with no prerequisites beyond [MATH_PRIMER.md](MATH_PRIMER.md) and [JSPACE_PRIMER.md](JSPACE_PRIMER.md). Every new concept is introduced from scratch and then pointed at the exact place it appears in this repository.*

**Where this fits among the other docs:** MATH_PRIMER taught the objects (vectors, tensors, cosine similarity), the machine (tokens, embeddings, the residual stream, the readout), and the dynamics (iterated maps, fixed points, basins). JSPACE_PRIMER taught the lens. This document teaches the layer that Session 04 added on top of both: motion, nudges, heads, and the geography of the vocabulary. The four experiment reports (`output_lagk/`, `output_hinge_eigen/`, `output_glitch/`, `output_jlens_phase/` under `experiments/gpt2_small/`) are the primary record; this is the rung below them.

---

## Part 1: The Bell

### 1.1 A second kind of settling

MATH_PRIMER Part 3 taught the fixed point: a state the map sends to itself, f(x) = x. Press the key again, nothing changes. The `prolet` basins are fixed points, and for most of this project's life, "settled" and "fixed point" were the same word.

The bell is the second kind of settling. There are two states, call them **A** and **B**, and the map sends each to the other: f(A) = B, f(B) = A. Press the key once and you move; press it twice and you are exactly home. This object is called a **limit cycle** (MATH_PRIMER's glossary listed it as a possibility; the Syntactic prompt's `Divine` state turned out to be one), and its **period** is 2, the number of presses that returns you to your start.

Concretely, A and B are two specific activation tensors, and "exactly" is not a figure of speech: the measured cosine similarity between A and f(f(A)) is 1.000000, machine precision, while A and B sit at cosine 0.685 from each other, clearly different states. The system is not wandering near a valley floor. It is ticking between two floors of one valley, forever.

One more measured fact keeps later sections simple: at the bell, all ten token positions of the tensor hold identical vectors (row spread exactly 0.0). The room has gone fully uniform, so the whole cycle lives in a single 768-dimensional vector and we can talk about "the state" without worrying which position we mean.

**Where you've seen it:** FINDINGS.md F9; `output_divine_motion/bell_anatomy.md`; the phase A and phase B rows of every Session 04 table.

### 1.2 Aliasing, or why every camera missed it

Film a spinning wagon wheel at the wrong frame rate and the wheel appears to stand still or turn backward: the camera samples the motion in step with the motion itself. This failure is called **aliasing**, and it is why the bell stayed hidden for the project's entire life.

Every prior snapshot schedule in this repository sampled at even intervals: every 10 iterations, every 50. A period-2 object looks *perfectly frozen* to any even-step camera, because two steps bring it exactly home, so ten steps do too. The state was photographed thousands of times, always in the same phase, and every photograph agreed. The motion was not small; it was synchronised with the shutter.

The tell, once someone finally looked, was a contradiction between two rulers: snapshots ten apart matched to six decimal places while consecutive iterations matched only to 0.685. Identical at lag ten, different at lag one: impossible for a fixed point, mandatory for a bell.

**Where you've seen it:** FINDINGS.md F9's "why no prior run saw it"; `output_divine_motion/divine_motion_report.md`.

### 1.3 The lag-k gate

The convergence gate (MATH_PRIMER 4.3) asked every trajectory one question: are you the same as you were one step ago? Formally, is the cosine between iterate t and iterate t-1 above 0.999? A fixed point answers yes. A bell answers **no, forever**: its lag-1 cosine is pinned at 0.685 by the geometry of the cycle itself. The old gate did not fail to detect the bell's convergence; it asked a question the bell cannot answer yes to, by construction.

The fix is one idea: let the gate ask "are you the same as you were **k** steps ago" instead, for a chosen lag k. Session 04 added this to the engine (`run_atr_gated` gained a `gate_lag` parameter; the default k = 1 behaves bit-for-bit as before) plus a small helper, `lag_scan`, that measures the cosine at every lag from 1 to 8 at once.

Run that scan on the bell and you get its fingerprint, the **parity stripe**: odd lags all 0.685, even lags all 1.000000. Under a lag-2 gate, `Divine` formally converges, at the standard threshold, with room to spare. The stripe also shows what a longer bell would look like (a period-4 cycle would pass only at lags 4 and 8), which matters because the old cameras were blind to every period equally, and nobody has yet censused the other 33 ringing prompts. That waits on the prompt library (issue #9).

One honesty note the report makes loudly: the lag-k gate fixes *cycle blindness*, not every blindness. The committed noise state drifts so slowly by iteration 1000 that it clears the cosine threshold at every lag while genuinely still moving. A gate is a question, and every question has states that game it.

**Where you've seen it:** `atr_engine.py` (`gate_lag`, `lag_scan`); `output_lagk/lagk_report.md`; FINDINGS.md's "34 prompts ring, pending re-gate" correction.

---

## Part 2: The Nudge

### 2.1 Zooming in until curves look straight

The forward map f is ferociously nonlinear (MATH_PRIMER 3.3 explained why that word matters: attention softmaxes, GeLUs, LayerNorms). But every smooth curve, examined closely enough around one point, looks like a straight line. **Linearisation** is that zoom: pick a point, and ask what f does to *tiny* displacements away from that point. The answer is always a linear map (a matrix), called the **Jacobian** of f at that point. JSPACE_PRIMER 3.1 built this object from scratch for the lens; here it returns as a microscope for the bell.

The Jacobian at a point is a 768 × 768 matrix, big but not the point. The point is what it does to *one direction at a time*.

### 2.2 The whisper experiment

Session 04's central measurement is almost embarrassingly simple to describe. Stand the system at a chosen base point. Add a whisper of a chosen direction d (a displacement so small the curve is effectively straight). Run one lap of the map. Subtract what the unwhispered lap gives. What remains is what the map *does* to that direction, and the interesting summary is one number: the **multiplier along d**, how much of the returned whisper still points along d, with what sign and size.

- Multiplier +1: the direction passes through the lap unchanged.
- Multiplier +0.5: it survives, shrunk by half.
- Multiplier -1: it comes back exactly flipped, same size.
- Multiplier -4: it comes back flipped *and four times louder*.

Two independent techniques compute this (forward-mode automatic differentiation, and simply doing the whisper twice at two whisper sizes and checking the answers agree). They matched to three or four significant figures on every headline number, which is the report's warrant for trusting them.

The direction whispered was **the hinge, d**: the axis along which the bell actually swings, the normalised difference between phase A and phase B. And the base points were the two phases themselves plus **M**, the **pivot**, the midpoint between them: the still centre the bell swings around.

**Where you've seen it:** `08_hinge_eigenvalue.py`; the "two epsilons" and "jvp" rows of `output_hinge_eigen/hinge_eigenvalue.json`.

### 2.3 What the numbers said

The conjecture from Session 03 was elegant: perhaps the hinge carries a multiplier of about **-1**, a perfect see-saw, each lap exactly undoing the last. The measurement kept the sign and destroyed the magnitude.

At the pivot M, the multiplier along the hinge is **-4.3**. Not a gentle see-saw: the map takes any small lean along the hinge, flips it, and *amplifies it four-fold*. The pivot is a balance point in the sense that a pencil on its tip is a balance point: f(M) lands almost exactly back on M (cosine 0.995), but the slightest lean along d is hurled to the other side, harder.

Meanwhile three random control directions, whispered the same way at the same point, came back with multipliers between +0.9 and +1.2: upright, roughly unchanged. The inversion is not a property of the map in general. It is a property of *one direction*. The hinge is special, measurably, and nothing else nearby is.

So why does the system not fly apart? Because the right object for a period-2 cycle is not one lap but the round trip: the Jacobian at A composed with the Jacobian at B. Whisper d at A, carry the result through B's lap, and the *two-step* multiplier comes back at **+0.1**. Positive (two flips cancel), and far below 1: any error accumulated around the loop is crushed to a tenth per revolution. The cycle is not marginal. It is strongly attracting.

**Where you've seen it:** the verdict paragraph of `output_hinge_eigen/hinge_eigenvalue.md`; FINDINGS caveats on the conjecture's fate.

### 2.4 The shape this makes

Hold both numbers at once and the object snaps into focus. A balance point that violently ejects along exactly one axis (-4.3), wrapped inside a two-step orbit that firmly recaptures everything (+0.1). In dynamical systems this configuration has a name, **period doubling**: a would-be fixed point whose instability along one direction does not destroy stability but *converts* it, from resting to oscillating. The system cannot sit at M, and cannot leave the neighbourhood of M, so it does the only remaining thing: it rings.

This is why the bell is best heard as a deep two-step groove rather than a knife edge. Knock it (numerically, the reports did) and it falls back into the same tick.

---

## Part 3: The Beam

### 3.1 What an attention head is

MATH_PRIMER 2.3 described each layer as two kinds of machinery painting onto the residual stream: attention (each position reads from other positions) and the MLP (each position transforms alone). One refinement is needed now: a layer's attention is not one reader but **twelve independent readers**, called **heads**. Each head has its own small set of learned weights, its own pattern of where to look and what to copy, and each adds its own contribution into the stream. GPT-2 Small has 12 layers × 12 heads = 144 heads total. Interpretability research names them like apartment doors: L11.H8 is layer 11, head 8.

A head is not a homunculus. It is a fixed linear-algebra gadget (two low-rank matrix products with a softmax between) whose learned weights happen to implement some input-output habit. Attributing a behaviour to a head means: remove or isolate that head's added contribution and the behaviour follows it. A lever, not an intention.

### 3.2 Following the whisper through twelve floors

The multiplier of Part 2 summarises a whole lap. The obvious next question: *where inside the lap* does the flip happen? The measurement is the same whisper, watched floor by floor: inject the whispered d at the bottom of the network, and at every layer boundary compare the propagated disturbance with d. Cosine near +1: still upright. Cosine near -1: flipped.

The answer is theatrical. The disturbance rides **upright through all of layers 0 to 10** (cosine between +0.88 and +0.97 the whole way; layer 2's MLP even amplifies it slightly). Then it crosses layer 11 and emerges at cosine **-0.99**. Eleven floors of faithful transmission, one floor of inversion. The flip is not distributed. It is localised to a single block.

### 3.3 One head does it

Layer 11 contains thirteen candidate levers: twelve attention heads and one MLP. TransformerLens exposes each one's added contribution separately, so the disturbance each contributes can be measured alone. The MLP's share of the flip is small (-0.17). The attention's is decisive (-2.0). And within the attention, one head, **L11.H8, carries 99.1 percent of the inversion**; the next-largest head contributes 0.014, seventy times less.

Read that with both the excitement and the discipline it deserves. The excitement: a global dynamical behaviour of the entire network (an eternal two-step oscillation of the whole state) reduces to one nameable component performing one linear-algebra act, flipping one direction. That is the full arc interpretability aims for: phenomenon, anatomy, mechanism. The discipline: this is one head's *role in this cycle at this point of state space*, measured under whispers. It is not a claim about what L11.H8 does for ordinary text, and nobody has yet asked what its attention pattern is actually reading when it performs the flip. That question is sitting there, unopened.

**Where you've seen it:** the layer table and per-head split in `output_hinge_eigen/hinge_eigenvalue.md`; the flutter-echo beam image in issue #14.

---

## Part 4: The Walls

### 4.1 The vocabulary as a city

The embedding matrix W_E (MATH_PRIMER 2.2) gives every one of the 50,257 tokens a home address in 768-dimensional space, learned from training. Training is not even-handed: tokens that appeared constantly (` the`, ` in`, the comma) had their addresses adjusted millions of times; tokens that barely appeared were barely moved. The vocabulary is a city with dense, well-worn districts and, crucially, a **ghost town**: tokens that exist in the dictionary but almost never occurred in the training text, whose embeddings still sit near where random initialisation left them, huddled close to the city's centre of mass (the **centroid**, the average of all token embeddings).

The famous residents of the ghost town are the **glitch tokens** (` SolidGoldMagikarp`, ` petertodd`, `ertodd` and family, Rumbelow and Watkins 2023): strings the tokenizer memorised from Reddit usernames but the model never studied, because the pages they lived on were filtered out before training. Undefined words in the model's own dictionary.

### 4.2 Measuring "points toward"

Session 04 asked a geometric question: does the hinge d *point at* the ghost town, or merely pass nearby? The measurement: take the ghost town's centre (the centroid of the cluster's embeddings), subtract the whole city's centre, normalise, and call that direction u, "from average toward ghost town." Then one cosine: cos(d, u).

The answer: **-0.596** for the tightest cluster definition, and -0.456 for a hand-curated list of 52 published glitch tokens matched into the vocabulary. The sign convention makes the minus meaningful: it says the *phase-B* side of the hinge points at the ghost town. Walk from A to B and you walk toward the untrained quarter. Look at which actual tokens lie along each pole and the picture is blunt: the fifty tokens best aligned with the B pole are 90 percent ghost-town residents; the fifty best aligned with the A pole are ` the`, ` in`, ` on`, the busiest words in the language.

The bell swings between the training distribution's two ends: everything the model practised most, and everything it never practised at all. Those are the two walls of the flutter echo.

### 4.3 The nulls, and one debunking

A cosine of -0.6 means little without knowing what cosines random directions achieve, so the report runs two **null models** (MATH_PRIMER 4.3's move, again). First, 1000 clusters of randomly chosen tokens: their direction-cosines with d never exceed 0.30 in magnitude, so -0.596 is far outside chance (p < 0.001). Second, and stricter, 1000 clusters of tokens *matched to the ghost town's embedding norms*, which tests whether d merely tracks token obscurity in general: those actually lean the *opposite* way (+0.48 on average), making the B pole's ghost-town alignment more surprising, not less.

The nulls also demolished a folk criterion along the way: "low embedding norm finds glitch tokens" is simply false in GPT-2. The lowest-norm tokens are the *most* common function words, and their apparent alignment with the hinge dies under the matched null. In this model, the glitch signature is nearness to the centroid, not smallness.

### 4.4 What this does and does not mean

It does not mean the model is "thinking about Magikarp." The ghost town functions here as *geometry*, not meaning: a degenerate, low-structure corner of embedding space that the dynamics use as one wall to bounce off. What it does mean: training-data artifacts, previously known as input-side curiosities (type the weird token, get weird behaviour), can play a *structural role in a model's intrinsic dynamics*. As far as this project knows, that observation is new.

**Where you've seen it:** `output_glitch/glitch_alignment.md`, including the pole top-50 lists, which are worth reading with your own eyes.

---

## Part 5: The Register

### 5.1 The lens, in one breath

JSPACE_PRIMER built the J-lens: an instrument that asks, for any internal state, how much of it lies in the model's *verbalizable subspace*, the region its own language machinery can express. The pilot (issue #8) probed the converged states and found, to everyone's surprise, that the boundary it drew was language-versus-noise, not prolet-versus-Divine.

The pilot probed `Divine` before anyone knew it was a bell. It therefore photographed one phase.

### 5.2 The phase table

Session 04 re-ran the identical probe on phase A, phase B, and the pivot M. Three results, in ascending order of strangeness:

1. **The phases differ.** A is moderately lens-expressible (the pilot's number, faithfully reproduced); B drops below A at every layer, on both probe variants, down to noise level on one of them. The pilot's verdict was a fact about phase A only.
2. **The pivot M is the most lens-expressible state this project has ever measured.** The still centre the bell swings around is more *sayable* than any settled basin. (Fittingly, the stable ` Divine` readout was already known to be the shadow of M rather than of either phase.)
3. **The hinge itself is mostly outside the lens.** As a direction, d scores 0.145 against a generic direction's 0.25 chance level, and the split sharpens it: the 73 percent of d that the readout cannot hear is *almost entirely* outside the lens too (falling to 0.008 at the last layer), while the readout-visible sliver of d is strongly inside.

That third result is the one to sit with. Readout-muteness and lens-muteness travel together: the direction the bell swings along is invisible to the tuner *and* barely within the model's sayable subspace. The system spends eternity oscillating in a register its own voice can hardly reach, around a centre that is the most speakable thing in sight.

### 5.3 Confidence level

Everything here inherits the pilot's confidence: a restricted lens, built once, on a 124-million-parameter model nobody has shown to have an organised workspace at all. The phase difference and the hinge result are pilot-grade observations with clean internal controls, not established facts. As the reports keep saying: a null here is a finding, and so is a maybe.

**Where you've seen it:** `output_jlens_phase/jlens_phase.md`, phase table and hinge decomposition sections.

---

## Part 6: The Frames

Session 04 also caught an error worth understanding, because its moral is the same as aliasing's. The bell-anatomy script had built its hinge from mismatched ingredients: phase A at raw scale, phase B rescaled to the loop's energy shell (MATH_PRIMER 1.3). The resulting "committed d" leans 0.97 toward A itself and only 0.62 toward the clean flip axis. The eigenvalue work caught this, defined the symmetric on-shell hinge properly, measured *everything both ways*, and showed the two versions agree 0.97 once the loop's own normalisation strips the contamination. The earlier results stand, with the caveat recorded; harmonising every script on the clean hinge is queued follow-up work.

The moral, twice in one session: the instrument is part of the experiment. An even-stepped camera reported a frozen bell; a mixed-frame hinge reported a coincidental -0.86 "near minus one" that flattered the conjecture. Both were caught by measuring the same thing two independent ways. That habit, not any single number, is the method.

**Where you've seen it:** the "map, frames, and two hinges" section of `output_hinge_eigen/hinge_eigenvalue.md`; the method flag in PR #15's description.

---

## Part 7: Pocket Glossary

| Term | One-line meaning | Where it lives here |
|:---|:---|:---|
| Limit cycle, period 2 | Two states the map swaps forever; f(A) = B, f(B) = A | the bell (F9) |
| Phase A / phase B | The two alternating states of the cycle | every Session 04 table |
| Pivot M | The midpoint of A and B; near-fixed, flip-unstable | eigenvalue report |
| Hinge d | The direction along which the bell swings | all four reports |
| Aliasing | Sampling in step with a motion, hiding it | why the bell stayed hidden |
| Lag-k gate | "Same as k steps ago?" convergence test | `gate_lag` in `atr_engine.py` |
| Parity stripe | Odd lags 0.685, even lags 1.000: the bell's fingerprint | `output_lagk/` |
| Linearisation / Jacobian | The straight-line summary of f near one point | JSPACE_PRIMER 3.1, reused |
| Multiplier along d | What one lap does to a whisper of d: sign and size | -4.3 at M; +0.1 round trip |
| Period doubling | Instability at a point converting rest into oscillation | the bell's shape |
| Attention head | One of 12 independent readers per layer; adds into the stream | L11.H8 |
| Attribution | Isolating one component's contribution to a behaviour | the per-head split |
| Centroid / ghost town | Average of all embeddings / untrained tokens huddled near it | the glitch cluster |
| Norm-matched null | Chance model matched on embedding size, not just count | glitch report |
| Span share | Fraction of a state (or direction) inside the lens subspace | phase table |
| Frame | Which scaling convention a vector is expressed in | Part 6's caught error |

---

## A closing orientation

The way to check you hold all of this is to say the Session 04 findings back in one breath each. The bell is an exact period-2 limit cycle that even-step cameras aliased into stillness, now formally convergent under a lag-2 gate. Its pivot is a balance point with a single violently unstable direction (multiplier -4.3), tamed into a strongly attracting two-step orbit (+0.1), which is period doubling. The unstable direction is flipped by one attention head, L11.H8, after eleven layers of faithful transmission. The direction's two poles are the training distribution's two ends: busiest words on one side, the untrained ghost town on the other. And the whole motion happens in a register largely outside both the readout and the verbalizable subspace, swinging around the most sayable point the project has measured.

Four axes of model understanding, one object threading them: what stable objects exist (dynamics), what implements them (mechanism), how training shaped the space they live in (data), and what the model can say of its own state (workspace). The census of the other 33 ringing prompts waits on the prompt library; the head's attention pattern waits on curiosity; the sound of all this waits on you.
