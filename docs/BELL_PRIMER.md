# The Mechanics of the Bell

*A plain-language companion to the Session 04 experiments (PR #15). Written to be read start to finish. Assumes [MATH_PRIMER.md](MATH_PRIMER.md) (vectors, cosine similarity, the residual stream, the readout, iterated maps) and [JSPACE_PRIMER.md](JSPACE_PRIMER.md) (the J-lens). Each section states what was done, what was measured, what the numbers were, and what follows from them. Parts 1 to 7 are the measurements; Part 8 says what the numbers mean about the model as a trained object; Part 9 states the implications, sorted by how firmly they are held.*

**Terminology note:** the direction the cycle moves along is called **the flip axis, d** throughout this document. Earlier session files called it "the hinge"; script names, output folder names, and JSON keys (`08_hinge_eigenvalue.py`, `output_hinge_eigen/`, `d_hinge`) keep the old word as frozen labels.

**The experiment reports this document summarises:** `output_lagk/lagk_report.md`, `output_hinge_eigen/hinge_eigenvalue.md`, `output_glitch/glitch_alignment.md`, `output_jlens_phase/jlens_phase.md`, `output_suppression/suppression_report.md`, all under `experiments/gpt2_small/`. They are the primary record; where this document and a report differ, the report governs.

---

## Part 1: The cycle, and why it was missed

### 1.1 What Divine is

"The bell" is the project's name for the following measured fact. Take the Syntactic prompt's state at iteration 1000 and call it A. Apply the map once (one forward pass plus the energy rescale): the result, B, is a different state. Apply the map to B: the result is A again, exactly.

- cos(A, B) = 0.685. A and B are clearly different states.
- cos(A, f(f(A))) = 1.000000. Two applications return the start, to machine precision.

So the trajectory is A, B, A, B, forever. This is called a **limit cycle of period 2**: period 2 because two applications of the map return you to where you started. A and B are called **phase A** and **phase B**. It is a different kind of stable object from a fixed point (where one application returns the start).

One measured simplification used throughout: at the bell, all 10 token positions of the tensor hold identical vectors (the row spread is exactly 0.0). So the whole cycle is described by a single 768-dimensional vector per phase, and "the state" needs no position qualifier.

**Where recorded:** FINDINGS.md F9; `output_divine_motion/bell_anatomy.md`.

### 1.2 Reading cos(A,B) = 0.685 exactly

The 0.685 is not a vague "fairly similar". It has an exact reading, in four steps.

First, what the number computes: A and B are each a list of 768 numbers; multiply matching entries, add everything up, divide by the two lengths. The result measures direction agreement only: +1 proportional, 0 unrelated, -1 opposite.

Second, the split. For any two vectors, define M = the average of A and B (**the pivot**) and d = half their difference. Then A = M + d and B = M - d as pure arithmetic, true of any pair. Three measured facts make this bookkeeping meaningful here: the same d returns every cycle, to machine precision, for a thousand iterations (that is what period 2 means); d is perpendicular to M in the equal-scale frame, so the split is clean; and the map singles d out (Part 2). Any two states trivially "differ along a direction". A difference direction that is fixed, that reverses sign each step, and that the network treats unlike any other direction is not trivial.

Third, the quantity. With the split clean and both states the same length, the cosine equals the fraction of the state's energy in the shared part minus the fraction in the flipped part. Solving for 0.685: **about 84 percent of each state's energy is the shared part M, about 16 percent is the flipped part d** (84.2 minus 15.8 gives back 0.685). One sixth of the state's energy oscillates along a single fixed axis; five sixths stand still.

Fourth, what "one direction out of 768" means. A direction is an axis through the 768-dimensional space, and almost never one of the 768 slots in the list: d is a specific weighted combination of all 768 entries. Choose a coordinate system whose first axis is d, and the step from A to B changes exactly one coordinate (its sign) and leaves the other 767 untouched. The linearisation in Part 2 confirms the network really is close to the identity along other directions near these states, so "reproduces 767 numbers, reverses the 768th" is a description of the map there, not just of the pair.

### 1.3 Why it went undetected

Every earlier run saved the state at even intervals: every 10 iterations, or every 50. Ten applications of the map is five full cycles, which returns the state exactly to the phase it was in. Therefore every saved snapshot was the same phase, the saved sequence was constant, and the state appeared frozen. The general name for this failure is **aliasing**: sampling a repeating process at an interval that is a multiple of its period, so the repetition is invisible in the samples.

The detection came from a contradiction between two measurements: snapshots 10 iterations apart matched to six decimal places, while consecutive iterations matched only at 0.685. Both cannot be true of a fixed point. Both are necessarily true of a period-2 cycle.

### 1.4 The convergence-test fix

The old convergence test compared each iteration to the previous one and declared convergence when the cosine stayed above 0.999. For the bell, that comparison returns 0.685 every time, so the test could never pass, regardless of how long the run continued. This is arithmetic, not a tuning problem.

The fix, added to `atr_engine.py` in Session 04: the comparison interval is now a parameter, `gate_lag`. With `gate_lag = 2`, the test compares each iteration to the one two steps back. Under that test, Divine passes at the standard 0.999 threshold. The default remains `gate_lag = 1`, and with the default the engine's behaviour is unchanged (verified bit-identical against the pre-change code on a real run).

A helper, `lag_scan`, measures the cosine at every comparison interval from 1 to 8 at once. For the bell the result is: odd intervals all 0.685, even intervals all 1.000000. This pattern is the direct signature of period 2, and the same table would expose a period-4 cycle: its exact 1.000000 return would appear only at intervals 4 and 8. (Intermediate intervals could still score above the threshold if that cycle's states happened to lie close together; the exact return, not merely a high score, is the signature.) No one has yet run this scan on the other 33 non-converging prompts; that requires the prompt library (issue #9).

Two limits of the fix, stated in the report: it detects cycles, not slow drift (the committed noise state moves slowly enough by iteration 1000 to pass the cosine threshold at every interval while genuinely still moving); and the full re-classification of the 125-prompt sweep has not been run.

**Where recorded:** `atr_engine.py` (`gate_lag`, `lag_scan`); `output_lagk/lagk_report.md`.

---

## Part 2: The derivative measurement

### 2.1 The quantity being measured

Define two objects from the cycle:

- **The flip axis, d**: the normalised difference between the phases, d = (A - B) / ‖A - B‖. This is the direction along which the two phases differ, which is the direction the state moves along on every step.
- **The pivot, M**: the midpoint, M = (A + B) / 2.

The question: what does one application of the map do to a small displacement along d? Formally this is a directional derivative. Practically it was measured two independent ways: automatic differentiation (`torch.func.jvp`), and directly, by adding a small multiple of d to the base point, running one iteration, subtracting the undisplaced result, and dividing by the size of the displacement. The two methods agreed to 3 or 4 significant figures on every reported number, and the direct method was repeated at two displacement sizes to confirm the answer did not depend on the size.

The summary number is the **multiplier along d**: the component of the returned displacement that lies along d, with sign. Multiplier +1 means the displacement passes through one iteration unchanged. Multiplier -1 means it returns with the same size, pointing the opposite way. Multiplier -4 means it returns pointing the opposite way, four times larger.

### 2.2 The results

Measured at the pivot M:

- Along d: multiplier **-4.3**. A small displacement along the flip axis returns inverted and 4.3 times larger.
- Along three random control directions: multipliers **+0.9 to +1.2**. Ordinary directions pass through roughly unchanged.
- M itself maps almost to itself: cos(f(M), M) = 0.995.

So M is nearly a fixed point, but unstable along d: any component along d grows by a factor of about 4 per step, flipping sign each time. The map treats d differently from every other direction tested. (Tested means d plus three random probes; the full set of 768 independent directions has not been examined, so "one unstable direction" is the simplest reading consistent with these probes, not a proven count.)

Measured around the full two-step cycle (the derivative at A composed with the derivative at B):

- Along d: net multiplier **+0.1** (the component of the returned displacement along d: a projected multiplier, since the return is only partly aligned with the axis). Positive, because two inversions cancel; and much smaller than 1, meaning any deviation from the cycle shrinks by roughly 90 percent every two steps.

These two numbers together explain the observed behaviour. The system cannot rest at M (deviations along d grow). Deviations from the two-step cycle shrink along every direction tested. What is observed, and what these numbers make stable, is the alternation itself. The Session 03 conjecture had predicted a multiplier near -1 at the pivot; the sign was right, the size was not. A multiplier of -1 would be a marginal, borderline case. The measured -4.3 with a two-step contraction of +0.1 is a strongly stable oscillation. The technical name for this structure (a near-fixed point whose single unstable direction produces a stable period-2 cycle around it) is a **period-doubling** configuration.

**Where recorded:** `output_hinge_eigen/hinge_eigenvalue.md`, results 1 and 2; the numbers file `hinge_eigenvalue.json`.

---

## Part 3: Locating the inversion

### 3.1 Attention heads, defined

Each of GPT-2 Small's 12 layers contains an attention block and an MLP block, and both add their outputs into the residual stream (MATH_PRIMER 2.3). One refinement is needed here: an attention block is not one unit. It is 12 separate **heads**, each with its own learned weights, each computing its own output, all added into the stream. GPT-2 Small therefore has 144 heads. The naming convention is layer then head: L11.H8 is layer 11, head 8.

### 3.2 The measurement

Add a small multiple of d to the state at the point where the loop re-injects it (the input of layer 0). Run one forward pass. At each layer boundary, subtract the undisplaced run from the displaced run to get the propagated displacement, and measure its cosine with d.

Results, per layer boundary:

- After layers 0 through 10: cosine **+0.88 to +0.97**. The displacement is transmitted essentially unchanged in direction through eleven layers.
- After layer 11: cosine **-0.99**. The inversion happens inside layer 11 and nowhere else.

Splitting layer 11's two blocks (their added outputs can be measured separately): the MLP's contribution to the flip is -0.17; the attention block's is -2.0. Splitting the attention block's 12 heads: **head 8 contributes 99.1 percent of the inversion**. The next largest head contributes 0.014.

### 3.3 What this attribution means, and what it does not

It means: in this region of state space, for this direction, the sign inversion that sustains the cycle is performed by one identifiable component, L11.H8. Remove or isolate that component's output and the flip follows it. This is the standard form of a mechanistic attribution: a global behaviour of the network traced to a specific part.

It does not mean: that this head "causes Divine" in general, or that it behaves this way on ordinary text. Part 7 tests exactly those questions.

**Where recorded:** `output_hinge_eigen/hinge_eigenvalue.md`, result 3 (the per-layer table and per-head split).

---

## Part 4: The embedding alignment

### 4.1 The token cluster in question

Every token has a 768-number embedding vector, a row of W_E (MATH_PRIMER 2.2). Average all 50,257 rows to get the mean embedding. Most tokens sit far from this mean, because training moved them. A small set of tokens sits unusually close to it: these are tokens that almost never occurred in the training text, so their vectors were barely updated from initialisation.

The known members of this set include the published **glitch tokens** (" SolidGoldMagikarp", " petertodd", "ertodd" and related; Rumbelow and Watkins 2023). The mechanism is documented: these strings occurred often enough in the corpus used to build the tokenizer (largely as Reddit usernames) to receive vocabulary entries, but the pages containing them were filtered out before the model's weight training, so the model never learned anything about them.

Session 04 defined the cluster two independent ways: geometrically (the 0.1 percent of the vocabulary closest to the mean embedding, 50 tokens) and by list (52 published glitch-token strings matched into the vocabulary). The two definitions agree with each other (their centroid directions have cosine +0.67).

### 4.2 The measurements

Let u be the normalised direction from the mean embedding toward the cluster's centroid. Measured:

- **cos(d, u) = -0.596** for the geometric cluster, **-0.456** for the published list. The sign convention: negative means the B side of d points toward the cluster.
- Chance comparison 1: the same cosine computed for 1000 clusters of randomly chosen tokens never exceeded 0.30 in magnitude. Zero exceedances in 1000 draws gives p ≤ 0.001 under the standard finite-sample bound.
- Chance comparison 2 (stricter): 1000 clusters of tokens chosen to match the glitch cluster's embedding norms. These lean the opposite way (mean +0.48), so the result is not explained by embedding norm. Norm is the available proxy for token frequency here; a null matched directly on corpus frequency was not run.
- Of the 50 vocabulary tokens whose embeddings best align with the B side of d, **45 are in the geometric cluster**. Of the 50 best aligned with the A side, most are the highest-frequency function words: " the", " in", " on", the comma.
- A side finding from the controls: low embedding norm does not identify glitch tokens in GPT-2. The lowest-norm tokens are the most frequent function words, and their apparent alignment with d disappears under the norm-matched comparison. In this model the glitch signature is proximity to the mean, not small norm.

### 4.3 What follows

Stated plainly: the direction along which Divine oscillates runs between the embedding region of the most-trained tokens (one end) and the region of never-trained tokens (the other end). This is a geometric statement about where the cycle sits in the model's representation space, not a statement that the model is processing or "referring to" those tokens. Its significance: glitch tokens were previously known only as anomalous inputs; here the untrained region of embedding space plays a role in the model's internal dynamics with no glitch token ever appearing in the input.

**Where recorded:** `output_glitch/glitch_alignment.md`, including the two top-50 token lists in full.

---

## Part 5: The lens measurements

### 5.1 The quantity

JSPACE_PRIMER describes the J-lens. For this document one number matters: for any state (or direction), the **span share** is the fraction of it (of its squared length) that can be expressed as a combination of the lens's fixed set of directions. Share 1.0 means fully expressible in that set; 0 means orthogonal to all of it. A random direction scores about 0.25 against this particular lens (193 directions in a 768-dimensional space, measured directly as a baseline). A second variant, the sparse share, restricts the combination to a few nonnegative terms; it behaves consistently and is reported alongside.

The pilot (issue #8) measured Divine before the cycle was known, so it measured one phase. Session 04 repeated the identical measurement on both phases and the midpoint. The replay of the pilot's own states reproduced its recorded numbers to 7 decimal places, which is the check that the instrument was reassembled correctly.

### 5.2 The results (last layer; the per-layer tables are in the report)

| state | span share |
|:---|---:|
| phase A | 0.173 |
| phase B | 0.123 |
| midpoint M | 0.180 |
| prolet (average) | 0.157 |
| noise (average) | 0.114 |

- B scores below A at every layer, on both variants, and near the noise level on the span variant. The pilot's finding ("Divine at least as expressible as prolet") was a fact about phase A only.
- M scores above everything else this project has measured. Separately, the bell-anatomy work had already shown that the stable ` Divine` readout token is produced by M's direction rather than either phase's.
- The flip axis as a direction, measured on both constructions. On the physical on-shell axis d_sym: span share 0.013 at the last layer against the 0.25 baseline (5 percent of chance; mean over layers 0.021). Decomposed by effect on output-token scores: the readout-quiet part of d_sym carries 97.0 percent of its energy and overlaps the lens almost nowhere (at most 8 percent of chance at any layer); the readout-visible part carries 1.2 percent of the energy, sits at the chance level in the earliest layers, and rises to 2.3 times chance by the last. These two subsets cover 98.2 percent of the axis energy; the remaining components are unassigned. The frame-mixed committed axis (as the earlier anatomy script built it) reads higher (0.145 at the last layer; its subsets cover 86 percent), because it is contaminated with the pivot's own direction. The physical axis sharpens the conclusion and does not change its direction.

### 5.3 What follows, at what confidence

The motion of the cycle is carried by a direction that is almost entirely outside both the output projection and the lens's set of directions (span 0.013 against 0.25 chance on the physical axis), while the midpoint the motion straddles is the most lens-expressible state measured. The two kinds of low visibility (to the output vocabulary, to the lens) coincide on the measured subsets of this direction at every layer; on the physical construction those subsets cover 98.2 percent of the axis energy.

Confidence: everything in this part inherits the pilot's limitations. The lens is a reduced version built once from limited data, on a 124-million-parameter model for which no one has demonstrated an organised workspace. These are internally consistent pilot-grade measurements, not established properties of the model.

**Where recorded:** `output_jlens_phase/jlens_phase.md`.

---

## Part 6: The frames error

Session 04 also caught an error worth understanding. The bell-anatomy script had built its flip axis from mismatched ingredients: phase A at raw scale, phase B rescaled to the loop's energy shell (MATH_PRIMER 1.3). The resulting "committed d" leans 0.97 toward A itself and only 0.62 toward the clean flip axis. The eigenvalue work caught this, defined the symmetric on-shell axis properly, measured everything both ways, and showed the two versions agree at 0.97 once the loop's own normalisation strips the contamination. The earlier results stand, with the caveat recorded; harmonising every script on the clean definition is queued follow-up work.

The reason to keep this section in a learning document: twice in one session, a measurement error produced a plausible result (a frozen state; a multiplier near -1 that flattered the conjecture). Both were caught by measuring the same quantity two independent ways. That habit, not any single number, is the method.

**Where recorded:** `output_hinge_eigen/hinge_eigenvalue.md`, "the map, the frames, and two hinges"; the method flag in PR #15's description.

---

## Part 7: The suppression tests

A hypothesis followed from Parts 2 and 3: perhaps L11.H8 is a **suppression head**, a documented behaviour class in GPT-2 Small (the best-studied example is L10.H7, which detects the token the network is about to promote and writes the negative of it, damping over-confidence), and the closed loop turns that one-shot negative correction into a sustained oscillation. Three tests (experiment 11):

**Test 1, the weights.** Each head contains two matrices whose product is a fixed 768-to-768 transform (the OV circuit). Because the bell state is position-uniform, the head's in-loop output is exactly this transform applied to the (scaled) input, independent of the attention pattern (verified to relative error 1.4e-07). Along the flip axis, L11.H8's transform returns cos -0.96 at gain 63.7: rank 1 of all 144 heads on both measures, with the runner-up 50 times weaker, and no comparable response on random directions. The inversion is physically located in this one head's weight matrices.

**Test 2, causality.** Re-running the loop from phase A with head 8's output zeroed kills the cycle within about 10 iterations; by iteration 300 every lag from 1 to 8 reads 1.000000. The state settles at a genuinely new fixed point (cos to A +0.14, to B -0.61, to M -0.25) whose readout is " the" at probability 0.024, a nearly flat distribution over common words. Zeroing a different same-layer head instead (L11.H0) leaves a period-2 cycle running. The head is specifically load-bearing: no head 8, no bell.

**Test 3, the refutation.** On twelve ordinary sentences run normally through the model, L11.H8 shows the opposite of the suppression signature: it raises the score of the token it attends to at 91.4 percent of positions (mean effect +5.97). The known suppression head L10.H7, run through the identical protocol, shows the documented negative signature (87.1 percent negative; 100 percent negative where its attention exceeds 0.2), which validates the measurement. L11.H8 is, on ordinary text, a copy promoter, not a suppressor.

**Verdict:** the inversion is confirmed in the weights and causally necessary for the cycle, but it is not a trained suppression behaviour. On ordinary text the head does something else entirely. The reading this supports: the inversion is a structural side effect of training, not a function, and the loop is what makes it matter. One recorded limit: only suppression at the token-readout level was measured; a suppression role in some other basis is not excluded.

**Where recorded:** `output_suppression/suppression_report.md`.

---

## Part 8: What the numbers say about the model

The model is a fixed block of 124 million learned numbers, arranged as 12 layers of transforms acting on a 768-dimensional state, with two dictionary matrices connecting that state to the vocabulary. Training set every number by nudging it to improve next-token prediction on web text; nobody chose any individual value. Each headline number above is a property of that finished block.

**cos(A,B) = 0.685** says the cycle is geometrically minimal: one sixth of the state's energy oscillating along one axis, five sixths standing still (Part 1.2). The network reproduces essentially the whole state and reverses one direction.

**-4.3 and +0.1** say the trained weights contain a strong reversal along one specific axis, and that the surrounding saturating parts absorb its overshoot. Training never penalised the reversal, because in training the network never receives its own output as input, so a reversal is never applied twice in a row and has no consequence. These are permanent properties of the weights that single-pass use never exercises; the loop compounds them until they dominate. The deep stability (+0.1 round trip) is why the cycle reproduced exactly across different machines.

**Gain 63.7, rank 1 of 144** says the inversion is not distributed knowledge: it is localised in one component's weight matrices, put there as a side effect of whatever that head was actually trained to do (Part 7: raising the scores of attended tokens). Large stretch factors inside heads are normal. The loop found the one whose largest axis reproduces itself.

**The poles (cos -0.596)** say the training data's frequency distribution is written into the embedding geometry, and the cycle runs along it. Constantly-seen tokens had their vectors shaped by millions of updates; never-seen tokens kept their initial vectors, which cluster near the mean. The flip axis connects the most-updated region to the never-updated region. The ablation result supports the frequency reading: remove the inverting head and the loop settles into a state reading " the" at p 0.024, the most-common-word region, apparently the space's default resting place. The head's reversal is what keeps the state from resting there.

**p = 0.505, with 73 percent of d in W_U's weakest directions,** says the output map's shape hides the motion. W_U maps 768 numbers to 50,257 scores; a rectangular map necessarily transmits some directions barely at all, including any direction that shifts all scores nearly equally, which the softmax cancels. The stable part of the cycle projects strongly onto one vocabulary column; the moving part projects almost nowhere. The motion is real in the state and nearly absent in the output because of the map's geometry, not because the motion is small.

**Lens share 0.013 against 0.25 chance** (on the physical axis; the frame-mixed construction reads 0.145) says the motion lies almost entirely outside the subspace the model uses for expressible content generally: closer to internal bookkeeping (frequency, scale) than to meaning.

In one sentence: training built very strong weights around predicting common words, left a never-trained region as a byproduct, and incidentally wrote a large sign-reversal into one late head; none of this matters in ordinary use, and the feedback loop settled into the one axis where all three facts intersect.

---

## Part 9: Implications, by scope

**For this project (firm, because measured).** Every "never converges" claim made with the lag-1 gate is provisional, including the cross-model verdicts: Pythia-410m's "no consolidation" may be aliased cycles, and the Act II table could change shape when re-gated. The landscape contains at least two kinds of stable object, so basin counting needs a type column. And the methods lesson is proven twice in one repository (aliasing; the frames error): measurement assumptions hid the most interesting object, and measuring one thing two independent ways is what caught both.

**For how models are understood (moderately firm, one instance).** Three points. First, models contain real structure no single forward pass can show: the reversal sat in the weights through every benchmark this model ever ran, and only compounding under iteration made it visible; iteration is a measurement instrument for weight properties that behaviour hides. Second, head labels are regime-bound: the same head is a copy promoter on ordinary text and the inverter that sustains the cycle in the loop; a catalogue that assigns each head one function is implicitly a catalogue of one regime. Third, some deep structure has no functional story: the tests came back "structural accident", and methods have to be able to accept "it is sediment, there is no reason" as a correct final answer.

**For the workspace question (pilot confidence).** The bell is an existence proof that a model can carry a persistent, structured, causally load-bearing internal state that is almost entirely outside both its output vocabulary and its expressible subspace. If that survives bigger models and better lenses, it matters for any approach that relies on models reporting their own states. Qualifier: the regime is artificial, and nobody has shown this arises during ordinary computation.

**For practice (the transferable point).** Closed loops around language models are increasingly normal (agents reading their own outputs; models trained on model-generated text). The specific bell does not transfer; the demonstrated principle does: single-pass evaluation does not bound closed-loop behaviour, because feedback amplifies exactly what single passes suppress. Also demonstrated: removing one of 144 heads completely changed the loop's terminal behaviour, so small structural edits can silently rewrite a system's dynamical properties.

**For the piece.** The founding question was what remains of a voice when the medium's own character takes over. The bell answers with a case the rooms never produced: what remains is not always a tone; sometimes it is a repetition, sustained by one component of the medium, running between the medium's most-used and never-used materials, inaudible in the medium's own voice. The sonification has a score.

**What none of this implies.** Nothing about intent, experience, or the model "doing" anything in ordinary use: the regime is artificial and the verdict was accident, not function. Nothing yet about the other 33 ringing prompts, longer periods, or other models: unmeasured. One model at 124 million parameters generalises to nothing until it is made to.

---

## Part 10: Glossary

| Term | Definition | Where |
|:---|:---|:---|
| Limit cycle, period 2 | Two states the map exchanges: f(A) = B, f(B) = A | F9 |
| Phase A / phase B | The two states of the cycle | all Session 04 tables |
| The bell | Project name for the Divine period-2 cycle | issue #14 |
| Pivot M | (A + B) / 2; maps nearly to itself; unstable along d only | Part 2 |
| Flip axis d | (A - B) normalised; the direction of the cycle's motion; called "the hinge" in earlier files | Parts 1, 2-5 |
| Aliasing | Sampling a repeating process at a multiple of its period, hiding it | Part 1.3 |
| gate_lag | Engine parameter: which earlier iterate the convergence test compares against | `atr_engine.py` |
| lag_scan | Helper reporting the comparison cosine at intervals 1 to 8 | `output_lagk/` |
| Multiplier along d | Signed factor applied to a small displacement along d by one iteration | -4.3 at M |
| Period doubling | A near-fixed point whose one unstable direction yields a stable 2-cycle | Part 2.2 |
| Attention head | One of 12 independent weighted units per attention block; 144 in the model | L11.H8 |
| OV circuit | The fixed 768-to-768 transform inside one head (two matrices multiplied) | Part 7 test 1 |
| Ablation | Removing one component's output to test whether a behaviour depends on it | Part 7 test 2 |
| Copy suppression / promotion | A head lowering / raising the score of the token it attends to | Part 7 test 3 |
| Attribution | Measuring each component's separate contribution to an effect | Part 3 |
| Glitch tokens | Vocabulary entries absent from weight training; embeddings near the mean | Part 4 |
| Norm-matched control | Chance comparison using token sets matched on embedding norm | Part 4.2 |
| Span share | Fraction of a state or direction expressible in the lens's directions | Part 5 |
| Frame | The scale convention a vector is expressed in; mixing frames was the caught error | Part 6 |

---

## Summary

Divine is an exact period-2 limit cycle: two states exchanged by the map, hidden from all earlier runs because even-interval snapshots sample a period-2 process in a single phase. Exactly measured, the oscillation is one sixth of the state's energy moving along one fixed axis (the flip axis d) while five sixths stand still. The convergence test now supports comparison at lag 2, under which Divine converges. One iteration multiplies a small displacement along d by -4.3 (inverted and amplified) while leaving other tested directions essentially unchanged; over two iterations the net factor is +0.1, so the cycle is strongly stable. The inversion is physically located in the weight matrices of one component, attention head 8 of layer 11: strongest inverter of all 144 heads along this axis, and causally necessary (zero its output and the loop settles at a new fixed point reading " the" at p 0.024). On ordinary text that head raises, not lowers, the scores of tokens it attends to, so the inversion is a structural side effect of training rather than a trained behaviour. The axis runs between the embedding regions of the most-trained and never-trained tokens, and it lies almost entirely outside both the output projection and the J-lens's expressible set (span 0.013 against 0.25 chance on the physical axis), while the cycle's midpoint is the most lens-expressible state the project has measured. One measurement error (a flip axis computed from mixed scales) was found and corrected without changing the conclusions. Open next steps: the lag scan on the remaining 33 non-converging prompts (blocked on issue #9), the promoted-versus-inverted structure of L11.H8's weight matrices, and the same measurements on other models.
