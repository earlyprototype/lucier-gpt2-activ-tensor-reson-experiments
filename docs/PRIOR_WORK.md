# Prior Work

This document places the ATR experiments (closed-loop reinjection of GPT-2 Small's final-layer residual stream into layer 0,
energy-renormalised, iterated to exhaustion) against the published record: what the nearest neighbouring work did and found, how it
relates to each ATR result, and what, on the evidence here, has no prior occupant. Citations that could not be checked against their
primary sources are marked with an asterisk and rest on secondary descriptions (title, venue, abstract-level claims); unmarked citations
were checked against primary material. Terminology follows BELL_PRIMER.md: the
bell (the exact period-2 limit cycle), phases A and B (its two states), the pivot M (their midpoint), the flip axis d (their normalised
difference), L11.H8 (layer 11, head 8).

The ATR findings the entries refer to, by number:

1. Five semantic basins from language inputs, distinct non-semantic basins from noise; the count is regime-dependent.
2. States become position-uniform early in the loop (all token positions identical).
3. One prompt yields an exact period-2 limit cycle, missed by lag-1 convergence tests and even-interval sampling.
4. The cycle's flip is inverted by a single head, L11.H8, through its OV circuit (the head's fixed 768-to-768 output transform): multiplier
   -4.3 at the pivot M, +0.1 around the two-step loop. On ordinary text the same head is a copy promoter.
5. The flip axis d runs between the most-trained and never-trained regions of embedding space (cos to the glitch-cluster direction -0.596).
6. The motion is near-invisible to the unembedding and to lens instruments; it lives mostly outside the verbalizable subspace.
7. Cross-model contrast: GPT-2 Medium one basin, Pythia-160m one, Pythia-410m none under a lag-1 gate now considered suspect.

## The nearest neighbours

**Wang, Li, Yan, Cheng, Zhang, Unveiling Attractor Cycles in Large Language Models: A Dynamical Systems View of Successive Paraphrasing (ACL
2025).**\* https://aclanthology.org/2025.acl-long.624/ and https://arxiv.org/abs/2502.15208
The established nearest neighbour. Iterated paraphrasing through
the model's text interface converges to stable periodic states, prominently period-2 cycles, robust to temperature, alternating prompts, and
alternating models; the authors built a dedicated periodicity metric because convergence metrics miss cycles, independently confirming the
diagnostic point of finding 3. The differences bound the overlap: their loop runs over discrete text, their cycles are approximate (each
paraphrase resembles the one two steps back), and they report no mechanism. The bell is activation-level, exact to machine precision, and
mechanistically attributed (findings 3, 4): convergent evidence at a different interface.

**Trained-loop architectures.** Geiping et al., Huginn (arXiv 2502.05171): a 3.5B model trained to iterate a middle block; some tokens fall
into stable orbits, the closest trained-model analogue of finding 3, but the model is trained for the loop, receives fresh input each step,
and the orbits are read as functional. Hao et al., Coconut (arXiv 2412.06769) feeds the last hidden
state back as the next input embedding: ATR's feedback edge, built by training, run for a few steps, no attractor map. Zhang et al., Soft
Thinking (arXiv 2505.15778) is the closest inference-only relative (training-free self-feedback on a frozen model), mediated by the output
distribution rather than the residual stream. Ouro (arXiv 2510.25741) and retrofitted recurrence (arXiv 2511.07384) convert pretrained
stacks into loops by training. The looped-LM literature names its failure mode latent collapse, the hidden state falling into an
input-independent fixed point, and engineers against it (STARS, arXiv 2605.26733; Solve the Loop, arXiv 2605.12466). All\*. What this literature suppresses or exploits, ATR studies descriptively in a model never trained for the loop.

**Blayney et al., A Mechanistic Analysis of Looped Reasoning Language Models (arXiv 2604.11791).**\*
Compares Huginn, Ouro, and retrofitted-recurrence Llama, and diagnoses convergence with lag-1 successive-iterate difference norms on the
looped residual stream. Finding 3 shows this gate is period-2-blind as a matter of arithmetic: a 2-cycle registers as a constant nonzero
lag-1 difference, or as a fixed point under stride-2 sampling, never as a cycle. The lag-k correction (`gate_lag` and `lag_scan` in
`atr_engine.py`) is therefore a citable methodological contribution against live literature, and it retroactively questions ATR's own
Pythia-410m null (finding 7).

**Deep equilibrium models.** Bai, Kolter, Koltun (NeurIPS 2019) replaced explicit depth with root-finding for the fixed point of one
weight-tied block, with the founding admission that plain forward iteration often fails to converge; the repair toolchain exists because
undamped iteration stalls or oscillates (Anderson acceleration, arXiv 2410.19460; Jacobian regularisation, arXiv 2106.14342; monotone
operator networks, arXiv 2006.08591). Standard fixed-point numerics give the mechanism class: an eigenvalue of magnitude above 1 with
negative sign at a fixed point produces oscillation around it. Finding 4 is that signature in a pretrained transformer (a period-doubling
configuration: a near-fixed point whose one unstable direction yields a stable 2-cycle), localised to a named component, which no DEQ work
does. All\*.

**Marcus, Westervelt, Dynamics of Iterated-Map Neural Networks (Phys Rev A 1989).**\*
https://neuron.eng.wayne.edu/tarek/MITbook/ref/refs.html
The classical anchor for finding 3: in discrete-time parallel-update neural networks, when a fixed point destabilises, the generic new
attractor is a period-2 oscillation. ATR adds the transformer instantiation and the single-head OV mechanism, which has no analogue in these
homogeneous models.

**Sussillo, Barak, Opening the Black Box (2013).**\* https://direct.mit.edu/neco/article/25/3/626/7854
The methodological ancestor: find fixed points of a trained recurrent network, linearise around them, read the computation from the
attractor skeleton. ATR is this program transplanted to a transformer made recurrent by an external loop.

**Dong, Cordonnier, Loukas, Attention is Not All You Need (ICML 2021).**\* https://arxiv.org/abs/2103.Pure self-attention converges doubly exponentially to a rank-1, token-uniform state; skip connections
and MLPs counteract the collapse. Finding 2 is this token-uniformity bias expressed under closed-loop iteration, where effective depth
reaches hundreds of blocks; the counterweight result is consistent with finding 1, since with MLPs and skips present the collapse is not to
one global point. Geshkovski, Letrouit, Polyanskiy, Rigollet model tokens as interacting particles that cluster as depth grows, the final
configuration set by the input\*. The oversmoothing line anticipates finding 2 and supports finding 1; none of it
describes an unplanned oscillation coexisting with position-uniform states.

**Attention as associative memory.** Ramsauer et al. (ICLR 2021, arXiv 2008.02217) identify attention with the update rule of a continuous
modern Hopfield network whose fixed points are global averages (a position-uniform state, finding 2) or metastable subset averages
(resembling the basins of finding 1). The energy-descent guarantee holds only for the idealised symmetric update; a full block with MLP,
LayerNorm, and external renormalisation has no Lyapunov function, which is why a limit cycle is possible at all. Energy Transformer (arXiv
2302.07253) and Hyper-SET (arXiv 2502.11646) build blocks that provably descend an energy and so produce fixed points only; Hyper-SET
independently arrives at ATR's two mechanics (norm-constrained states, repeated block) as design principles. All\*.

**The loop's preconditions.** Transformer Layers as Painters (AAAI 2025, arXiv 2407.09298) ran frozen pretrained layers in altered orders
and loops, the nearest published practice of iterating frozen layers off-distribution; it did not renormalise, close the full
output-to-input loop, or map attractors. Heimersheim and Turner (LessWrong 2023) measured roughly 4.5 percent
per-layer residual norm growth in GPT-2 class models: without the energy renormalisation the loop diverges in norm, so the attractors are
properties of the renormalised map. Both\*.

## GPT-2 internals

**Elhage et al., A Mathematical Framework for Transformer Circuits (Anthropic 2021).**\*
https://transformer-circuits.pub/2021/framework/index.html
The residual stream is a shared communication channel; heads decompose into QK circuits (where to
attend) and OV circuits (what is written); copying is scored by positive real eigenvalues of the full OV circuit W_U W_OV W_E. This supplies
the formalism of finding 4: the -4.3 action on the flip axis d is a strong anti-copying eigenmode, and the head-level attribution is this
framework applied to closed-loop dynamics.

**Lens instruments and their failure modes.** The logit lens (nostalgebraist 2020) decodes intermediate residual states through the final
LayerNorm and the unembedding W_U; its structural failure mode is that it reads only components aligned with W_U's strong directions. The
tuned lens (Belrose et al. 2023, arXiv 2303.08112) documents the logit lens's brittleness and per-model variability. Finding 6 is a concrete instance of the known blind spot: any lens defines a verbalizable subspace and misses its
complement, where the closed-loop motion lives, so lens-based accounts of the loop would wrongly report stasis. Both\*.

**The head catalogue, and where L11.H8 is not.** Induction heads (Olsson et al. 2022): prefix matching plus copying, the copying OV raising
the attended token's logit; GPT-2 Small induction heads 5.5, 5.8, 5.9, 6.9, paper\*.
The IOI circuit (indirect object identification; Wang et al. 2022, arXiv 2211.00593\*): 26 heads including negative
name movers 10.7 and 11.10, which write against the correct name; head list
https://raw.githubusercontent.com/ArthurConmy/Automatic-Circuit-Discovery/main/acdc/ioi/utils.py. Copy suppression (McDougall, Conmy,
Rushing, McGrath, Nanda 2023, arXiv 2310.04625\*): L10.H7 detects the currently predicted token and writes against
its unembedding; this is the class the suppression tests (BELL_PRIMER Part 7) show L11.H8 opposes on ordinary text, where it raises the
attended token's score at 91.4 percent of positions. Successor, greater-than, and year heads (arXiv 2312.09230; arXiv 2305.00586)\* place documented number and date machinery in layers 5 to 9. L11.H8 appears in none of these catalogues; the only
documented layer 11 copy machinery (11.10) has the opposite sign to finding 4.

**Kissane, Krzyzanowski, Bloom, Conmy, Nanda 2024, Attention Output SAEs (arXiv 2406.17759).**\* https://arxiv.org/abs/2406.17759, per-head cards
https://robertzk.github.io/gpt2-small-saes/
The only public per-head documentation of L11.H8. SAEs (sparse autoencoders, learned dictionaries of
directions) were trained on GPT-2 Small attention outputs and the top features attributed to each of the 144 heads. A direct read of the
L11.H8 card (the card itself labels the head "11.8") shows its top feature (1958) puts positive logits almost entirely on glitch and undertrained tokens (" guiActiveUn", "ertodd",
"ThumbnailImage", "ActionCode", "externalToEVA", byte fragments) and negative logits on the most frequent function tokens (" the", ",", "
and", " in", " a", " to", " of"); the next four features promote numerals, dates, years, and round quantities, with glitch tokens
("rawdownload", "oreAndOnline", "embedreportprint", " TheNitrome") at their negative ends. This is exactly the polarity of finding 5: the
head's output features already span a frequent-token versus glitch-token axis, the axis the flip runs along. The pattern is head-specific by
control: the L11.H10 card shows ordinary verb and event-structure features and no glitch axis. The record contains no interpretation, prose
account, or causal test of these features. Caveat: extreme vocabulary logit lists surface glitch tokens
spuriously (aizi, below), which tempers but does not remove the card evidence.

**Precedent for readout-invisible computation.** Entropy neurons (Gurnee et al., arXiv 2401.12181; Stolfo et al., arXiv 2406.16254) have
high weight norm and near-zero direct logit effect because they write into the effective null space of W_U (whose singular values drop
sharply near index 755), acting on entropy through the final LayerNorm scale. Cancedda, Spectral Filters, Dark Signals, and Attention Sinks
(ACL 2024, arXiv 2402.09221) found dark low-band residual signals that barely affect logits yet carry essential function. All\*. These are the strongest precedent for finding 6: GPT-2 components
demonstrably route computation through readout-invisible directions; ATR extends this to an attention head OV output carrying a limit cycle,
with 73 percent of d in W_U's weakest directions. The Anthropic workspace paper (Transformer Circuits, July 2026)\*
https://transformer-circuits.pub/2026/workspace/index.html gives finding 6 its sharpest vocabulary: a small mid-layer subspace carries most
causal effect on outputs, and most activation variance lies outside it.

**Anisotropy and outlier dimensions.** Ethayarajh 2019 (arXiv 1909.00512): GPT-2 representations occupy a narrow
anisotropic cone, the most-trained pole of finding 5. Rogue and outlier dimensions (Timkey and van Schijndel, arXiv
2109.04404; Kovaleva et al., arXiv 2105.06990; Puccetti et al., arXiv 2205.11380): a few dimensions dominate similarity yet barely matter
behaviourally, and outlier magnitudes track token frequency; the dissociation between what dominates geometry and what drives behaviour
parallels findings 5 and 6. All\*.

## Glitch tokens

Glitch tokens are vocabulary entries that received few or no weight updates in training; in GPT-2 they cluster near the mean embedding.

**Rumbelow and Watkins, SolidGoldMagikarp I to III (LessWrong / Alignment Forum, Feb 2023).**\*
https://www.lesswrong.The discovery posts: tokens closest to the embedding centroid behave anomalously, cannot be repeated
by the model, and derail generation; the recurring geometric marker is centroid proximity, not embedding norm; the provenance is a
tokenizer-corpus mismatch (Reddit r/counting usernames, game logs, and boilerplate earned byte pair encoding merges while the training
corpus excluded them). This characterises the never-trained pole of finding 5, including its resistance to verbalisation. Purely
input-driven prompting; no internal dynamics.

**Land and Bartolo, Fishing for Magikarp (EMNLP 2024, arXiv 2405.05417).**\*
https://arxiv.org/abs/2405.05417 and https://github.com/cohere-ai/magikarp
Systematic undertrained-token detection with about 90 model reports including all four GPT-2 sizes:
1,236 to 2,301 candidates and 36 to 68 verified undertrained tokens per size (excluding special and single-byte). The repo states that tied-embedding models such as GPT-2 need hand-selected unused-token
reference sets because the plain weight indicator is unreliable there; the paper's indicator is cosine distance to the mean of known-unused
token (un)embeddings after removing a shared component, not raw norm\*. This is the closest
methodological relative of ATR's criterion, and it agrees with ATR's negative result: in GPT-2, low embedding norm does not identify glitch
tokens (the lowest-norm rows are frequent function words; the signature is proximity to the mean embedding). No source states that
falsification as a measured GPT-2 result, so it stands as a citable standalone.

**The detection-tool family.** GlitchHunter (FSE 2024, arXiv 2404.09894) found 7,895 glitch tokens across 7 LLMs and confirmed they cluster
in embedding space; GlitchMiner (arXiv 2410.15052) and AnomaLLMy (arXiv 2406.19840) detect via predictive entropy and API confidence; Secret
Dictionary (arXiv 2605.22005) and UTF fingerprinting (arXiv 2410.12318) detect from weight geometry alone. Supporting geometry: Mu and
Viswanath, All-but-the-Top
(ICLR 2018, arXiv 1702.01417) established a dominant frequency-linked, mean-anchored direction in embedding
spaces, the geometry the flip axis d is a dynamical expression of; Watkins, Mapping the Semantic Void, probed the mean-embedding
neighbourhood and found it structured; a mechanistic LessWrong post explains why unspeakable tokens are silent under tied embeddings (no
direction makes them the argmax). All\*. aizi's random-direction baseline\* supplies the caveat that extreme logit lists surface glitch tokens spuriously.

**The confirmed null.** No located work connects glitch or undertrained tokens to a model's internal dynamics. In every source above they
are anomalous inputs: discovered by prompting, detected by querying or weight inspection, exploited or repaired. The null held under
repeated searches with varied phrasings across 2023 to 2026. Two boundary cases are preempted explicitly. GlitchProber (ASE 2024, arXiv
2408.04905)\* reads internal activations, but only as a classifier signal evoked by a glitch token present in the
input; no closed loop, no autonomous dynamics. The successive-paraphrasing cycles paper (above) has the period-2 phenomenology but never
mentions glitch tokens, embedding centroids, or the untrained region. Nothing in the record resembles finding 5's measurement: a dynamical
mode aligned with the glitch-cluster direction (cos -0.596, significant under uniform and norm-matched nulls, 45 of 50 pole-aligned tokens
in the near-centroid cluster, no glitch token in the input).

## Text-level loops and the lineage

**Degeneration and repetition self-reinforcement.** Holtzman et al. (ICLR 2020, arXiv 1904.09751): maximisation-based decoding drives
generation into repetitive loops, the everyday token-level attractor of the generation map. Xu et al. (NeurIPS 2022, arXiv 2206.02369)
quantified self-reinforcement in GPT-2: the more a sentence appears in context, the higher the probability of producing it again. Both\*.

**Model collapse.** Shumailov et al. (Nature 2024, arXiv 2305.17493): training generation n+1 on generation n's output loses distribution
tails first, then collapses variance. Alemohammad et al. (ICLR 2024, arXiv 2307.01850) and Dohmatob et al. (ICML 2024, arXiv 2402.07043)
identify tail truncation as the mechanism. All\*. This is a different object: the learning map iterated across
generations of models, where ATR
iterates the inference map across states of one frozen model. Both lose low-probability structure first, but ATR's loop has no training
signal, so its degeneration cannot be blamed on estimation error; it is a property of the learned map itself. No paper states or tests this
correspondence.

**Telephone games and paraphrase attractors.** Translation Party (2009) is the earliest popular demonstration that iterating a learned
text-to-text map finds fixed points. Perez et al., When LLMs Play the Telephone Game (ICLR 2025, arXiv 2407.04503)
D: transmission chains evolve toward attractor states in property space, with attractor strength depending on task constraint. Kaplanski
(2026, arXiv 2605.02236) measures how much injected text moves a settled loop into another basin and whether the move persists: the
text-level counterpart of ATR's perturbation and basin-escape experiments. All\*.

**Self-refinement convergence.** Self-Refine (NeurIPS 2023, arXiv 2303.17651) closes the loop through text plus an instruction, with
convergence imposed by a stopping rule, not analysed as dynamics. Huang et al. (ICLR 2024, arXiv 2310.01798): intrinsic self-correction
without external feedback fails to improve and often degrades, consistent with dissolution rather than convergence to meaning. Both\*.

**The Lucier lineage.** Alvin Lucier's I Am Sitting in a Room (1969) is the procedure of feeding a medium its own output until the medium's
character dominates. The located record transposes it to computational substrates three times. Backes, i am sitting in a machine\* https://www.martinbackes.com/i-am-sitting-in-a-machine/: an artificial voice through an MP3 encoder 3000 times, a
stated homage, the iterated operator a codec. Santos, The Degradation of Speech (2023)\*
https://dorothysantos.com/portfolio/the-degradation-of-speech/: repeated reading into a neural speech recogniser, human in the loop. Vats,
Crandall, Goree, A Markovian View of Iterative-Feedback Loops in Image Generative Models: Neural Resonance and Model Collapse (2026, under
review)\* https://arxiv.org/abs/2602.19033: proves a broad class of iterative feedback
processes converges to low-dimensional invariant structure in latent space, treats diffusion loops, style transfer, and Lucier's piece in
one framework, and is the closest published Lucier-to-neural-network analogy: image models, no language model, no per-basin semantic
mapping, no limit-cycle taxonomy. No located work runs the analogy on a language model, and none at the activation level.

## What remains ours, on this evidence

Each claim below, with the nearest work it must be distinguished from. Scope note: every claim means that this review found no such
work, not proof of absence; absence claims are bounded by the review's coverage, and work under different vocabulary could overturn one.

1. **The frozen-model residual-stream attractor census.** We found no work that re-injects the full final-layer residual tensor of an unmodified
   pretrained LM into layer 0, renormalises energy, and iterates to exhaustion, mapping basins over a prompt corpus versus noise (findings
   1, 2, 7). Distinguish from Coconut and Soft Thinking (the same feedback edge, used for reasoning, no attractor map) and Layers as
   Painters (frozen-layer loops without the closed loop or the census).
2. **The exact cycle with full mechanism.** No prior report of an exact discrete limit cycle in pretrained transformer activation dynamics,
   and no prior localisation of a cycle to a named attention head's OV circuit with measured per-step multipliers (findings 3, 4).
   Distinguish from Wang et al. (approximate period-2 at the text interface, no mechanism) and Marcus and Westervelt (period-2 in
   homogeneous neural maps, no transformer, no component).
3. **The glitch-region structural role.** We found no work tying the untrained vocabulary region to any internal dynamical mode; the record
   treats glitch tokens exclusively as inputs (finding 5). Distinguish from GlitchProber (internal activations as a classifier signal for a
   glitch token in the input) and the Kissane et al. L11.H8 card (the same axis in static SAE features, uninterpreted and untested).
4. **The lag-k correction.** Live looped-LM literature diagnoses convergence with lag-1 successive-difference norms; we found no work flagging that this
   gate misclassifies period-2 cycles, which finding 3 demonstrates concretely (and which makes ATR's own Pythia-410m null provisional).
   Distinguish from Blayney et al. (the gate in use) and Wang et al. (a periodicity metric at the text level only).
5. **The readout-invisible persistent state, at pilot confidence.** We found no work placing iterated dynamics inside the unembedding's weak
   directions; the precedents (entropy neurons, dark signals, the workspace paper) establish the invisible subspace statically (finding 6).
   Distinguish from Cancedda (static spectral bands, no dynamics).
6. **The Lucier frame for language models.** The analogy has been executed for codecs, speech recognisers, and image latents, never for a
   language model and never at the activation level. Distinguish from Vats et al. (the same frame, image latents, convergence only).
7. **The low-norm falsification.** The literature jointly implies raw low embedding norm is the wrong undertrained-token criterion for
   GPT-2, but no source states the measured result (lowest-norm rows are frequent function words; the signature is mean proximity).
   Distinguish from Land and Bartolo (the tied-embedding caveat, stated for their indicator, not as a GPT-2 measurement).

Caveat: absence of evidence is bounded by this review's coverage (arXiv, ACL, NeurIPS, ICML, ICLR, LessWrong, Alignment Forum, as of
July 2026). A same-niche preprint under different vocabulary (for example "activation recycling" or "representation feedback") could
exist outside the phrasings this review covers.

## Sources

URLs for the works cited above, deduplicated, grouped by topic.

Dynamical systems: https://github.com/locuslab/deq , https://wiki.math.ntnu.no/_media/ma2501/2014v/fixedpoint.pdf ,
https://arxiv.org/html/2410.19460 , https://arxiv.org/abs/2106.14342 , https://arxiv.org/abs/2006.08591 , https://arxiv.org/pdf/2502.05171 ,
https://arxiv.org/abs/2604.11791 , https://arxiv.org/abs/2605.26733 , https://arxiv.org/abs/2605.12466 , https://arxiv.org/abs/2510.25741 ,
https://arxiv.org/abs/2511.07384 , https://arxiv.org/pdf/2412.06769, https://arxiv.org/abs/2103.03404,
https://people.lids.mit.edu/yp/homepage/data/2023_transformers1.pdf , https://arxiv.org/abs/2407.09298 ,
https://www.lesswrong.com/posts/8mizBCm3dyc432nK8/residual-stream-norms-grow-exponentially-over-the-forward,
https://arxiv.org/abs/2008.02217v3 , https://arxiv.org/abs/2302.07253 , https://arxiv.org/abs/2502.11646 ,
https://direct.mit.edu/neco/article/25/3/626/7854 , https://neuron.eng.wayne.edu/tarek/MITbook/ref/refs.html ,
https://aclanthology.org/2025.acl-long.624/ and https://arxiv.org/abs/2502.15208, https://arxiv.org/abs/2402.09221 ,
https://transformer-circuits.pub/2026/workspace/index.html (secondary coverage https://thezvi.substack.com/p/no-space-like-j-space )

GPT-2 mechanistic interpretability: https://transformer-circuits.pub/2021/framework/index.html,
https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens , https://arxiv.org/abs/2303.08112 ,
https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html , https://arxiv.org/abs/2211.00593 ,
https://raw.githubusercontent.com/ArthurConmy/Automatic-Circuit-Discovery/main/acdc/ioi/utils.py , https://arxiv.org/abs/2310.04625 ,
https://arxiv.org/abs/2406.17759 , https://robertzk.github.io/gpt2-small-saes/ , https://arxiv.org/abs/2312.09230 ,
https://arxiv.org/abs/2305.00586 , https://arxiv.org/abs/2401.12181 , https://arxiv.org/abs/2406.16254 , https://arxiv.org/abs/1909.00512 ,
https://arxiv.org/abs/2109.04404 , https://arxiv.org/abs/2105.06990 , https://arxiv.org/abs/2205.11380

Glitch tokens: https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation,
https://www.lesswrong.com/posts/Ya9LzwEbfaAMY8ABo/solidgoldmagikarp-ii-technical-details-and-more-recent ,
https://www.lesswrong.com/posts/8viQEp8KBg2QSW4Yc/solidgoldmagikarp-iii-glitch-token-archaeology , https://arxiv.org/abs/2405.05417 and
https://github.com/cohere-ai/magikarp, https://aclanthology.org/2024.emnlp-main.649/ , https://arxiv.org/abs/2404.09894 ,
https://arxiv.org/abs/2408.04905 , https://arxiv.org/abs/2410.15052 , https://arxiv.org/abs/2406.19840 , https://arxiv.org/abs/2605.22005 ,
https://arxiv.org/abs/2410.12318 , https://arxiv.org/abs/1702.01417,
https://www.lesswrong.com/posts/c6uTNm5erRrmyJvvD/mapping-the-semantic-void-strange-goings-on-in-gpt-embedding ,
https://www.lesswrong.com/posts/dFbfCLZA4pejckeKc/a-mechanistic-explanation-for-solidgoldmagikarp-like-tokens ,
https://aizi.substack.com/p/explaining-solidgoldmagikarp-by-looking

Text-level loops and the lineage: https://arxiv.org/abs/1904.09751 , https://arxiv.org/abs/2206.02369 ,
https://www.translationparty.com , https://arxiv.org/abs/2407.04503, https://arxiv.org/abs/2605.02236 ,
https://arxiv.org/abs/2303.17651 , https://arxiv.org/abs/2310.01798 , https://arxiv.org/abs/2305.17493 ,
https://www.nature.com/articles/s41586-024-07566-y , https://arxiv.org/abs/2307.01850 , https://arxiv.org/abs/2402.07043 ,
https://www.martinbackes.com/i-am-sitting-in-a-machine/ , https://dorothysantos.com/portfolio/the-degradation-of-speech/ ,
https://arxiv.org/abs/2602.19033, https://arxiv.org/abs/2505.15778
