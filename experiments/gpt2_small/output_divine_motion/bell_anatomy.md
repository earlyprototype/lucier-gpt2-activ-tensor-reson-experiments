# Cycle Anatomy: Inside the Divine Period-2 Cycle

*Terminology: the flip axis d was called "the hinge" in earlier revisions of these documents; script names, folder names, and JSON keys keep the old word.*


*Follow-up to [divine_motion_report.md](divine_motion_report.md). Runner: [`06_bell_anatomy.py`](../06_bell_anatomy.py); raw numbers: [`bell_anatomy.json`](bell_anatomy.json). Single Divine trajectory (the Syntactic prompt), states recovered from the saved iteration-1000 checkpoint.*

## Questions asked

1. What is under phase B's readout distribution (is there a second voice)?
2. Is the flip a literal sign-flip along some axis, and where does that axis live?

## Results

**Period-2 verified exactly:** cos(A, f(f(A))) = 1.000000.

**One token set, two probability levels.** Phase B's top-10 is the same token set as phase A's, in nearly the same order, at different probabilities: `Divine` falls from 0.505 to 0.225 while `【` rises from 0.064 to 0.126; coherence is 0.318 in both phases and at the midpoint. There is no hidden second coherent cluster of related tokens. The cycle has a single token set; the phases differ in probability, not content.

**Energy shifts between positions.** At the last token position, phase A carries norm 1612 and phase B only 464 (full-tensor norm is conserved by construction). The oscillation redistributes energy across positions each step; the loop's re-normalisation pumps it back.

**The oscillation has exactly one flip axis.** Writing A = M + d and B = M - d, the per-position flip axes are identical: mean pairwise cosine 1.0000 across all ten positions. The whole tensor tilts on a single global direction. This makes the negative-eigenvalue reading of the cycle nearly literal: one rank-1 direction that the normalised map inverts each pass.

**The flip axis is almost perfectly invisible to the readout.** The axis d produces a logit response of 33 against 612 for equal-norm random directions: a ratio of 0.054, far more suppressed than the per-step average (0.295). Decomposed against the unembedding's singular directions, 73% of the axis's energy sits in the bottom-100 (lowest-response) directions and only 13% in the top-100. The pivot M is similarly in the low-response corner (67% bottom-100). The Divine phenomenon inhabits the model's least speakable subspace.

**The riders (what little does swing).** Tokens whose logits rise most toward phase A: `Change, Divine, Release, Form, Fin, Air, Dou, Ground, Physical, Wind` (a coherent game/elemental-move vocabulary). Toward phase B: `reddits, ertodd, ModLoader, espie, annis, quickShipAvailable, ocrats, orkshire, colonists`. Several of these (`ertodd`, a fragment of ` petertodd`; `quickShipAvailable`; and neighbours) match the published GPT-2 anomalous "glitch token" cluster (the SolidGoldMagikarp family, Rumbelow and Watkins 2023): under-trained tokens whose embeddings sit in a degenerate corner of embedding space. Phase B leans toward that corner. This is consistent with the earlier speculation that the Divine attractor sits near the anomalous-token region, and now has direct evidence.

## Interpretation

The Divine cycle is a rank-1 self-negating mode: a single direction the forward map inverts each pass, swinging between a game-move-vocabulary pole and the glitch-token pole, with the swing itself almost entirely invisible to the vocabulary projection. The stable `Divine` argmax is the shadow of the pivot M, which both phases share.

## Caveats

One trajectory, one prompt, one model. The glitch-token identification is by inspection against published lists, not a systematic test. Whether all 34 Divine prompts share this same flip axis remains open (blocked on prompt_library restoration, issue #9).
