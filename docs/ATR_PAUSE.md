# ATR work: the pause is lifted

## Status: LIFTED (2026-07-31, operator ruling)

The understanding gate that stood here was removed by TC's explicit ruling on
2026-07-31, recorded during the alignment review (PR #103): "we remove the
gate, that's my official position."

No experiment is blocked by this document any more. Work is sequenced instead
by the experiment queue in [ALIGNMENT_REVIEW.md](ALIGNMENT_REVIEW.md) §5, whose
ordering TC has delegated to the review process. Anything in the record or the
issues that says "gated by ATR_PAUSE" or "blocked by the pause" should be read
as: queued in that §5 order, nothing more.

## What this file was

From 2026-07-25 to 2026-07-31 this file held an operator-set pause: no new ATR
experiments until a cold-writeup-plus-live-prediction understanding gate was
passed. The full original text is in git history (`git log -- docs/ATR_PAUSE.md`).
The file is kept, in this lifted form, because committed documents and open
issues link to it by name.

The gate's purpose (direction driven by understanding rather than momentum) is
carried forward by other means: the precondition-and-controls protocol for the
fixed-point work (ALIGNMENT_REVIEW.md §3.2), and the standing rules in §5 there
(every number script-generated; archive spec; corrections land before dependent
work).
