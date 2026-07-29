# Methodology: corpus provenance auditing

The protocol behind [RESULTS.md](RESULTS.md), written so it can be checked, argued with, or
run against a different corpus. `RESULTS.md` reports what was found; this document states how,
and — more importantly — what the design can and cannot decide.

---

## 1. Decomposing the hypothesis before testing it

The commissioned hypothesis was that WebText "contains evidence of non-western asymmetric
warfare via botfarming (be it karma farming, brigading otherwise)". Two independent claims
are bundled there, and a design that does not separate them will answer the easy one and
quietly imply it answered both.

| | Claim | Evidence class | Verdict available? |
|---|---|---|---|
| **(a)** | Influence-operation *content* is present in the corpus | The documents themselves | Direct |
| **(b)** | It arrived by *inauthentic amplification* | Reddit-side vote provenance | **Structurally unavailable** |

Claim (b) is a claim about **who voted and how**. WebText's construction rule — outbound
links from Reddit posts with ≥ 3 karma (Radford et al. 2019, §2.1) — means the vote history
is upstream of the corpus and was never released with it. No amount of reading the documents
recovers submitter identity, account age, vote timing, or subreddit.

This is a **scope condition, not a null result**, and the distinction matters. A null result
says "we looked and found nothing"; a scope condition says "this instrument cannot see it".
Reporting (b) as "not supported" without that qualifier would misrepresent a blind spot as a
measurement.

**What was done instead:** test (a) directly and exhaustively; test the one *text-visible*
consequence of (b) — republication and duplicate copy (§5); and state the remainder as out of
reach. The floor this sets is worth naming: a 3-karma threshold is a handful of votes. It is
a bar that requires no coordination whatsoever to clear, so even a positive finding on (a)
would carry little weight for (b).

---

## 2. Corpus selection and its bias

**Used:** OpenAI's `gpt-2-output-dataset` release of original WebText — 250,000 train +
5,000 valid + 5,000 test = 260,000 documents, 152.2M byte-pair-encoding tokens.

This is the **held-out** portion of the WebText scrape, not the split GPT-2 trained on. That
substitution is forced (the training split was never published) and is defensible: both
splits come from one scrape, one collection rule, one deduplication pass, partitioned at
random. For questions about *what the collection pipeline admitted*, the held-out sample is
an unbiased draw.

It would not be defensible for a question about *what GPT-2 memorised* — that needs the
actual training documents. This study asks the former, and says so.

**Also used:** `domains.txt` from `openai/gpt-2`, the top 1,000 domains across the **entire**
corpus including the training split. This is the corrective for the sample's blind spots and
is near-independent of the text scan: one measures links across everything, the other content
across a sample. Where two near-independent instruments agree (§7 of RESULTS.md), the finding
is much stronger than either alone.

**Integrity.** All four files are pinned by SHA-256 in `01_fetch_corpus.py` and re-verified on
every run; the recorded MD5s match the Content-MD5 headers Azure serves. Raw files are
gitignored (692 MB); the analysis outputs that every claim rests on are committed.

---

## 3. Indicator construction

### Inclusion rule

An indicator is admitted only if a **public, citable attribution** ties the domain, brand, or
persona to a state influence apparatus — a government designation, a platform takedown, or a
named forensic investigation. Every entry in `indicators.json` carries its `attribution`
array. Sources used: the U.S. Department of Justice's Internet Research Agency indictment
(Feb 2018), U.S. Treasury/OFAC designations (Apr 2021), the U.S. State Department Global
Engagement Center's *Pillars* report (Aug 2020), FireEye's Iranian network attribution
(Aug 2018), the New Knowledge report to the Senate Select Committee on Intelligence
(Dec 2018), and the Office of the Director of National Intelligence assessment (Jan 2017).

**Excluded by rule:** outlets that are merely ideologically aligned, contrarian, or
frequently wrong. "Publishes things I think are false" is not an attribution, and a list
built that way measures the compiler's politics rather than the corpus.

### Tiering is an evidence gradient, not a severity ranking

The tiers exist because a hit means something **different** in each, and collapsing them
produces a headline number that cannot be interpreted.

- **Tier A (covert, 27 indicators / 19 domains).** Assets with documented inauthentic
  distribution and little organic audience. **A Tier A hit is the hypothesis's real test** —
  such material had no honest route into a link aggregator.
- **Tier B (overt state media, 27).** RT, Sputnik, Xinhua, Press TV and peers. Presence
  proves the pipeline admitted state content; it does **not** prove the karma was inauthentic,
  because these outlets have real readers who really post their links.
- **Tier C (documented amplifiers, 3).** Western-hosted proxies that launder state narratives.
- **Tier M (meta-discussion, 9).** Vocabulary of coverage *about* influence operations. This
  tier is a **control**, not evidence: it measures how much the corpus discusses the subject,
  which is the confound a naive keyword count would silently absorb.
- **Tier BASE (baseline, 11).** Mainstream outlets for rate contrast. Al Jazeera is included
  deliberately as a state-funded broadcaster whose presence in western sharing habits is
  normalised — it makes the Tier B interpretation argue for itself rather than assume itself.

### Expected-null sentinels

Russian-language Prigozhin properties (`riafan.ru`, `politexpert.net`) and North Korean state
domains are included **knowing** they will be near-zero in an English-dominated corpus. This
is deliberate: an unasked question and a recorded null look identical in a summary table.
Including them converts an assumption into a measurement.

### Window discipline

WebText closed in **December 2017**. Attribution dates routinely postdate that (2018–2021);
what matters is whether the asset was *publishing in-window*. Every indicator carries
`active_in_window`. The documented covert Chinese English-language networks ("Spamouflage",
reported from 2019) fall **outside** the window and are therefore absent by construction —
recorded in `coverage_notes` so their absence is never read as a finding.

---

## 4. Matching, and why the boundary rules are unit-tested

Substring matching would wreck this study. `rt.com` is a substring of `support.com`;
`Xinhua` is a prefix of `Xinhuanet`; `TEN_GOP` sits inside `OFTEN_GOPHER`.

**Domain rule:** extract domain-shaped tokens, then match when the indicator pattern is the
**registrable tail** of the token. `www.rt.com` and `amp.rt.com` hit `rt.com`; `support.com`
and `rt.com.au` do not. Multi-label patterns work the same way, which is why `people.com.cn`
does not satisfy `people.cn`.

**Phrase rule:** alphanumeric/underscore boundaries on both sides, honouring per-indicator
case sensitivity. `Russia Today` matches the brand and not "in russia today the weather".

Both rules are pinned by 7 tests in `tests/test_webtext_provenance_scanlib.py`, including the
false-positive cases explicitly. `scanlib.py` is kept free of input/output so the rules are
testable in isolation.

### The capture-bias correction that changed the study

The initial scan found **zero** `sputniknews.com` text hits against a census expectation of
hundreds. The cause is structural: **articles rarely contain their own domain**. Domain
matching cannot see a Sputnik article; it can only see someone *linking* to one.

The fix was to add a `Sputnik` phrase indicator with `fp_risk: high` recorded up front, and
let classification separate the satellite and Pluto's *Sputnik Planum* from the outlet. That
one indicator produced **74** state-produced documents where the domain indicator produced 0.

The general lesson, and the reason limitation 3 in RESULTS.md is stated as strongly as it is:
**every count here is a floor.** A scan of this kind under-counts by construction, and the
under-count is *worse for outlets whose content spreads without attribution* — precisely the
material the hypothesis is about.

---

## 5. Classification: the step that decides the study

A corpus **full of BBC reports about the Internet Research Agency** and a corpus **containing
Internet Research Agency articles** produce identical keyword counts and mean opposite things.
Every document-indicator pair therefore gets exactly one verdict.

| Verdict | Means | Typical deciding evidence |
|---|---|---|
| `origin` | The document *is* the source's content | `MOSCOW (Sputnik) —` dateline; `© Sputnik /` photo furniture; `Editor: … 丨CCTV.com`; `@chinadaily.com.cn` contact |
| `laundered_origin` | Source's content, carried in via a rehost | "originally posted at strategic-culture.org"; `ORIGINALLY PUBLISHED AT RT.COM`; `Press TV original title:` |
| `citation_amplification` | Third party approvingly building on the source | "Sources: GlobalResearch.ca"; story built on an rt.com URL |
| `wire_carriage` | Third party carrying wire copy with attribution | `BEIJING, Jan. 16 (Xinhua) --`; "(With files from Sputnik \| Reuters \| AFP)" |
| `mention` | The document talks *about* the source | "state-run Press TV said"; "state-funded RT.com claims" |
| `false_positive` | Not the entity at all | *Deep in the Heart of Texas*; Sputnik Planum; Heart of Texas Foundation |
| `ambiguous` | Undecidable by rule, awaiting adjudication | — |

Three of these count as **state-produced** in the headline figure: `origin`,
`laundered_origin`, `wire_carriage`. `citation_amplification` and `mention` are reported
separately and never folded in.

### Adjudication protocol

Heuristics propose; a human reading the document disposes. Rules:

1. **A recorded adjudication always overrides the heuristic**, never the reverse.
2. **Mandatory review:** every Tier A pair, every Tier B/C *domain* pair, every pair the
   heuristics left `ambiguous`, and a capped sample of each large phrase family.
3. **Every adjudication carries a note stating the deciding evidence** — the quoted dateline,
   byline, or credit line. `output/adjudications.json` is the audit trail; a verdict without
   a reason is not admissible.
4. **`03_classify_hits.py` reports any unreviewed Tier A pair by name** in
   `classification_summary.json`, so an incomplete review cannot pass silently.
5. **Zero pairs may remain `ambiguous`** in the final record. 213 of 605 pairs are
   hand-adjudicated; the residue is heuristic verdicts on families whose pattern was
   established by reading a substantial sample of them.

### Deliberate conservative bias

Where evidence was genuinely ambiguous, the verdict went to `mention` — the reading *least*
favourable to the hypothesis. Combined with the capture bias in §4, both known biases push
the same way: **the reported state-produced count is an underestimate.** This is the correct
direction for a study whose framing invites motivated reasoning; a study that finds less
than the truth cannot be accused of manufacturing its own conclusion.

---

## 6. The coordination probe, and why its null is weak

If the same state-produced text entered the corpus through several URLs, each copy cleared
the karma gate separately — republication doing exactly what laundering is for. That is the
one fingerprint of claim (b) that survives into the text.

**Method:** 8-word shingles; **exact** pairwise Jaccard similarity over all 578 non-baseline
hit documents (N is small enough that brute force is exact — no minhash approximation);
single-link clustering at ≥ 0.50, with all pairs ≥ 0.25 also recorded so the null cannot hide
behind a badly chosen threshold.

**Result:** zero pairs at either threshold.

**Why this null is weak, stated plainly:**

- WebText was **already deduplicated** by OpenAI. Some of what this probe looks for was
  removed before the corpus was published.
- It can only compare documents that the scan already flagged — coordination among documents
  no indicator caught is invisible to it.
- Near-duplicate *text* is one possible signature of coordination among many; its absence
  does not exclude coordinated *submission* of distinct documents.

It is reported because a recorded null is worth more than an unasked question, not because it
settles anything.

---

## 7. Threats to validity

| Threat | Direction | Handling |
|---|---|---|
| Reddit vote provenance unavailable | Fatal to claim (b) | Declared as a scope condition, not converted to a null |
| Held-out split ≠ training split | Unknown, likely negligible | Same pipeline; census covers everything as corrective |
| Articles rarely name their own domain | **Under-count** | Phrase indicators added; all counts stated as floors |
| Indicator list bounded by public attribution | **Under-count** | Inclusion rule stated; post-window gaps recorded in `coverage_notes` |
| Ambiguity resolved toward `mention` | **Under-count** | Declared; both known biases run the same direction |
| Census labels are second-level only | Both | `people` (People magazine) flagged and excluded from totals; `rt` collision noted |
| Adjudicator is also the study author | Confirmation risk | Every verdict carries quoted evidence so any reader can overturn it |
| Deduplication upstream | Weakens §6 null | Stated at the point the null is reported |

The last one is the one to watch. The mitigation is not independence — there was none — but
**legibility**: `adjudications.json` records the deciding quote for every hand verdict, so
disagreeing with this study requires reading 213 notes, not re-running 260,000 documents.

---

## 8. Porting this regime to another corpus

The design is corpus-agnostic. To run it against C4, The Pile, RedPajama, or a private
scrape:

1. **Re-derive the window.** Indicator `active_in_window` flags are specific to WebText's
   December 2017 close. A 2021 corpus admits the Spamouflage cluster this one excludes.
2. **Rebuild the baseline tier** to match the corpus's own composition — the baseline exists
   to give the state-media rate a denominator that means something *for that corpus*.
3. **Re-test the boundary rules.** They encode assumptions about how domains appear in text;
   a corpus of stripped HTML or of PDFs behaves differently.
4. **Expect the capture bias to be worse, not better,** anywhere documents are further from
   their source page than WebText's scraped articles.
5. **Keep the tier separation.** The single most misleading thing this study could have
   produced is one undifferentiated count of "influence-operation hits" — a number that would
   have been dominated by Xinhua wire attributions in Reuters copy and by BBC reporting on
   troll farms, and would have supported a conclusion the evidence refutes.

---

## References

- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I. (2019).
  *Language Models are Unsupervised Multitask Learners.* §2.1 describes WebText's
  construction: outbound Reddit links with ≥ 3 karma, scraped and deduplicated, Wikipedia
  removed.
- OpenAI. *gpt-2 model card* (`model_card.md`) — WebText provenance and the top-1,000 domain
  list.
- OpenAI. *gpt-2-output-dataset* — the 260,000-document public sample.
- Attribution sources for individual indicators are recorded per-entry in `indicators.json`.
