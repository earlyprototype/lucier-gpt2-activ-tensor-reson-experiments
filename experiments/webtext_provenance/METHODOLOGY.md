# Methodology: corpus provenance auditing

The protocol behind [RESULTS.md](RESULTS.md), written so it can be checked, argued with, or
run against a different corpus. `RESULTS.md` reports what was found; this document states how,
and — more importantly — what the design can and cannot decide.

---

## 0. The design flaw at the centre of this study, stated first

The first version of this study consisted only of §§1–5 below: a list of publicly attributed
actors, matched against a corpus. That design is **confirmatory**. It answers "are these
specific known things present?" and it is structurally incapable of answering "is anything
coordinated present?"

The consequences are not marginal:

- It **cannot discover an unattributed operation** — precisely the case with the most
  intelligence value.
- Its coverage gaps are inherited from the attribution record, not from the corpus. The
  post-2017 window gap (§3) is not an incidental caveat; it is this flaw showing through.
- Its null result is **uninterpretable without a positive control**. The original
  near-duplicate check compared only documents the list had already flagged, found nothing,
  and reported a clean null — but had never demonstrated it could find duplication anywhere.
  Run over the whole corpus, the same method finds **727 clusters**. The null only became
  meaningful once the detector was shown to work.
- Adding confidence intervals does not repair it. Bounds on a lookup are bounds on the
  lookup, and the design's question presupposes much of its answer.

**The correction is to invert the order: let the corpus nominate anomalies by structure, and
consult identity only afterwards.** That is what §§6.3–6.4 do, and their results are what
forced a correction to the study's headline claim (§6.4). Where the two approaches disagree,
the source-agnostic result is the one to believe, because it is the one that could have come
out otherwise.

A reader who wants only the defensible core should read §§6.3–6.4 and treat §§1–5 as a
well-bounded lookup with known blind spots.

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

## 6. Inference, and the two tests that need no list

### 6.1 Why exact methods

Every central result is a zero-count observation, and the normal approximation to a binomial
is not merely inaccurate there — it is degenerate, returning a zero-width interval around
zero. `statlib.py` therefore implements exact Clopper-Pearson bounds, exact binomial tests
and the power inversion in pure Python, so the repository gains no dependency;
`tests/test_webtext_provenance_statlib.py` cross-verifies each against scipy where scipy is
installed and skips where it is not.

The quantities that matter:

- **A bound, not a shrug.** Zero hits in 260,000 documents gives p < 1.152 × 10⁻⁵ at 95%,
  ceilinging each absent covert asset at **~92 documents corpus-wide**. Bonferroni across
  the 18-domain family loosens that to ~181, which does not change the conclusion.
- **Power, so the null can be read.** The scan would have caught an asset contributing ~50
  documents corpus-wide with 80% probability. It says nothing about a handful of documents
  in eight million.
- **Recall, measured rather than asserted.** Comparing census expectation against text
  capture estimates scanner recall per domain: **0.04 for RT, 1.02 for Sputnik, 0.39 for
  Press TV**. Pooling numbers that heterogeneous would be arithmetic without meaning; the
  pooled ×2 figure exists in `statistics.json` and should not be quoted bare.

### 6.2 What inference does NOT apply to

`domains.txt` is a **census, not a sample** — it reports the whole corpus. Confidence
intervals and significance tests are meaningless against it: RT ranking 92nd is a fact, not
an estimate with sampling error. §6.4 uses it only as a fixed expectation against which to
measure something else, which is a different use.

And no statistic anywhere here speaks to inauthentic amplification. Vote provenance was never
released. The bounds constrain **content prevalence** only.

### 6.3 Corpus-wide duplication: the test with a positive control

The original coordination probe compared only the 578 documents the indicator list had
flagged, found zero pairs, and reported a null. That null was worthless: the method had
never been shown capable of finding duplication at all.

**Method now:** rolling polynomial hash over 8-word shingles, vectorised per document;
bottom-*k* MinHash sketch (*k* = 8) for all 260,000 documents; inverted index over sketch
values; candidate pairs are documents sharing ≥ 2 sketch values; exact Jaccard on candidates
only; single-link clusters at ≥ 0.50. Sketch buckets above 200 documents are skipped as
boilerplate artefacts — recorded, since a silent cap reads as coverage.

**Result:** 727 clusters, 2,860 documents, 13,102 pairs at ≥ 0.50 — and **zero clusters
containing a state-produced document**. The single cluster touching any indicator is
`base-breitbart`, a baseline: six columns sharing a verbatim author-bio block.

The detector demonstrably works, which is what converts the null from an absence of evidence
into evidence of absence *for duplication specifically*. The residual weaknesses stand:
WebText was already deduplicated upstream, and coordinated submission of *distinct* documents
would leave no duplication trace at all.

### 6.4 Rank-frequency residuals: anomaly before identity

Quoting "RT ranks 92nd" supplies no baseline and invites the reader to imagine one. The
analysis is to fit the corpus's own rank-frequency law across all 1,000 domains and ask which
domains exceed it — with no list involved.

**Method:** ordinary least squares of log(links) on log(rank), all 1,000 domains. Outliers at
residual z ≥ 2. A head-truncated refit (excluding the top 20 ranks, where aggregators flatten
the curve) is reported alongside, so no conclusion rests on the fit choice. **Identity is
consulted only after the outlier set is fixed.**

**Result:** R² = 0.9969, an extremely tight fit. Two positive outliers — `imdb` (z = +3.19)
and `reuters` (z = +2.22), neither an influence operation. Every state domain sits on the
line: **rt −0.20**, sputniknews +0.09, presstv −0.10, globalresearch −0.47, telesurtv −0.57.
The *baselines* deviate far more (nytimes −7.34, foxnews −2.41).

This forced a correction to the study's headline. RT's rank is not anomalous; it is within a
fifth of a standard deviation of prediction. If those outlets were being pushed into WebText
inauthentically, it left no trace in the link distribution.

**Stated no more strongly than the method allows:** rank is assigned *by* link count, so
these residuals have no clean null distribution. They are descriptive distances from a fitted
line and are deliberately **not** converted to p-values. Positive residuals have many innocent
causes — publication volume, absence of a paywall, archive depth.

### 6.5 What a fuller regime would add, and why it is not here

Two source-agnostic tests are not a complete forensic toolkit. The standard additions are
**burstiness** (coordinated submission clusters in time) and **stylometric clustering**
(shared authorship across nominally unrelated outlets). Neither is possible on this release:
the corpus carries no timestamps and no authorship metadata. That is a limit of the data, and
it is the reason the mechanism question stays unanswerable rather than merely unanswered.

---

## 7. Threats to validity

| Threat | Direction | Handling |
|---|---|---|
| **Confirmatory design: no discovery power** | **Fatal to §§1–5 as a forensic regime** | Not mitigated — *superseded* by §§6.3–6.4, which invert the order. §§1–5 stand as a bounded lookup |
| Reddit vote provenance unavailable | Fatal to claim (b) | Declared as a scope condition, not converted to a null |
| Held-out split ≠ training split | Unknown, likely negligible | Same pipeline; census covers everything as corrective |
| Articles rarely name their own domain | **Under-count** | Measured, not asserted: recall 0.04–1.02 by domain (§6.1) |
| Indicator list bounded by public attribution | **Under-count** | Inclusion rule stated; a symptom of the confirmatory design, not a separate defect |
| Ambiguity resolved toward `mention` | **Under-count** | Declared; both known biases run the same direction |
| Census labels are second-level only | Both | `people` (People magazine) flagged and excluded from totals; `rt` collision noted |
| Adjudicator is also the study author | Confirmation risk | Every verdict carries quoted evidence so any reader can overturn it |
| Heuristics tuned after reading the data | Inflates agreement | κ reported as **in-sample fit, not validation** (0.72 where the rules committed; 45% abstention) |
| Deduplication upstream | Weakens §6.3 null | Stated where the null is reported; the positive control is what makes it readable at all |
| Zipf residuals lack a null distribution | Overclaim risk | Never converted to p-values; head-truncated refit reported alongside |

Two deserve emphasis. **The confirmatory-design threat is not mitigated, it is replaced** —
no amount of care inside a lookup makes a lookup into a discovery method, which is why
§§6.3–6.4 exist and why they outrank §§1–5 wherever the two disagree.

And on the adjudicator threat, the mitigation is not independence — there was none — but
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
6. **Run the source-agnostic tests first.** §§6.3–6.4 need nothing but the corpus, cost
   about a minute of compute, and establish whether the corpus contains detectable
   coordination *at all* before any list is written. Doing them first also gives the
   indicator scan a positive control it otherwise lacks. The order in this repository is
   the historical one, not the recommended one.
7. **Never report a null without its bound and its power.** "We found none" is not a result.
   "Fewer than N at 95%, from a scan with 80% power against M" is.

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
