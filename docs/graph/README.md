# The Graphs

*Three views of the same study, drawn rather than written.*

The prose in `docs/` is linear: FINDINGS.md runs F1 to F17, JOURNEY_MAP.md runs Phase 0 to
Phase 5, and a reader arrives at the end of either one holding a list. But the study is not a
list. It is a small tangle of claims that support, qualify, correct and occasionally kill each
other, and the shape of that tangle is the thing the markdown cannot show you. These three
graphs draw it.

They are generated files. Nothing is authored in the browser: each `build_*.py` script writes a
JSON document into `_data/`, and `viewer.html` renders any of the three — pick one from the
**Graph** dropdown in the header. Regenerate freely — the scripts take no arguments and are
idempotent.

```
docs/graph/
├── README.md                    ← you are here
├── viewer.html                  ← the interactive viewer (graph selector: all three)
├── build_evidence_graph.py      ← generator: the evidence graph
├── build_dissolution_graph.py   ← generator: the dissolution graph
├── build_isomorphism_graph.py   ← generator: the isomorphism graph
├── _data/
│   ├── entities.json            ← evidence graph
│   ├── dissolution.json         ← dissolution graph
│   ├── isomorphism.json         ← isomorphism graph
│   └── visual_config.json       ← shared colours, shapes, edge styles
└── tests/
    ├── test_graphs.py           ← schema + content checks on the three JSON files
    └── smoke_test.mjs           ← drives the viewer in a real browser
```

The `_data/*.json` files are committed. They are generated, but they are also the deliverable —
the graphs must be readable by someone who has cloned the repository and not run anything.

---

## The three graphs

### 1. The evidence graph — `entities.json`

**What it shows:** every hypothesis, finding and concept in the study as a node, every run and
model that bears on them as a node, and the epistemic relations between them as edges. H-fingerprint
sits there in refuted red with the cross-model sweep and the null model pointing arrows at it.
F9 sits there resolving F2. The Brouwer claim sits there with a correction attached.

**Why it exists:** because the single most important fact about this project is that its founding
hypothesis was refuted by its own validation programme, and a list of findings is a poor way to
show that. In the graph the refutation is a shape: a claim with three arrows converging on it and
nothing leading out. You can see which findings are load-bearing and which are decoration by
counting edges.

### 2. The dissolution graph — `dissolution.json`

**What it shows:** the descent itself. Prompts on the left, terminal basins on the right, and
between them the waypoint tokens each prompt passes through, banded by iteration —
`ash → Canad → Ag → FT → capit → injustice → Rousse → prolet`, drawn as a flow rather than
recited as a sentence. It is built per model, so GPT-2 Small's five-way fan and GPT-2 Medium's
single funnel into `D` can be put side by side.

**Why it exists:** the dissolution tables in `experiments/*/output/dissolution_pathways.md` are
wide, repetitive and almost unreadable at 125 rows. As a layered flow the convergence is obvious
at a glance, and so is the thing the tables hide — how *early* the prompts stop being distinct
from one another. Read the provenance note in that script's docstring before quoting any share
from it: the published tables print only the first ~10 prompts of each register, so the graph is
built from a visible subset, and its basin shares are not the convergence-gated shares quoted in
the README.

### 3. The isomorphism graph — `isomorphism.json`

**What it shows:** Lucier's apparatus down one side, the transformer's down the other, joined by
`analogous-to` edges whose descriptions state the shared mathematical role. Room ↔ weight
matrices, audio signal ↔ residual stream tensor, tape recorder ↔ TransformerLens hook, pure drone
↔ terminal token. Then the more interesting half: `breaks-down-at` edges running from the acoustic
concepts into the five nonlinearities (LayerNorm, softmax attention, dynamically recomputed QKV,
GeLU MLP, residual connections) and into the three findings that stop the analogy being an
identity.

**Why it exists:** because an analogy that is never tested is decoration, and this one has been
tested to destruction in three specific places. A room mode is a fixed point; `Divine` is an exact
period-2 limit cycle, home only every second pass (F9). A room's modes are a property of the room;
GPT-2 Small's basins are not a property of the weights, since noise driven through the same weights
finds eighteen different ones (F4). And the argument that promoted the analogy to a theorem — every
normalised transformer must have basins, by Brouwer — is in the graph too, in correction blue, with
the reason it fell over attached to it: the L2 shell is a sphere, and a sphere is not convex
(JOURNEY_MAP Key Discovery 11, corrected 2026-07-23).

The isomorphism graph is hand-authored rather than parsed. Every node carries a `doc_ref` naming
the passage it transcribes, and the generator refuses to write a node whose `doc_ref` does not
resolve to a registered source. Nothing in it originates in the graph.

---

## The vocabularies

Two of the three files share one schema. `entities.json` and `isomorphism.json` are both
evidence graphs: the top level is `metadata`, `claims`, `runs`, `sources`, `relationships`,
and the viewer renders them the same way.

`dissolution.json` is the exception. It is not a claim graph at all — it is six token-flow
graphs, one per model, so its top level is `metadata` and `models`, and each model carries its
own `nodes` and `edges` keyed by iteration rather than by claim. The viewer detects the
difference and switches to a separate render mode for it (see **Viewing** below). Everything in
this section describes the evidence schema, which `dissolution.json` does not use.

### Node types and shapes

| Array | `type` | Shape | What it is |
|:---|:---|:---|:---|
| `claims` | `hypothesis` | diamond | Something asserted and testable |
| `claims` | `finding` | dot | Something established by a run |
| `claims` | `concept` | hexagon | A defined object the study reasons with |
| `runs` | `run` | square | An experiment |
| `runs` | `model` | triangle | A model the experiments were run on |
| `runs` | `null-model` | triangle-down | A control: same procedure, cause removed |
| `sources` | `doc` | box | A document in this repository |
| `sources` | `artefact` | ellipse | A generated output (figure, report, tensor file) |
| `sources` | `prior-work` | star | Something outside the repository |

### Claim status

Status is the colour of a node. It says how the claim stands *now*, not how it was born.

| Status | Colour | Meaning |
|:---|:---|:---|
| `supported` | `#2E7D5B` | Evidence points at it and nothing has knocked it down |
| `refuted` | `#B3423F` | Evidence points against it; it does not stand |
| `not-supported` | `#8F5A57` | A null result: the evidence failed to back it, without contradicting it |
| `qualified` | `#B9812F` | It stands, but not in the form first stated |
| `retired` | `#8A8F94` | Withdrawn — not disproved so much as no longer claimed |
| `corrected` | `#5B7DB1` | The claim was wrong in a specific, identified way, and the correction is recorded |
| `open` | `#6B4C8A` | Asked, not answered |
| `untested` | `#9AA3A8` | Designed, never run |

The table is ordered as a severity gradient, and the three negative values are not
interchangeable. `refuted` is for evidence that *contradicts* a claim (H-supp: "Refuted with
the opposite sign"). `not-supported` is for a claim the evidence simply failed to back —
H-J1's disposition literally opens "**Not supported at pilot confidence**", and before this
value existed it had to be filed as `qualified`, which put a null result in the same bucket as
genuine partial support. `qualified` is for a claim that survives in a narrowed or mixed form,
including the mixed case where one half fails and another is strengthened — that is why H3
("Weakened further at close; **coherence half upgraded**") stays `qualified` and is *not* a
null: its semantic-coherence half gained permutation support (F8), even as its corpus-causal
half failed cross-model (F3).

A status outside this vocabulary is not dropped from the interface: the viewer appends any
unrecognised value it finds in the data to the legend and filter chips, and colours it with the
default grey `#7f8c8d`.

### Edge types

Three families. The first carries epistemic force and has a direction that matters; the second is
plumbing; the third is resemblance and makes no claim about truth.

| Family | Type | Style | Says |
|:---|:---|:---|:---|
| Epistemic | `supports` | solid green, arrow | A gives reason to believe B |
| | `refutes` | dashed red, arrow | A is evidence against B |
| | `qualifies` | dashed amber, arrow | A narrows the conditions under which B holds |
| | `corrects` | dotted blue, arrow | A identifies a specific error in B |
| | `retires` | dotted grey, arrow | A is the reason B was withdrawn |
| | `supersedes` | dotted blue, arrow | A replaces B |
| | `tests` | solid purple, arrow | A was run in order to decide B |
| Structural | `produced-by` | thin grey, arrow | B made A |
| | `run-on` | thin grey, arrow | A was executed against model B |
| | `evidenced-by` | thin grey, arrow | The numbers behind A live in B |
| | `documented-in` | thin grey, arrow | A is written down in B |
| Associative | `analogous-to` | dashed blue, no arrow | A and B occupy the same structural position |
| | `breaks-down-at` | dashed blue, no arrow | The analogy at A fails here, for a stated reason |
| | `builds-on` | thin grey, arrow | A takes B as a starting point |
| | `cites` | thin grey, arrow | A references B |
| | `relates-to` | thin grey, arrow | A bears on B in a way the description names |

Every edge carries a `description` saying *why*, and the rule the generators enforce is that no
description may be filler. If an edge cannot be given a specific sentence, it should not be drawn.

---

## Regenerating

From `docs/graph/`:

```bash
python3 build_evidence_graph.py
python3 build_dissolution_graph.py
python3 build_isomorphism_graph.py
```

No arguments, no dependencies beyond the standard library, no network. Each script prints its node
and edge counts, validates that every edge endpoint resolves to a real node, and writes one file
into `_data/`. A non-zero exit means the JSON was not written; the failing check is printed above
the exit. The dissolution generator additionally prints its parsed basin shares next to the
documented ones from FINDINGS.md, so any drift between the graph and the record shows up in the
console rather than in the picture.

## Checking

Two suites, and they check different things. From the repository root:

```bash
python3 -m pytest docs/graph/tests/ -q     # the data
node docs/graph/tests/smoke_test.mjs       # the page
```

`test_graphs.py` reads the three JSON files and checks the things a generator can get wrong
quietly: that every edge endpoint resolves, that no node id is duplicated, that statuses and edge
types are in vocabulary, that no edge description is empty, and that every `doc_ref` and `path`
pointing into the repository actually exists on disk. It also pins a handful of facts against
FINDINGS.md — that `h-fingerprint` is refuted by exactly F3 and F4, that F9 corrects F2 — so that
a regeneration cannot silently rewrite the epistemic record. It skips rather than fails if
`_data/` has not been built.

`smoke_test.mjs` serves the directory, drives `viewer.html` in headless Chromium and asserts that
all three graphs actually draw, that the model switch re-renders, that search and the details panel
and the timeline scrubber work, and that nothing lands in the console. It needs Playwright
(`npm i -D playwright`, or set `PLAYWRIGHT_PATH`); the two CDN scripts are served from a local
mirror so a bad day at unpkg cannot turn into a red test.

## Viewing

The viewer loads its data over `fetch`, so it needs a server; opening `viewer.html` from the
filesystem will give you an empty canvas and a CORS complaint in the console.

Serve from the **repository root**, not from `docs/graph/`. The `doc_ref` values in the data are
repo-relative (`docs/FINDINGS.md#f9-...`), so the in-page markdown reader can only open them if
the repo root is what is being served:

```bash
cd <repo root>
python3 -m http.server 8000
# then open http://localhost:8000/docs/graph/viewer.html
```

Serving `docs/graph/` directly still draws all three graphs; only the markdown panel goes dark,
because the documents sit above the server root.

Everything runs locally and nothing is uploaded. The page does make two external requests: it
pulls `vis-network` from unpkg and `marked` from jsDelivr at load time. Offline, the graph will
not draw — vendor those two scripts locally if you need it to.

### The graph selector

The **Graph** dropdown in the header swaps between Evidence, Dissolution and Isomorphism without
reloading the page. It defaults to Evidence.

- **Evidence** and **Isomorphism** render in evidence mode: status colouring, status/type filter
  chips, the timeline scrubber, the details panel and copy-evidence-chain all apply. In the
  isomorphism graph, claims that declare a `side` are pinned into columns — acoustic on the left,
  transformer on the right, shared down the middle.
- **Dissolution** renders in its own mode: a left-to-right layered layout with one column per
  iteration band, node size scaled by how many prompts pass through that token, edge width by how
  many prompts make that transition, and colour by the terminal basin the path ends in. A **Model**
  dropdown picks between the six sweeps and a **Register** chip row filters by prompt register. The
  timeline scrubber is hidden here — dissolution is ordered by iteration count, not by date, so a
  date cursor would mean nothing. The status and type chips are hidden for the same reason.

### On a phone

At 768 CSS px and below the same page rearranges itself so the graph gets the screen instead of
the controls. Nothing is removed — every control is still reachable, and the desktop layout at
769px and above is byte-for-byte what it always was.

- The header keeps only the **Graph** selector and the search box. The title truncates.
- **Filters** opens the status/type chips — or, in dissolution mode, the register and basin chips
  — as a panel that *overlays* the graph rather than pushing it down. Closed by default. The
  button reports how much is being hidden: plain `Filters` when every chip is on, `Filters (14 of
  16)` when it is not, so a shut panel can never leave you wondering where the nodes went.
- **More** holds the secondary controls: Show All, Reset View, Refresh Data, Chat with AI, the
  **Colour by** select and — in dissolution mode — the **Model** select.
- **Legend** opens the legend as an overlay; it is closed by default rather than being squeezed
  into an unreadable stub.
- The breadcrumb appears once you have actually navigated somewhere; its placeholder row is
  hidden.
- Tapping a node opens the details panel as a **bottom sheet** with its own scroll and a close
  button, over the full-width graph, instead of a 380px side panel that pushed the graph off the
  edge.
- The timeline bar stays where it is, compacted from 58px to 46px.

All three disclosures are `<button aria-expanded>`, so they work from the keyboard; `Escape`
closes whichever is open, then the details sheet. Every touch target is at least 40x40 CSS px, and
the expand/collapse transitions respect `prefers-reduced-motion`.

Measured at 393x830 (a Nothing Phone 2a viewport), in all three graph modes: 125px of chrome above
the graph, a 659px graph — 79% of the viewport — and no horizontal page scroll.

### Run order

`_data/visual_config.json` — the shared colours, shapes and edge styles all three graphs read — is
written by `build_evidence_graph.py`. Run that one at least once before viewing; the other two
generators do not write it. The viewer falls back to its own built-in canonical palette if the file
is missing, so a stale or absent `visual_config.json` degrades the colours rather than breaking
the page.

The viewer is built with the **evidence** template from
[knowledge-graph-kit](https://github.com/earlyprototype/knowledge-graph-kit) — a small toolkit for
rendering claim-and-evidence structures as navigable graphs. The template supplies the layout, the
status colouring, the timeline scrubber and the filtering; the evidence schema above is its
contract, which is why the evidence and isomorphism graphs pour into the same page unchanged. The
dissolution render mode is a local addition on top of the template.

---

## How to read a refutation trail

This is the thing worth learning, because it is what the graphs are for.

Find a red node — `H-fingerprint` is the one to start with. Red means the claim does not stand.
Now follow the arrows *into* it: each one is an edge of type `refutes` or `qualifies`, and each one
comes from a finding, which in turn hangs off a run. Click through and you get the whole trail in
four hops:

> **H-fingerprint** — *basin profiles read training bias from any model* — is refuted by **F3**,
> which is evidenced by **run 2, the cross-model sweeps**, run on **GPT-2 Medium**, which shares
> GPT-2 Small's corpus and produces no semantic basins at all. And it is refuted again, from a
> different direction, by **F4**, evidenced by **run 3, the null model**, where noise through the
> same weights finds eighteen attractors none of which are the five.

Two independent trails converging on one dead claim. That is what a refutation looks like when it
is done properly, and it is legible in the graph in about five seconds.

Three habits make the rest of it readable:

- **Direction is the argument.** An arrow always runs from the evidence to the claim, never the
  other way. If you are looking at a node and want to know why anyone believes it, read the arrows
  coming in. If you want to know what depends on it, read the arrows going out.
- **Grey is not weak.** A `documented-in` edge carries no epistemic force at all — it only says
  where a thing is written down. Filter the structural family out when you are auditing an
  argument, and back in when you are trying to find the file.
- **Blue is the interesting colour.** `corrected` nodes are places the project was wrong in a way
  it could name. There are more of them than most write-ups admit to, and they are kept on purpose:
  a corrected claim with its correction attached is worth more than a claim quietly deleted.

## What this shows that the markdown doesn't

The documents are honest, but they are sequential, and three things are invisible in a sequence.

**Load-bearing versus decorative.** In FINDINGS.md every finding gets a heading and a paragraph,
which gives them all the same apparent weight. In the graph they get the weight their connections
earn: some have arrows running everywhere and some have two, and the difference is visible before
you have read a word.

**Convergent refutation.** FINDINGS.md tells you the fingerprint hypothesis was refuted, then tells
you about the null model several pages later. You have to hold both in your head to notice that
they are two independent attacks on the same claim from different directions. The graph puts both
arrows on the same node and the point makes itself.

**The corrections as a class.** Retired discovery 9, retired discovery 10, corrected discovery 11
and the withdrawn `Divine` phrasing are scattered across four documents and eight months. Colour
them all and they become one visible feature of the project rather than four footnotes — and the
pattern in them is legible: the errors were nearly all over-claims of generality, and each was
caught by asking the same question, *is this a property of the room, or of what we played into it?*

**And, in the isomorphism graph, the shape of an analogy under strain.** ISOMORPHISM.md has a table
of eight correspondences and a table of five nonlinearities, on separate pages, and a reader must
do the work of connecting them. Drawn, the two tables are one object: nine pairings — the table's
eight, plus the room's friction against the L2 rescale — with nine dashed lines running out of the
acoustic side into the places the pairing stops holding. The
correspondence and its limits arrive at the same time, which is the only honest way to present
either.
