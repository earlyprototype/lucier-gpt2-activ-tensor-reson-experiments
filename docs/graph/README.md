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

The dropdown has a fourth entry, **Threads**, which is not a fourth graph: it is the evidence
graph re-coloured by readiness rather than by epistemic status. See
[Threads](#threads--a-fourth-view-not-a-fourth-graph) below.

```
docs/graph/
├── README.md                    ← you are here
├── viewer.html                  ← the interactive viewer (graph selector: three graphs + Threads)
├── build_evidence_graph.py      ← generator: the evidence graph
├── build_dissolution_graph.py   ← generator: the dissolution graph
├── build_isomorphism_graph.py   ← generator: the isomorphism graph
├── build_threads_report.py      ← generator: the threads report (THREADS.md + threads.json)
├── THREADS.md                   ← generated: open threads, blockers, ranked opportunities
├── _data/
│   ├── entities.json            ← evidence graph
│   ├── dissolution.json         ← dissolution graph
│   ├── isomorphism.json         ← isomorphism graph
│   ├── threads.json             ← readiness overlay for the evidence graph
│   └── visual_config.json       ← shared colours, shapes, edge styles
└── tests/
    ├── test_graphs.py           ← schema + content checks on the three JSON files
    └── smoke_test.mjs           ← drives the viewer in a real browser
```

The `_data/*.json` files are committed. They are generated, but they are also the deliverable —
the graphs must be readable by someone who has cloned the repository and not run anything.

---

## The three graphs (and one view of the first)

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

### Threads — a fourth view, not a fourth graph

**What it shows:** the evidence graph's own nodes and edges, re-coloured by **readiness** instead
of epistemic status. Not "is this true?" but "can this move, and what would it cost?". The colour
classes are read out of `_data/threads.json`, which `build_threads_report.py` writes by walking
the graph *and the working tree*:

| Class | Means |
|:---|:---|
| answered, unrecorded | the evidence is already on disk; the record has not caught up |
| unblocked | was blocked, and the blocker no longer holds on disk |
| answerable from disk | the inputs are committed, the answer has not been extracted — analysis, not a run |
| needs compute | open, and answering it takes a fresh run |
| blocked | a named blocker still holds, or cannot be settled from the tree |
| frontier | nothing downstream: no incoming `corrects` / `supersedes` / `qualifies` / `builds-on` |
| undeveloped | introduced and dropped: degree ≤ 1, or no epistemic edge at all |
| *(anything else)* | neutral and de-emphasised — the report says nothing about it |

**Why it exists:** because the two things worth acting on are the two things prose hides.

The first was a claim the record had fallen behind. H4 was, until 2026-07-31, the only hypothesis
in the graph with a `tests` edge and no verdict edge, while the notebook that edge points at,
`experiments/gpt2_small/spectral_resonance.ipynb`, carried executed output in 8 of its 14 cells,
cell 9 reading `NOT SUPPORTED … Mean |cos sim| 0.2387 … Heads > 0.9: 5 / 144`, with FINDINGS.md,
JOURNEY_MAP.md and the notebook's own status banner all still saying "not run". The operator's
ruling in issue #54 settled the disposition and closed the gap. The graph could not have
caught this on its own: it is parsed *from* those documents, so it faithfully reproduced their
staleness. Only the cross-check against the filesystem catches it, which is why the classification
is an overlay and not a graph property, and this catch is why the check exists.

The second is a shared blocker. `blocks` / `blocked-by` edges are drawn thick, dashed and purple,
and a blocker the graph has no node of its own for — an issue number, an artefact named only in
prose — is synthesised as a `database`-shaped node sized by how many claims it gates. Issue #9
becomes a five-armed hub over F10, F15 and three of the question nodes; issue #8 a two-armed hub
over H-J1 and the full J-lens build. Read linearly, F10 and F15 are eight findings apart and
nothing pairs them.

A ranked side list carries the report's `low_hanging_fruit` in the report's own order, cheapest
tier first; clicking an entry opens it and narrows the graph to it and its neighbours. **Reset
View** clears that focus.

Threads has no timeline — readiness is a statement about the tree as it stands right now — and no
**Colour by** control, because colour is the whole point of the mode. It swaps the status chip row
for a readiness chip row and keeps the type chips, so the chip bar keeps its two groups and the
desktop layout does not move.

If `_data/threads.json` is missing, Threads still draws: every node goes neutral and the legend
says to run the generator. The other three graphs are unaffected either way.

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
and the timeline scrubber work, that returning to Evidence from another graph restores its own
layout with no hierarchical leak, that the phone budget holds — and that nothing lands in the console. Its
sixteen assertions cover all four views. The fourteenth is Threads: that the overlay renders from
`_data/threads.json` — readiness colours reaching the vis DataSet, the ranked low-hanging-fruit
list populated from the report and driving the graph when a row is clicked, `blocks` / `blocked-by`
drawn as gate edges — and that none of it leaks, so after Threads → Dissolution → Evidence the
synthesised blocker hubs are gone, status colours are back, the gate edges have their evidence
style again and the graph is still clickable. If `_data/threads.json` has not been generated that
assertion fails with an actionable message and the other fifteen still run.

The fifteenth pins the ordered arrival, which is the property the front door rests on: two loads in
separate browser contexts must put every node on the same integer coordinates, nothing may move
over a 2.5s window after arrival (every node pinned, zero displacement — that is what "physics off"
means here, since `physics.enabled` deliberately stays `true`), and a round trip out to Dissolution
and back must restore those coordinates exactly rather than approximately. Assertion 13 cannot see
this: it checks the mode flags on the return path and tolerates generous positional drift, because
it was written when the arrival was still a physics cloud. Dropping the ordered layout on the way
back leaves 13 green and only 15 red. Both budgets are exact — one unit of tolerance is one unit of
physics.

The sixteenth pins the other half of the front door: that the ordered index can be *read*. It
measures the effective caption size — node font times the fitted zoom — against
`ORDERED_TARGET_LABEL_PX`, checks that at least one type lane and one date band are actually
named, and clicks through each heading to confirm the screen-space overlay does not intercept a
click meant for the node beneath it. Assertion 15 cannot see any of this either: put the grid
pitch back to the round numbers it started with and the captions collapse to ~5px, which is the
bug the readability pass existed to fix — and 15 stays green, because every node is still exactly
where it deterministically belongs. Both halves were confirmed by injecting each fault into a
throwaway copy and watching 16 go red on its own. It needs Playwright
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

### Arrival: the graph is the index

The evidence graph is the front door of this project, and it opens as an **index**, not as a
settled physics cloud. Four things follow from that, and they are the contract:

**1. The layout is computed, not simulated.** On load every node is placed by a pure function of
the data — `applyOrderedLayout()` in `viewer.html` — and pinned with vis's `fixed`. Nothing
consults a random seed, the clock, the viewport or the previous render, so two consecutive loads
put every node on the same integer coordinates. The physics solver is left switched on but has no
degrees of freedom to move: with every node pinned there is nothing to integrate, so there is no
stabilisation wobble and nothing shifts under the pointer while you are reading. Measured at
1600x950: settled 96ms after first paint, zero drift within a load, and byte-identical positions
across loads (`network.getPositions()` compared verbatim).

**2. It is ordered so it can be read.** One lane per entity type across the screen — hypotheses,
findings, questions, concepts, then runs, models, docs, artefacts, prior work — and assertion date
down it, earliest band at the top, undated last. So a lane is a chronology: the findings run in
date order, the hypotheses of the same week sit level with them, and the colour of each is its
disposition. Inside one (type, date) cell nodes wrap into a small block sorted by id, so nothing
about the arrangement is arbitrary. The **Arrangement** section at the bottom of the legend says
this on the page.

**3. Physics is a mode you enter, not the state you arrive in.** The **Explore** button releases
every node and hands the graph to the solver. **Back to the index** re-pins the coordinates
computed at load — the same integers, not merely the same shape — and re-frames. The same has to
hold on the return leg of every graph switch, because leaking a layout across a switch is exactly
the bug documented above `layoutOverride = null` in `loadAndInitialize()`: evidence -> explore /
dissolution / isomorphism / threads -> evidence all restore the identical arrangement, verified by
comparing `network.getPositions()` before and after. The ordered layout deliberately does *not* use
`layoutOverride`; it pins node coordinates, exactly as the isomorphism side layout does, so
`layoutOverride` stays `null` for the whole evidence family and cannot be the thing that leaks.

**4. It opens on the claims.** 93 assertions read as an index; 175 nodes do not. The type chips
arrive with `run`, `model`, `null-model`, `doc`, `artefact` and `prior-work` off — one click on a
chip, on **everything**, or on **Show All** brings them back. Dissolution, isomorphism and threads
open on everything they have.

Isomorphism keeps its side-pinned columns and dissolution keeps its hierarchical layered layout;
the ordered grid is for the graphs that would otherwise arrive as a cloud, which is evidence and
threads.

### Search

Search is the largest control on the page and has a line of its own above everything else, because
looking something up is what an index is for. It matches **labels, ids and description text** —
plus paths, scripts, sides, roles, dissolution tokens and, in threads mode, the report's own
wording — so `Brouwer` finds both the discovery and the concept without you knowing either id, and
`period-2` finds the twelve nodes that discuss it. The line underneath reports how much of the
index you are looking at: `93 of 175 nodes — the claims`, or `12 of 175 match "period-2"`.

### Front matter: entry points

The `Index` chip row is the contents page. Each entry sets the graph's whole state — chips, focus,
timeline cursor — rather than navigating anywhere:

| chip | id | what it sets |
| --- | --- | --- |
| by claim | `#entry-claims` | the arrival state: the four claim types, all statuses |
| by question | `#entry-questions` | the `question` type only |
| what's open | `#entry-open` | the open threads from `_data/threads.json` |
| what changed | `#entry-changed` | both ends of every `corrects` / `retires` / `supersedes` edge |
| what's blocked | `#entry-blocked` | blocked claims *and* the blockers gating them |
| everything | `#entry-everything` | all 175 nodes, every chip on |

`what's open` and `what's blocked` read `_data/threads.json` through `loadIndexOverlay()`, which is
a separate global from the threads-mode overlay so the two can never be confused. If the file is
missing the load warns (never errors) and the entry points that depend on it are hidden rather than
left to disappoint; `what changed` and `what's blocked` also draw on edges the graph already
carries, so `what's blocked` survives a missing report. Any manual chip click or a search
supersedes the entry point that set the view, and says so in the line under the search box.

### The graph selector

The **Graph** dropdown in the header swaps between Evidence, Dissolution, Isomorphism and Threads
without reloading the page. It defaults to Evidence.

- **Evidence** and **Isomorphism** render in evidence mode: status colouring, status/type filter
  chips, the timeline scrubber, the details panel and copy-evidence-chain all apply. Evidence
  arrives in the ordered index grid described above. In the isomorphism graph, claims that declare
  a `side` are pinned into columns instead — acoustic on the left, transformer on the right, shared
  down the middle — with physics still resolving `y`, so it keeps the layout it always had.
- **Dissolution** renders in its own mode: a left-to-right layered layout with one column per
  iteration band, node size scaled by how many prompts pass through that token, edge width by how
  many prompts make that transition, and colour by the terminal basin the path ends in. A **Model**
  dropdown picks between the six sweeps and a **Register** chip row filters by prompt register. The
  timeline scrubber is hidden here — dissolution is ordered by iteration count, not by date, so a
  date cursor would mean nothing. The status and type chips are hidden for the same reason.
- **Threads** renders the evidence nodes in the same ordered arrival as the evidence graph,
  coloured by readiness, with a readiness chip row in place of the status row, a ranked
  low-hanging-fruit list at the top right, and no timeline. Details, search, focus and the markdown reader work exactly as they do in
  evidence mode; the readiness block sits above the epistemic record rather than instead of it.

### The timeline, demoted

The scrubber is a good instrument at the wrong altitude: it needs you to already know which node to
watch, so it cannot be the greeting. It has moved out of the fixed 58px bar across the bottom of
the window and into the chip bar, sitting alongside the status and type filters as one more axis
you can slice by. Every id, handler and behaviour is unchanged — `#timeline`, `#timeline-play`,
`#timeline-bar`, `onTimelineInput()`, the space-bar playback shortcut and the hide-in-dissolution/
threads rule all work exactly as before. The window's bottom edge now belongs to the graph.

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
  into an unreadable stub. In Threads mode the ranked low-hanging-fruit list rides along inside it,
  since its desktop panel has nowhere to go on a phone — no fourth toolbar button, no extra chrome.
- The breadcrumb appears once you have actually navigated somewhere; its placeholder row is
  hidden.
- Tapping a node opens the details panel as a **bottom sheet** with its own scroll and a close
  button, over the full-width graph, instead of a 380px side panel that pushed the graph off the
  edge.
- The **Index** entry points and the timeline ride inside the same **Filters** disclosure as the
  chips, so the front matter and the scrubber cost nothing from the chrome budget.
- The timeline is no longer a fixed bar across the bottom, so the details bottom sheet and the
  graph both reach the bottom edge of the window.

All three disclosures are `<button aria-expanded>`, so they work from the keyboard; `Escape`
closes whichever is open, then the details sheet. Every touch target is at least 40x40 CSS px, and
the expand/collapse transitions respect `prefers-reduced-motion`.

Measured at 393x830 (a Nothing Phone 2a viewport), in all three graph modes and in Threads, with
the Filters panel both shut and open: 125px of chrome above the graph, a 705px graph — 84.9% of
the viewport — and no horizontal page scroll. (It was 659px / 79% before the timeline moved off the
bottom edge.)

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
