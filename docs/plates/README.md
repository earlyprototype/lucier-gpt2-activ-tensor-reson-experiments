# The plates

Six visual works drawn from this project's committed archives, plus the
pipeline that regenerates them. This file is the log: what each plate is, what
it is made from, where it is published, and what it got wrong before it got it
right.

Nothing here runs the model. Every number comes from data already committed, so
the whole set rebuilds on any machine with the repo and torch, in seconds:

    python3 docs/plates/build_plates.py

## How the pipeline is arranged, and why

Each plate is a **template** (`_tpl_*.html`) holding a single `__DATA__` token,
and a **payload** (`data/*.json`) holding its numbers. The build inlines one
into the other and writes a self-contained page.

Both halves are committed. That is deliberate: a plate's design can be edited
without recomputing its data, and its data can be recomputed without touching
its design. The built pages are committed too, so the repo carries the actual
artefacts and not only a recipe for them.

| file | role |
|:--|:--|
| `build_plates.py` | the whole pipeline; one builder function per plate |
| `_tpl_*.html` | design and prose, with a `__DATA__` token |
| `data/*.json` | the inlined payload, also written out for reuse |
| `i-iii_room.html` … `vi_river.html` | the built, self-contained pages |

No external assets, no CDN, no libraries. The 3D in plate IV is hand-written
WebGL; plates I to III and V and VI are hand-written canvas.

## Where the plates are published

Publication log (2026-08-27): the claude.ai artifact links this file first
recorded died with the account that published them, which is the failure mode
of hosting work under an account rather than under the work. The plates'
permanent home is now the project site, which the pages workflow
(`.github/workflows/pages.yml`) rebuilds from main on every push, straight
from the committed pages in this directory. These links stay available for as
long as the repository keeps its name and its Pages deployment stays enabled,
which is the strongest guarantee any hosting gives; the committed pages
themselves survive anything short of losing the repository:

- Plates I to III: https://earlyprototype.github.io/lucier-gpt2-activ-tensor-reson-experiments/docs/plates/i-iii_room.html
- Plate IV: https://earlyprototype.github.io/lucier-gpt2-activ-tensor-reson-experiments/docs/plates/iv_bodies.html
- Plate V: https://earlyprototype.github.io/lucier-gpt2-activ-tensor-reson-experiments/docs/plates/v_specimens.html
- Plate VI: https://earlyprototype.github.io/lucier-gpt2-activ-tensor-reson-experiments/docs/plates/vi_river.html

## The plates

### Plates I to III, *Three Plates from a Room Made of Weights*
`i-iii_room.html` · https://earlyprototype.github.io/lucier-gpt2-activ-tensor-reson-experiments/docs/plates/i-iii_room.html

Every settled state as a point cloud (1,425 runs, 21 loudness settings, five
basins lit and everything else dark); the settling itself as traces across four
architectures; the sweep as a curve in loudness, growth and turn.

Sources: `experiments/nu_sweep/output/checkpoints/`,
`experiments/sink_geometry/output/trajectories.pt`,
`experiments/nu_sweep/output/angular_profile.json`.

Projection by principal components, 94.7 per cent of variance retained in
plate I, printed on the page. Dark renders as oscilloscope phosphor with
additive accumulation, light as plotter ink on paper.

### Plate IV, *Five Sentences Becoming One Shape*
`iv_bodies.html` · https://earlyprototype.github.io/lucier-gpt2-activ-tensor-reson-experiments/docs/plates/iv_bodies.html

Five prompts as deforming bodies over sixty passes, in four models. The surface
is a fixed linear reading of the state: identical states give identical bodies.
The reading equalises its 48 components for legibility, so apparent distances
between bodies are not raw distances, which is why the reported figures come
from the raw vectors.

Source: `experiments/sink_geometry/output/trajectories.pt`.

Finding: three of four models converge, GPT-2 medium from a separation of 0.043
to 0.0002, which is five unrelated sentences arriving at one state. Pythia 410M
is the exception and diverges.

**This plate reported the opposite result twice before it reported the right
one.** Separation was first measured in the display's own coordinates, and two
different display choices gave two opposite conclusions. The reported figures
now come from the raw state vectors and never from anything drawn. The page
says so, and the warning is kept rather than tidied away.

### Plate V, *One Head, One Specimen*
`v_specimens.html` · https://earlyprototype.github.io/lucier-gpt2-activ-tensor-reson-experiments/docs/plates/v_specimens.html

All 144 attention heads as specimen outlines built from their own singular
spectra, one square per head, with the four stability classes defined in a
legend beneath the plate and any specimen enlargeable by clicking it.

Sources: `experiments/_DATA/EXP_009/009c_spectral_data.pt`,
`experiments/gpt2_small/output_eigen_rescore/results.json`.

The plate found something unplanned: one specimen is visibly unlike the other
143, and it is layer 11 head 8. Rank one of 144 for the most closed form (its
strongest stretch factor is 11.5 times its second, where the runner-up manages
3.6), a leading eigenvalue of size 86.7 and negative sign, the largest in the
model, classified a sign-flipper, and separately known from the attribution
run to carry 99 per cent of the attention side of the direction reversal that
drives the studied two-beat cycle. Three independent measurements on one
square.

Limit: each head was looped in isolation, which is not what a head does inside
the working model. These are portraits of parts removed from the whole.

**Renamed and corrected (2026-08-27).** This plate first went out under a
title borrowed from an early photographic printing process, and its page
credited that process as the working method. The claim did not hold: the
borrowed process makes its image by physical contact with the specimen, and
this plate's forms are drawn through a mapping the page itself calls a
convention, so the borrowed title and its process note were removed rather
than defended, and the page was renamed from the plate's own words. Two
numeric claims were corrected at the same time, against the committed payload:
the median leading-eigenvalue size across the 144 heads is 1.9, not "near
one", and the attribution result is 99 per cent of the attention side of the
studied cycle's direction reversal, not "most of the per-pass turn", which was
never measured per head. The renaming changed the page's filename, so the
plate's previous published address no longer resolves; the plate lives at the
address above.

### Plate VI, *Six Ways of Dissolving*
`vi_river.html` · https://earlyprototype.github.io/lucier-gpt2-activ-tensor-reson-experiments/docs/plates/vi_river.html

Every archived word six arms of runs pass through, as an alluvial flow,
including a noise control; one arm never settles at all.

Source: `docs/graph/_data/dissolution.json` (the 70 prompts per model whose
full pathways were archived, of each 125-prompt sweep; 8 runs in the deep arm,
20 in the noise arm).

The only plate with no projection at all: nodes carry their own pass number, so
nothing is distorted. GPT-2 Small ends in five words; GPT-2 Medium funnels 52
words to one letter; Pythia 410M never consolidates. The noise control also
collapses, which is why the page says in plain words that collapse by itself is
not evidence of anything semantic.

**Built twice.** The first version arranged each pass on a ring in three
dimensions and was unreadable, because on a ring strands wrap behind the cable
and cross regardless of ordering. Crossing-reduction helped and did not fix it;
the form was wrong, so it is flat.

## Standing rules for this directory

1. **Every plate states its own limits on the page**, not only here: variance
   retained, what is measured versus what is drawn, and what the plate cannot
   show.
2. **A display choice must never become a finding.** Where a rendering decision
   could change a reported number, the number is computed from the raw data
   instead, and the page says which is which. Plate IV is the cautionary case.
3. **Corrections stay visible.** When a plate reported something wrong, the
   page and this log keep the correction rather than quietly replacing the
   text.
4. **No external assets.** Self-contained pages only.
5. **Rebuild before committing** a template change, so the built page and its
   template never drift apart.
6. **The build is deterministic and must stay that way.** The projection uses
   a randomised algorithm, so the seed is fixed at the top of
   `build_plates.py`. Without it the same archive produced a different
   projection on every build, axes flipping sign and points moving, and the
   committed payloads churned. Two consecutive builds are now byte-identical;
   if that ever stops being true, something has become unreproducible.

## Provenance of the five basin colours

The palette is gas-discharge lamps, which are physical light sources rather
than a picked spectrum: sodium amber, mercury cyan, neon, argon, xenon. Plates
I and III hold one colour per basin so a basin is recognisable across the
pair. Plates II and IV reuse the same five lamps to label the five prompts
instead, and each page says so in its reading text. Plate VI cannot hold
either scheme, because the six arms do not share a vocabulary, so it colours
by rank of ending instead and says so.
