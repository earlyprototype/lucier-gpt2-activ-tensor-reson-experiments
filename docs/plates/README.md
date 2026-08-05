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

## The plates

### Plates I to III, *Three Plates from a Room Made of Weights*
`i-iii_room.html` · https://claude.ai/code/artifact/d3afc583-2737-4add-b111-b69847d33aa4

Every settled state as a point cloud (1,425 runs, 21 injection levels, five
basins lit and everything else dark); the settling itself as traces across four
architectures; the sweep as a curve in loudness, growth and turn.

Sources: `experiments/nu_sweep/output/checkpoints/`,
`experiments/sink_geometry/output/trajectories.pt`,
`experiments/nu_sweep/output/angular_profile.json`.

Projection by principal components, 94.7 per cent of variance retained in
plate I, printed on the page. Dark renders as oscilloscope phosphor with
additive accumulation, light as plotter ink on paper.

### Plate IV, *Five Sentences Becoming One Shape*
`iv_bodies.html` · https://claude.ai/code/artifact/435b09eb-680a-4077-92de-de762f7cfa38

Five prompts as deforming bodies over sixty passes, in four models. The surface
is a fixed linear reading of the state, so two bodies look alike exactly when
the states do.

Source: `experiments/sink_geometry/output/trajectories.pt`.

Finding: three of four models converge, GPT-2 medium from a separation of 0.043
to 0.0002, which is five unrelated sentences arriving at one state. Pythia 410M
is the exception and diverges.

**This plate reported the opposite result twice before it reported the right
one.** Separation was first measured in the display's own coordinates, and two
different display choices gave two opposite conclusions. The reported figures
now come from the raw state vectors and never from anything drawn. The page
says so, and the warning is kept rather than tidied away.

### Plate V, *Cyanotypes of the Attention Heads*
`v_cyanotypes.html` · https://claude.ai/code/artifact/82bc3229-750a-49a8-90ce-40cd85d0aaa9

All 144 attention heads as specimen outlines built from their own singular
spectra, after Anna Atkins, *Photographs of British Algae: Cyanotype
Impressions*, 1843.

Sources: `experiments/_DATA/EXP_009/009c_spectral_data.pt`,
`experiments/gpt2_small/output_eigen_rescore/results.json`.

The plate found something unplanned: one specimen is visibly unlike the other
143, and it is layer 11 head 8. Rank one of 144 for the most closed form,
leading eigenvalue 86.7 which is the largest in the model, classified a
sign-flipper, and separately known from the attribution run to contribute most
of the per-pass turn. Three independent measurements on one square.

Limit: each head was looped in isolation, which is not what a head does inside
the working model. These are portraits of parts removed from the whole.

### Plate VI, *Six Ways of Dissolving*
`vi_river.html` · https://claude.ai/code/artifact/d16cf124-bc62-4524-b2cf-f1ed94714be9

Every word six runs pass through on the way to settling, as an alluvial flow,
including a noise control.

Source: `docs/graph/_data/dissolution.json`.

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
I to IV hold one colour per basin throughout so a basin is recognisable across
the set. Plate VI cannot, because the six runs do not share a vocabulary, so it
colours by rank of ending instead and says so.
