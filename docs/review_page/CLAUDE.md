# CLAUDE.md -- review pages


Fits are reviewed by eye on published claude.ai Artifact pages, one HTML file
each under `docs/review_page/` (committed), loading a data bundle from beside it
(gitignored, rebuilt by an exporter). All share one shell: an order/window list
with quality filters, a uPlot main panel with a residual strip, a species or
facts rail, and good/suspect/reject marks kept in the artifact's `db`
capability. The URLs are in the user's memory, not here.

| page | exporter | bundle |
|---|---|---|
| `index.html` (Kitt Peak), `niratl.html`, `niratl_clean.html` (the width-scale, clean-pixel refit, with per-page flatness badges from `docs/niratl_correction_score.json`), `fts1983.html` | `export_solar_review.py` (`--shard`), `merge_review_bundle.py` | `data/review*/` |
| `wallace2011.html` -- Kitt Peak flux atlas #2, regions 1-3 (a region switch; region 3 scored at T < 0.99), with the authors' transmission and corrected flux and per-window flatness badges from the scores in `data/corrected/solar/wallace2011/`; a top axis gives the row in `sptr.regN` | `export_solar_review.py` with both region summaries | `data/review_wallace2011_pub` |
| `baker2020.html` -- Baker et al. 2020's Goettingen telluric spectra, flat-Sun fits: 146 windows x 53 spectra, a spectrum switch (h/l), every spectrum's residual stacked by air mass and their median over the residual strip (r, computed in the page), the analysed features' ratios as ticks and a list beside the solar fits'; marks in `reviews/baker2020`, keyed by window v1 | `export_baker_review.py` (numpy, ~2 min; one chunk per window, 70 MB, so two publishes) | `data/review_baker2020/` |
| `igrins.html` -- A0V standards, DCT 2018-12-20 | `export_igrins_review.py --shard I/N`, then `--merge` | `data/review_igrins/` |
| `igrins_targets.html` -- science corrections, same night | `export_igrins_target_review.py` (numpy only, 30 s) | `data/review_igrins_targets/` |
| `arcturus.html` -- Arcturus atlas, full-coverage record, 598 page-epochs | `export_arcturus_review.py --shard I/N`, then `--merge` (`--no-species`: total only, numpy) | `data/review_arcturus/` |

Publish with the Artifact tool, mapping `manifest.json` and every chunk in the
bundle through `files` (`contentType: application/wasm` -- binary chunks are
published as `.wasm` because that is a type the host serves; the solar bundles
write them as `chunk_NNN.bin`, so map `chunk_NNN.wasm` from `chunk_NNN.bin`) and
declaring `capabilities: {db: {}}` on the first publish. Republishing from a new
conversation: read the page and list its files (`scope: "files"`) first, or the
publish is refused; check the chunk count matches, since files left out are kept.
The marks survive a republish -- they live in the page's `db`, keyed by window.
The solar bundles: `export_solar_review.py --summary ... --npz-dir <the run's output
directory>` into a raw directory, then `merge_review_bundle.py <raw> --output <pub>`;
Kitt Peak is `data/review` (raw `data/review_0`), niratl, its refit and June 1983 are
`data/review_{niratl_clean,1983}_pub`. `niratl.html` shows the run superseded on
2026-10-18 (its bundle is in `data/corrected/superseded_pre_widthscale/review/`);
`niratl_clean.html` shows the promoted one. A bundle stays under the 64 MB a publish may
carry: 36 MB for the A0V page, 33 MB for the targets page, 26.6 MB (20
chunks) for Arcturus, whose export takes ~22 min as four CPU shards.

What the exporters check, and the traps that shaped them:

- `export_igrins_review.py` rebuilds every order through
  `tellurix_igrins.standard.build_order_context` to split the transmission by species,
  and refuses an order-frame whose species do not multiply back to the total
  (6e-16) or whose total misses the cached `transmission` (2e-16) -- the second is
  what proves the summary's parameters made the cached arrays.
- `export_arcturus_review.py` reads the full-coverage record
  (`data/corrected/atlas/arcturus_atlas.h5`), never the trimmed
  `docs/arcturus_atlas_summary.json`. It rebuilds each page *window* from the
  record (one compile serves both epochs) and refuses a row whose species miss the
  total (3e-16) or whose total misses the cached `transmission` (tolerance 2e-6;
  measured 1e-8 to 3e-8 on CPU, all of it on masked saturated pixels). It ships
  the chunks as `chunk_NNN.wasm` already, so they map one to one. Marks are one
  document, `reviews/arcturus_atlas`, keyed `page:epoch`. The top axis is the
  atlas row index, matched by wavenumber against the page file; the axis is
  float32 offsets because the atlas prints wavenumbers to 0.01 cm-1 (steps vary
  by 3%). The atlas authors' telluric column is NaN above the fit's 1.05 ceiling.
- **Wallace 2011's rows are not evenly spaced** (0.0068-0.0079 cm-1 steps in one window, gaps near the
  B-band), and four region 2 windows sit up to 3 pixels off a uniform axis: the exporter ships those
  windows' own wavenumbers as float32 offsets (`axis_offset`), and the page uses them when present.
- **The solar manifest names the continua.** `export_solar_review.py` writes each
  window's `continua` -- the between-line loss of the MT_CKD water continuum and O2's
  collision-induced bands, in percent -- because the bundle ships them as one curve and
  that curve is what lifts the corrected spectrum above the observed. niratl shows it as a
  badge and a note; O2's 1.06 um band goes as the fitted O2 column squared, 0.01-2.7%
  across neighbouring windows.
- **Keys press buttons.** Each trace carries a `hotkey` (o observed, m model, s stellar,
  c continuum, t corrected; the rest per page, shown as a `<kbd>` in the legend) and
  1/2/3 are good/suspect/reject. The handler clicks the button the key names rather
  than reimplementing it, so a key cannot drift from what a click does. Keep j/k/h/l
  and the arrows free for stepping. The verdict panel sits at the foot of the left rail.
- **Keep every array aligned to its element size.** A uint16 view must start at an even
  byte and the float32 axis at a multiple of 4. The bit-packed mask is ceil(n / 8) bytes,
  and Wallace 2011's uneven pixel counts made it odd, which put every window after it in
  that chunk at an odd offset ("start offset of Uint16Array should be a multiple of 2", the
  plot frozen on the last good window). `export_solar_review.py` pads before the axis and
  after the mask; check `offset % 2` over a manifest after changing the layout.
- **Hand uPlot `null`, never `NaN`.** The pages blank masked pixels with NaN, which
  uPlot does not treat as a gap: a zoom whose range held one made that series'
  min and max NaN, the auto-ranged flux scale lost its extent, and the chart drew
  axes and nothing else. Every page passes its data through `gaps()` (NaN to null)
  at construction; the readouts keep the NaN arrays. With nulls and `spanGaps:
  false`, lines also break across masked pixels instead of bridging them.
- **Blank everything outside the fit mask before quantizing.** Outside the mask
  the degree-9 continuum is extrapolated to ~5e9 at an order's ends; left in, it
  set the uint16 scale, and the page's autoscale, and the spectra drew as a flat
  line. Inside the mask hot pixels still reach 10-40x the continuum, so the
  pages scale the flux axis from percentiles of the reliable pixels.
- IGRINS pixels are not uniform in wavenumber, so each entry (or, for the
  targets, each order, after checking every frame shares it) ships its own axis
  and detector columns. The top axis of every spectrum is the detector column.
- The targets page shows the corrected spectrum, the convolved operator divided
  out, the deepest telluric lines as ticks, each neighbouring order interpolated
  onto the overlap, another calibration as an overlay (full, target model,
  three-standard sparse), and the **PLP's own A0V-divided spectrum**
  (`SPEC_DIVIDE_CONT` of `spec_a0v.fits`, fetched with
  `download_rrisa_standard.py --extra spec_a0v.fits`; its arrays index directly by
  our detector columns, wavelengths agree exactly). The PLP divided by its own
  choice of standard -- HR 1558 (our frame 55) for the four Taurus targets,
  HD 53205 for GJ 281, 85 Gem (not one of ours) for YY Gem -- so a disagreement
  can be the standard as much as the method.
