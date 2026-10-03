# CLAUDE.md -- review pages


Fits are reviewed by eye on published claude.ai Artifact pages, one HTML file
each under `docs/review_page/` (committed), loading a data bundle from beside it
(gitignored, rebuilt by an exporter). All share one shell: an order/window list
with quality filters, a uPlot main panel with a residual strip, a species or
facts rail, and good/suspect/reject marks kept in the artifact's `db`
capability. The URLs are in the user's memory, not here.

| page | exporter | bundle |
|---|---|---|
| `index.html` (Kitt Peak), `niratl.html`, `fts1983.html` | `export_solar_review.py` (`--shard`), `merge_review_bundle.py` | `data/review*/` |
| `igrins.html` -- A0V standards, DCT 2018-12-20 | `export_igrins_review.py --shard I/N`, then `--merge` | `data/review_igrins/` |
| `igrins_targets.html` -- science corrections, same night | `export_igrins_target_review.py` (numpy only, 30 s) | `data/review_igrins_targets/` |

Publish with the Artifact tool, mapping `manifest.json` and every `*.wasm` in the
bundle through `files` (`contentType: application/wasm` -- binary chunks are
`.wasm` because that is a type the host serves) and declaring `capabilities:
{db: {}}` on the first publish. A bundle stays under the 64 MB a publish may
carry: 36 MB for the A0V page, 33 MB for the targets page.

What the exporters check, and the traps that shaped them:

- `export_igrins_review.py` rebuilds every order through
  `tellurix_igrins.standard.build_order_context` to split the transmission by species,
  and refuses an order-frame whose species do not multiply back to the total
  (6e-16) or whose total misses the cached `transmission` (2e-16) -- the second is
  what proves the summary's parameters made the cached arrays.
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
