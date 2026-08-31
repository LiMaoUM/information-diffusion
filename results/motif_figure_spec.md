# Figure Spec: motif z-score figures (revision fix)

- **Figure ID / paper**: Figures 3 (chain) and 4 (star), ICWSM 2027 revision.
- **Conclusion**: Cross-cutting reply motifs are over-represented on both
  platforms relative to a null model, but with different profiles: Bluesky
  spreads over-representation across many cross-cutting pairings, Truth Social
  concentrates it in Left-Right confrontations and in root-focused stars.
- **What changes in this revision**: ONLY the placement of the four motif
  family labels (Aligned, End shift, Mid shift, Diverse). They are currently
  drawn as semi-transparent watermarks inside the plot area, where they sit on
  top of the data points; R1 reported labels overlapping content. They move to
  a header band above the axes. Data, ordering, colors, markers, glyph row,
  legend, and axis ranges are unchanged.
- **Panels**: single panel per figure; 27 motifs on x, z-score on y.
- **Axes**: x = motif identity, shown as a vertical three-node glyph with
  L/C/R labels (no text tick labels); y = z-score against the null model,
  linear scale, zero line drawn.
- **Encoding**: paired dot plot. Circle = Truth Social (#CD3800), triangle =
  Bluesky (#4F94CD), gray connector between the pair, gray marker when
  |z| < 3. Background band per motif family.
- **Style**: matplotlib, existing palette retained (colorblind-safe
  orange/blue pairing), 800 dpi PNG as before.
- **Exclusions**: no in-plot watermark text; no significance stars; no change
  to the underlying z-scores.

## QA checklist
- [ ] No overlapping elements (labels, markers, glyphs, legend)
- [ ] Axis label present; zero line visible
- [ ] Legend inside bounds, not covering data
- [ ] Output DPI verified on the file
- [ ] Family labels legible and unambiguously aligned to their band
- [ ] Numbers match the z-score source files (spot-check one motif)
