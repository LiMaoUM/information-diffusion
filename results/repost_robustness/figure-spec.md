# Figure Spec: repost reconstruction robustness (DRAFT, pending Mao's confirmation)

- **Figure ID / paper**: R&R appendix figure, ICWSM 2027 revision (roadmap P1-1).
- **Conclusion**: The cross-platform similarity of repost cascade scaling is not
  an artifact of the reconstruction rule; every alternative rule yields a
  platform-by-size interaction near zero, far smaller than the reply-cascade
  divergence, and the published rule is the most conservative case.
- **Panels**: A: b3 for log breadth by reconstruction rule (point + 95% CI),
  with the random-rule ensemble as a band and the reply-cascade b3 as a
  reference marker. B: same for log depth.
- **Axes**: x = platform-by-size interaction b3 (log10 units, linear scale);
  y = reconstruction rule (categorical). Zero line marked.
- **Encoding**: forest plot (dot + CI whiskers); shaded vertical band for the
  random ensemble range; distinct marker for the reply reference.
- **Style**: draft at 300 DPI PNG for review; final via nature-figure skill at
  600 DPI PDF + PNG after spec confirmation. Colorblind-safe (blue/orange/gray).
- **Exclusions**: no per-draw points for all 40 random draws (band only), no
  titles inside panels beyond A/B tags, no gridlines except the zero line.

QA (draft): run checklist from ~/.claude/templates/figure-spec.md before showing.
