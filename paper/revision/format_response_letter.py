"""Normalize the grid table in response_letter.md.

Pandoc grid tables only parse when every "|" sits exactly under a "+" in the
rule lines, so hand-editing a cell shifts the pipes and the table renders wrong
or not at all. This reads whatever is in the file, recovers the cell text by
splitting on "|", and rewrites the table with the pipes aligned again.

response_letter.md is the source of truth: edit it freely, then run this.

Run:  python3 format_response_letter.py && pandoc response_letter.md \
          -o response_letter.pdf --pdf-engine=xelatex
"""
import sys
import textwrap

PATH = "response_letter.md"
W = [6, 12, 29, 60, 14]        # column widths in characters


def rule(sep="-"):
    return "+" + "+".join(w * sep for w in W) + "+"


def emit(cells):
    cols = [textwrap.wrap(c, w - 2, break_on_hyphens=False) or [""]
            for c, w in zip(cells, W)]
    out = []
    for i in range(max(len(c) for c in cols)):
        line = "|"
        for col, w in zip(cols, W):
            line += " " + (col[i] if i < len(col) else "").ljust(w - 2) + " |"
        out.append(line)
    return out


def main():
    lines = open(PATH).read().split("\n")
    start = next(i for i, l in enumerate(lines)
                 if l.startswith("+") and set(l) <= set("+-="))

    blocks, cur, header_at = [], [], None
    for l in lines[start:]:
        if l.startswith("+") and set(l) <= set("+-="):
            if cur:
                blocks.append(cur)
                cur = []
            if "=" in l:
                header_at = len(blocks)
            continue
        if l.startswith("|"):
            cur.append(l)
        elif l.strip() and not cur:
            continue          # trailing prose after the table, if any
    if cur:
        blocks.append(cur)

    rows = []
    for b, block in enumerate(blocks):
        cells = [[] for _ in W]
        for l in block:
            parts = l.split("|")
            if len(parts) != len(W) + 2:
                sys.exit(f"line splits into {len(parts)-2} cells, expected {len(W)}:\n  {l}")
            for k, p in enumerate(parts[1:-1]):
                if p.strip():
                    cells[k].append(p.strip())
        rows.append([" ".join(c) for c in cells])

    out = [rule()]
    for i, r in enumerate(rows):
        out += emit(r)
        out.append(rule("=") if header_at is not None and i + 1 == header_at else rule())

    text = "\n".join(lines[:start] + out) + "\n"
    for ch in "–—":
        if ch in text:
            sys.exit("en or em dash in the letter")
    open(PATH, "w").write(text)
    print(f"normalized {len(rows)-1} rows ({len(W)} columns)")


if __name__ == "__main__":
    main()
