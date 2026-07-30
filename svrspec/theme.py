"""One design system for both surfaces: the GUI and the delivery report.

Kept in a single module so the two cannot drift into looking like different
products. Tokens first -- nothing downstream hardcodes a colour, a font size or
a spacing value.

Dark and light are both first-class. The viewer's system preference is the
default signal; a `data-theme` attribute on the root element overrides it, and
that override must win in both directions.
"""

from __future__ import annotations

TOKENS = """
:root{
  /* surfaces */
  --bg-primary:#fbfbfa; --bg-secondary:#ffffff; --bg-tertiary:#f4f4f1;
  /* text */
  --text-primary:#1c1c1a; --text-secondary:#5c5c56; --text-tertiary:#8a8a82;
  /* lines + interaction */
  --border:#e2e2dd; --accent:#2f5fa8; --accent-hover:#24487e;
  --selected-bg:#eef3fb; --selected-border:#2f5fa8;
  /* Text drawn on top of a filled accent/success/secondary block. The fills
     are dark in light mode and light in dark mode, so this flips with them. */
  --on-fill:#ffffff;
  /* meaning */
  --success:#1a7f4b; --warning:#8a5d00; --error:#b3261e;
  /* type scale */
  --fs-xs:12px; --fs-sm:14px; --fs-md:16px; --fs-lg:20px; --fs-xl:28px;
  /* spacing rhythm, 4px base */
  --s1:4px; --s2:8px; --s3:12px; --s4:16px; --s5:24px; --s6:32px; --s7:48px;
  --radius:10px; --radius-sm:6px;
  --font:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans KR",sans-serif;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){
  :root{
    --bg-primary:#161615; --bg-secondary:#1e1e1c; --bg-tertiary:#262624;
    --text-primary:#ecece8; --text-secondary:#a8a8a0; --text-tertiary:#7a7a72;
    --border:#2e2e2b; --accent:#7aa7e8; --accent-hover:#9dc0f2;
    --selected-bg:#1f2836; --selected-border:#7aa7e8;
    --on-fill:#161615;
    --success:#4ec98a; --warning:#e0b04a; --error:#ff7b72;
  }
}
:root[data-theme="dark"]{
  --bg-primary:#161615; --bg-secondary:#1e1e1c; --bg-tertiary:#262624;
  --text-primary:#ecece8; --text-secondary:#a8a8a0; --text-tertiary:#7a7a72;
  --border:#2e2e2b; --accent:#7aa7e8; --accent-hover:#9dc0f2;
  --selected-bg:#1f2836; --selected-border:#7aa7e8;
  --success:#4ec98a; --warning:#e0b04a; --error:#ff7b72;
  --on-fill:#161615;
}
:root[data-theme="light"]{
  --bg-primary:#fbfbfa; --bg-secondary:#ffffff; --bg-tertiary:#f4f4f1;
  --text-primary:#1c1c1a; --text-secondary:#5c5c56; --text-tertiary:#8a8a82;
  --border:#e2e2dd; --accent:#2f5fa8; --accent-hover:#24487e;
  --selected-bg:#eef3fb; --selected-border:#2f5fa8;
  /* Text drawn on top of a filled accent/success/secondary block. The fills
     are dark in light mode and light in dark mode, so this flips with them. */
  --on-fill:#ffffff;
  --success:#1a7f4b; --warning:#8a5d00; --error:#b3261e;
}
"""

BASE = """
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{
  background:var(--bg-primary); color:var(--text-primary);
  font:400 var(--fs-md)/1.6 var(--font);
  -webkit-font-smoothing:antialiased;
}
h1,h2,h3{margin:0; font-weight:600; letter-spacing:-0.011em}
h1{font-size:var(--fs-lg)}
h2{font-size:var(--fs-md)}
h3{font-size:var(--fs-sm); color:var(--text-secondary)}
p{margin:0}
a{color:var(--accent); text-decoration:none}
a:hover{color:var(--accent-hover); text-decoration:underline}
code{font-family:var(--mono); font-size:0.92em}

/* Reading text gets a measure; data views get the full width. */
.prose{max-width:65ch; color:var(--text-secondary); font-size:var(--fs-sm)}

.card{
  background:var(--bg-secondary); border:1px solid var(--border);
  border-radius:var(--radius); padding:var(--s4) var(--s5);
}
.label{
  font-size:var(--fs-xs); font-weight:600; letter-spacing:0.06em;
  text-transform:uppercase; color:var(--text-tertiary);
}
.muted{color:var(--text-secondary)}
.dim{color:var(--text-tertiary)}

/* Wide content scrolls inside its own box; the page body never does. */
.scroll{
  overflow-x:auto; border:1px solid var(--border);
  border-radius:var(--radius); background:var(--bg-secondary);
}
table{border-collapse:collapse; width:100%; font-size:var(--fs-sm);
  font-variant-numeric:tabular-nums}
th,td{padding:var(--s2) var(--s3); text-align:left;
  border-bottom:1px solid var(--border); white-space:nowrap}
thead th{
  position:sticky; top:0; background:var(--bg-tertiary);
  font-size:var(--fs-xs); font-weight:600; letter-spacing:0.05em;
  text-transform:uppercase; color:var(--text-tertiary);
}
tbody tr:last-child td{border-bottom:none}
td.num,th.num{text-align:right}
td.wrap{white-space:normal; min-width:280px;
  color:var(--text-secondary); font-size:var(--fs-xs)}

/* Verdicts carry a word as well as a colour: colour alone is not a signal. */
.v-pass{color:var(--success); font-weight:600}
.v-marginal{color:var(--warning); font-weight:600}
.v-fail{color:var(--error); font-weight:600}

.note{
  border:1px solid var(--border); border-left:3px solid var(--warning);
  border-radius:var(--radius-sm); background:var(--bg-secondary);
  padding:var(--s3) var(--s4); font-size:var(--fs-sm); color:var(--text-secondary);
}
.formula{
  font:var(--fs-sm)/1.7 var(--mono); background:var(--bg-tertiary);
  border:1px solid var(--border); border-radius:var(--radius-sm);
  padding:var(--s3) var(--s4); overflow-x:auto; color:var(--text-secondary);
}
ul.notes{margin:var(--s2) 0; padding-left:var(--s5)}
ul.notes li{margin:var(--s1) 0; color:var(--text-secondary); font-size:var(--fs-sm)}

:focus-visible{outline:2px solid var(--accent); outline-offset:2px; border-radius:2px}
"""


def stylesheet(extra: str = "") -> str:
    """Tokens + base, plus whatever the surface adds on top."""
    return TOKENS + BASE + extra
