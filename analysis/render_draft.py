import re
import markdown

SRC = '/Users/max/Desktop/muon-feature-study/draft/muon-feature-fingerprint-draft.md'
OUT = '/Users/max/Desktop/muon-feature-study/index.html'

text = open(SRC).read()

FIGS = {'1': ('fig1_loss.png', 'Validation loss, six runs. Dotted lines are the matched-loss waypoints where checkpoints are taken.'),
        '2': ('fig2_stablerank.png', 'Stable rank of weight matrices at matched validation loss, mean over seeds and layers. Training progresses left to right.'),
        '3': ('fig3_match.png', 'Per-pair mean best feature match with 95% bootstrap confidence intervals over features.')}

def fig_sub(m):
    n = m.group(1)
    f, cap = FIGS[n]
    return f'<figure><img src="figures/{f}" alt="figure {n}"><figcaption>Figure {n}. {cap}</figcaption></figure>'

text = re.sub(r'\[FIGURE (\d): [^\]]*\]', fig_sub, text)

# shield math spans from the markdown parser (underscores etc.), restore after
math_spans = []
def shield(m):
    math_spans.append(m.group(0))
    return f'⟦MATH{len(math_spans)-1}⟧'
text = re.sub(r'\$\$.*?\$\$', shield, text, flags=re.S)
text = re.sub(r'\$[^$\n]+\$', shield, text)

body = markdown.markdown(text, extensions=['tables'])
for i, span in enumerate(math_spans):
    body = body.replace(f'⟦MATH{i}⟧', span)

html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Muon Learns the Same Features as Adam. It Packs Them Differently.</title>
<style>
  :root {{ --accent: #0e7c7b; --ink: #1d2129; --muted: #5c6470; --rule: #e4e7eb; }}
  body {{ margin: 0; background: #fcfcfb; color: var(--ink);
         font: 17px/1.65 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }}
  main {{ max-width: 720px; margin: 0 auto; padding: 56px 24px 96px; }}
  h1 {{ font-size: 1.9em; line-height: 1.2; letter-spacing: -0.015em; margin: 0 0 6px; }}
  h2 {{ font-size: 1.25em; margin: 2.2em 0 0.6em; letter-spacing: -0.01em; }}
  h1 + p em {{ color: var(--muted); font-style: normal; font-size: 0.85em; }}
  blockquote {{ margin: 1.6em 0; padding: 14px 20px; background: #f1f5f4;
               border-left: 3px solid var(--accent); border-radius: 0 6px 6px 0; font-size: 0.95em; }}
  blockquote p {{ margin: 0; }}
  table {{ border-collapse: collapse; margin: 1.4em auto; font-size: 0.88em; }}
  th, td {{ padding: 7px 14px; border-bottom: 1px solid var(--rule); text-align: left; }}
  th {{ border-bottom: 2px solid var(--ink); font-weight: 600; }}
  figure {{ margin: 2em 0; text-align: center; }}
  figure img {{ max-width: 100%; border: 1px solid var(--rule); border-radius: 6px; }}
  figcaption {{ font-size: 0.82em; color: var(--muted); margin-top: 8px; text-align: left; }}
  a {{ color: var(--accent); }}
  code {{ background: #f0f0ee; padding: 1px 5px; border-radius: 4px; font-size: 0.88em; }}
  hr {{ border: none; border-top: 1px solid var(--rule); margin: 2.5em 0; }}
  .katex-display {{ margin: 1.3em 0; }}
</style>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {{delimiters: [
    {{left: '$$', right: '$$', display: true}},
    {{left: '$', right: '$', display: false}}
  ]}})"></script>
</head>
<body><main>
{body}
</main></body>
</html>'''

open(OUT, 'w').write(html)
print('rendered', OUT)
