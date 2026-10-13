"""Portable HTML reports with embedded charts and evaluation provenance."""
import base64
import html
import json
import math
from pathlib import Path


def write_report(output, metrics, charts, provenance):
    output = Path(output)
    cards = []
    for key, value in metrics.items():
        display = f"{value:.3f}" if isinstance(value, (int, float)) and math.isfinite(value) else "Unavailable"
        cards.append(f'<div class="card"><span>{html.escape(key)}</span><strong>{display}</strong></div>')
    figures = []
    for title, path in charts:
        encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        figures.append(f'<section><h2>{html.escape(title)}</h2><img src="data:image/png;base64,{encoded}" alt="{html.escape(title)}"></section>')
    label = "Synthetic offline demonstration" if provenance.get("synthetic") else "Chronological holdout research"
    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Regime Research · Holdout Report</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#101820;color:#e5eaf0;font:16px/1.6 system-ui,sans-serif}}
main{{max-width:1120px;margin:auto;padding:64px 28px}}.eyebrow{{color:#75d6bf;letter-spacing:.12em;text-transform:uppercase;font-size:12px}}
h1{{font-size:clamp(32px,5vw,52px);line-height:1.1;margin:16px 0}}p{{color:#aab8c6;max-width:800px}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin:36px 0}}
.card{{background:#1c2a36;padding:24px;border:1px solid #304554;border-radius:12px}}.card span{{display:block;color:#aab8c6;font-size:13px}}strong{{font-size:28px}}
section{{background:#fff;color:#182832;padding:24px;border-radius:12px;margin:20px 0}}h2{{font-size:20px;margin:0 0 16px}}img{{width:100%;height:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#1c2a36;padding:24px;border-radius:12px;font-size:13px}}
</style></head><body><main><div class="eyebrow">Regime Research / {label}</div>
<h1>Regimes, risk, and the held-out path.</h1>
<p>Model and feature scaling are fitted on the training prefix. State probabilities use observations through each timestamp; strategy labels come from training only. Positions execute one bar later, with fees on executed turnover. The options overlay is a diagnostic signal and contributes no option P&amp;L.</p>
<div class="metrics">{''.join(cards)}</div>{''.join(figures)}
<h2>Evaluation record</h2><pre>{html.escape(json.dumps(provenance, indent=2))}</pre>
<p>{"All prices are generated fixtures. This report demonstrates the software workflow and provides no market performance evidence." if provenance.get("synthetic") else "Research output depends on the supplied data and assumptions; no live execution is performed."}</p>
</main></body></html>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document)
    return output
