"""
Local live dashboard + static report.

  Live:    carwatch.py --dashboard      -> http://127.0.0.1:8765  (localhost only)
  Report:  python dashboard.py speed_history.csv --limit 50 --out report.html

Plain HTML + canvas, no external libraries or network calls, so footage-derived
data never leaves the machine.
"""
from __future__ import annotations

import argparse
import json
import threading
from dataclasses import asdict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from speedlog import SpeedRecord, demo_records, fit_model, load_records

MAX_POINTS = 1500


def snapshot(records: list[SpeedRecord], limit: float, street: str, live=None, model=None,
             flags=None, meta=None) -> dict:
    over = sum(r.over for r in records)
    bases = {r.basis for r in records}
    return {"limit": limit, "street": street, "updated": datetime.now().strftime("%H:%M:%S"),
            "totals": {"over": over, "under": len(records) - over, "total": len(records)},
            "estimated_scale": "approx" in bases,
            "records": [asdict(r) for r in records[-MAX_POINTS:]],
            "live": live or [], "model": model if model is not None else fit_model(records),
            "flags": flags or {"counts": {}, "recent": []}, "meta": meta or {}}


class Dashboard:
    """Holds the latest snapshot and serves it on 127.0.0.1."""

    def __init__(self, port: int = 8765, evidence_dir: str | Path | None = None):
        self._lock = threading.Lock()
        self._data = json.dumps(snapshot([], 50, ""))
        outer = self
        ev_root = Path(evidence_dir).resolve() if evidence_dir else None

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.startswith("/data.json"):
                    with outer._lock:
                        body, ctype = outer._data.encode(), "application/json"
                elif self.path.startswith("/evidence/") and ev_root:
                    # serve only files inside the evidence folder (no path traversal)
                    from urllib.parse import unquote
                    target = (ev_root / unquote(self.path[len("/evidence/"):].split("?")[0])).resolve()
                    if ev_root not in target.parents or not target.is_file():
                        self.send_error(404); return
                    body = target.read_bytes()
                    ctype = {".html": "text/html; charset=utf-8", ".png": "image/png", ".mp4": "video/mp4",
                             ".json": "application/json"}.get(target.suffix, "application/octet-stream")
                elif self.path in ("/", "/index.html"):
                    body, ctype = PAGE.replace("/*DATA*/null", "null").encode(), "text/html; charset=utf-8"
                else:
                    self.send_error(404); return
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", port), H)   # localhost only, on purpose
        self.url = f"http://127.0.0.1:{port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def update(self, records, limit, street, live=None, model=None, flags=None, meta=None):
        s = json.dumps(snapshot(records, limit, street, live, model, flags, meta))
        with self._lock:
            self._data = s

    def close(self):
        self.server.shutdown()


def make_chart(records: list[SpeedRecord], limit: float, out_path: str | Path, street: str = "",
               flags: dict | None = None, meta: dict | None = None) -> Path:
    """Write a standalone HTML report (same charts as the live page) from recorded data."""
    data = json.dumps(snapshot(records, limit, street, flags=flags, meta=meta))
    html = PAGE.replace("/*DATA*/null", data)
    Path(out_path).write_text(html, encoding="utf-8")
    return Path(out_path)


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>CarWatch speed dashboard</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#16181d;--dim:#6b7280;--line:#e3e6eb;--over:#d64545;--under:#2f8f5b;--acc:#3b6fd4}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--card:#171a21;--ink:#e8eaee;--dim:#9aa1ad;--line:#2a2f3a;--over:#ef6b6b;--under:#4fbf83;--acc:#7aa2f7}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:18px;margin:0}.sub{color:var(--dim);margin:2px 0 14px}
.grid{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.card h2{font-size:13px;margin:0 0 8px;color:var(--dim);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.tiles{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:12px}
.tile b{font-size:30px;display:block}.tile span{color:var(--dim);font-size:12px}
canvas{width:100%;height:210px;display:block}
.banner{background:#fff4d6;color:#5a4300;border:1px solid #f0d98a;border-radius:8px;padding:8px 12px;margin-bottom:12px;display:none}
@media (prefers-color-scheme:dark){.banner{background:#3a3114;color:#f3dc8a;border-color:#5a4c1c}}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:4px 6px;text-align:right;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}th{color:var(--dim);font-weight:600;font-size:12px}
.chip{display:inline-block;margin:0 6px 0 0;padding:1px 8px;border-radius:99px;background:var(--line);font-size:12px;color:var(--ink)}
.fb{display:flex;align-items:center;gap:8px;margin:3px 0;font-size:12px}.fb i{display:block;height:10px;border-radius:3px;background:var(--over)}
.fl{margin-top:8px;max-height:150px;overflow:auto;font-size:12px}.fl div{padding:2px 0;border-bottom:1px solid var(--line)}.fl a{color:var(--acc)}
.sig{font-weight:700}.note{color:var(--dim);font-size:12px;margin-top:8px}
.live span{display:inline-block;margin:2px 6px 2px 0;padding:2px 8px;border-radius:99px;border:1px solid var(--line);font-size:12px}
.live .o{border-color:var(--over);color:var(--over)}
</style></head><body><main>
<h1>CarWatch speed dashboard</h1>
<div class="sub" id="sub">waiting for data…</div>
<div class="banner" id="banner">Speeds use an estimated scale (no calibration). Treat over/under counts as rough.</div>
<div class="tiles">
<div class="card tile"><b id="tOver" style="color:var(--over)">0</b><span>over limit</span></div>
<div class="card tile"><b id="tUnder" style="color:var(--under)">0</b><span>at or under</span></div>
<div class="card tile"><b id="tPct">–</b><span>share over</span></div></div>
<div class="grid">
<div class="card"><h2>Speed distribution</h2><canvas id="cHist"></canvas></div>
<div class="card"><h2>Recent vehicles</h2><canvas id="cSeries"></canvas></div>
<div class="card"><h2>Over vs under by hour</h2><canvas id="cHour"></canvas></div>
<div class="card"><h2>On screen now</h2><div class="live" id="live">–</div></div>
<div class="card"><h2>Flags (driving patterns)</h2><div id="flagBars"></div><div id="flagList"></div></div>
</div>
<div class="card" style="margin-top:12px"><h2>Linear regression: speed ~ colour + street + hour</h2>
<div id="model">–</div></div>
<p class="note">Flags show driving speed patterns only. They do not identify impaired drivers or prove a violation.</p>
</main><script>
const EMBED=/*DATA*/null;
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function setup(c){const r=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*r;c.height=h*r;const x=c.getContext('2d');x.scale(r,r);x.clearRect(0,0,w,h);return[x,w,h]}
function axes(x,w,h,ymax,yl){x.strokeStyle=css('--line');x.fillStyle=css('--dim');x.font='11px system-ui';x.lineWidth=1;
 for(let i=0;i<=4;i++){const y=h-24-(h-34)*i/4;x.beginPath();x.moveTo(34,y);x.lineTo(w-6,y);x.stroke();x.fillText(Math.round(ymax*i/4),2,y+3)}
 if(yl)x.fillText(yl,36,10)}
function hist(d){const[x,w,h]=setup(cHist),R=d.records;if(!R.length)return;const lim=d.limit,mx=Math.max(lim*1.8,...R.map(r=>r.speed_kmh)),bw=5,nb=Math.ceil(mx/bw),c=Array(nb).fill(0);
 R.forEach(r=>c[Math.min(nb-1,Math.floor(r.speed_kmh/bw))]++);const ym=Math.max(...c,4);axes(x,w,h,ym,'vehicles');
 const pw=(w-40)/nb;c.forEach((v,i)=>{const over=(i*bw)>=lim*1.1;x.fillStyle=over?css('--over'):css('--under');const bh=(h-34)*v/ym;x.fillRect(36+i*pw+1,h-24-bh,pw-2,bh)});
 const lx=36+(lim/bw)*pw;x.strokeStyle=css('--ink');x.setLineDash([4,3]);x.beginPath();x.moveTo(lx,8);x.lineTo(lx,h-24);x.stroke();x.setLineDash([]);x.fillStyle=css('--ink');x.fillText('limit '+lim,lx+4,18);
 x.fillStyle=css('--dim');x.fillText('km/h',w-34,h-8);[0,0.25,0.5,0.75].forEach(f=>x.fillText(Math.round(mx*f),36+(w-40)*f-8,h-8))}
function series(d){const[x,w,h]=setup(cSeries),R=d.records.slice(-120);if(!R.length)return;const mx=Math.max(d.limit*1.6,...R.map(r=>r.speed_kmh));axes(x,w,h,mx,'km/h');
 const ly=h-24-(h-34)*d.limit/mx;x.strokeStyle=css('--ink');x.setLineDash([4,3]);x.beginPath();x.moveTo(34,ly);x.lineTo(w-6,ly);x.stroke();x.setLineDash([]);
 R.forEach((r,i)=>{const px=38+(w-48)*(R.length>1?i/(R.length-1):.5),py=h-24-(h-34)*r.speed_kmh/mx;x.fillStyle=r.over?css('--over'):css('--under');x.beginPath();x.arc(px,py,3.5,0,7);x.fill()})}
function hourly(d){const[x,w,h]=setup(cHour),R=d.records;if(!R.length)return;const o=Array(24).fill(0),u=Array(24).fill(0);
 R.forEach(r=>{const k=Math.floor(r.hour)%24;r.over?o[k]++:u[k]++});let lo=24,hi=0;for(let i=0;i<24;i++)if(o[i]+u[i]){lo=Math.min(lo,i);hi=Math.max(hi,i)}
 lo=Math.max(0,Math.min(lo,hi-5));hi=Math.min(23,Math.max(hi,lo+5));const n=hi-lo+1,ym=Math.max(...o.map((v,i)=>v+u[i]),4);axes(x,w,h,ym,'vehicles');const pw=(w-40)/n;
 for(let i=lo;i<=hi;i++){const px=36+(i-lo)*pw+2,hu=(h-34)*u[i]/ym,ho=(h-34)*o[i]/ym;x.fillStyle=css('--under');x.fillRect(px,h-24-hu,pw-4,hu);x.fillStyle=css('--over');x.fillRect(px,h-24-hu-ho,pw-4,ho);
  if(n<=12||(i-lo)%2===0){x.fillStyle=css('--dim');x.fillText(i+'h',px+(pw-4)/2-7,h-8)}}}
function flagsCard(d){const F=d.flags||{counts:{},recent:[]},c=F.counts,el=document.getElementById('flagBars'),mx=Math.max(1,...Object.values(c));
 el.innerHTML=Object.keys(c).length?Object.entries(c).sort((a,b)=>b[1]-a[1]).map(([k,v])=>'<div class="fb"><span style="width:130px">'+k+'</span><i style="width:'+(150*v/mx)+'px"></i><b>'+v+'</b></div>').join(''):'<span class="note">No flags yet.</span>';
 document.getElementById('flagList').innerHTML='<div class="fl">'+F.recent.map(f=>'<div>'+f.time.slice(11)+' · '+f.vehicle+' · '+f.flag+' · <span style="color:var(--dim)">'+f.detail+'</span>'+(f.evidence?' · <a href="/evidence/'+f.evidence+'/summary.html" target="_blank">evidence</a>':'')+'</div>').join('')+'</div>'}
function model(d){const m=d.model,el=document.getElementById('model');if(!m.ready){el.innerHTML='<span class="note">Need more data: '+m.n+' / '+m.need+' vehicles recorded.</span>';return}
 let h='<div class="note" style="margin:0 0 8px">n = '+m.n+' &nbsp; R² = '+m.r2.toFixed(3)+' (adj '+m.adj_r2.toFixed(3)+') &nbsp; typical error ±'+m.rmse.toFixed(1)+' km/h &nbsp; baseline: '+Object.entries(m.baseline).map(e=>e[0]+' = '+e[1]).join(', ')+'</div>';
 h+='<table><tr><th>term</th><th>effect (km/h)</th><th>± se</th><th>t</th></tr>';m.terms.forEach(t=>{h+='<tr class="'+(t.sig?'sig':'')+'"><td>'+t.name+'</td><td>'+t.beta.toFixed(1)+'</td><td>'+t.se.toFixed(1)+'</td><td>'+t.t.toFixed(1)+'</td></tr>'});
 el.innerHTML=h+'</table><div class="note">Bold = |t| &gt; 2. '+m.note+'</div>'}
function render(d){if(!d)return;document.getElementById('sub').innerHTML=(d.street||'street not set')+' · limit '+d.limit+' km/h · updated '+d.updated+'<br>'+Object.entries(d.meta||{}).map(([k,v])=>'<span class="chip">'+k+': '+v+'</span>').join('');
 banner.style.display=d.estimated_scale?'block':'none';tOver.textContent=d.totals.over;tUnder.textContent=d.totals.under;
 tPct.textContent=d.totals.total?Math.round(100*d.totals.over/d.totals.total)+'%':'–';
 document.getElementById('live').innerHTML=d.live.length?d.live.map(v=>'<span class="'+(v.over?'o':'')+'">'+v.id+' '+v.color+' '+Math.round(v.speed)+' km/h</span>').join(''):'–';
 hist(d);series(d);hourly(d);flagsCard(d);model(d)}
async function poll(){try{const r=await fetch('/data.json',{cache:'no-store'});render(await r.json())}catch(e){}}
if(EMBED){render(EMBED);addEventListener('resize',()=>render(EMBED))}else{poll();setInterval(poll,1000);addEventListener('resize',poll)}
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description="Write a standalone HTML report from a speed history CSV")
    ap.add_argument("history", nargs="?", help="speed_history.csv (omit with --demo)")
    ap.add_argument("--demo", action="store_true", help="use clearly synthetic demo data")
    ap.add_argument("--limit", type=float, default=50)
    ap.add_argument("--street", default="")
    ap.add_argument("--out", default="carwatch_report.html")
    a = ap.parse_args()
    if not a.history and not a.demo:
        ap.error("give a history CSV or --demo")
    recs = demo_records() if a.demo else load_records(a.history)
    street = a.street or ("SYNTHETIC DEMO DATA" if a.demo else "")
    flags = meta = None
    if a.demo:
        flags = {"counts": {"OVER LIMIT": sum(r.over for r in recs), "HARSH ACCEL/BRAKE": 11, "WEAVING": 4},
                 "recent": [{"time": f"2026-01-01 {h}", "vehicle": v, "type": "car", "flag": f,
                             "detail": d, "evidence": None} for h, v, f, d in (
                     ("17:42:08", "V0388", "HARSH ACCEL/BRAKE", "p_HARSH=0.91 acc_min=-2.1 dv=2.4"),
                     ("17:41:55", "V0381", "OVER LIMIT", "63 +/-2 km/h vs limit 50 (calibrated scale)"),
                     ("17:40:12", "V0377", "WEAVING", "p_WEAVING=0.88 lat_amp=0.41 lat_cross=5"),
                     ("17:38:30", "V0369", "OVER LIMIT", "71 +/-2 km/h vs limit 50 (calibrated scale)"))]}
        meta = {"behaviour": "learned model", "tracker": "botsort", "fps": "30", "scale": "calibrated (demo)"}
    print(f"wrote {make_chart(recs, a.limit, a.out, street, flags, meta)}  ({len(recs)} vehicles)")


if __name__ == "__main__":
    main()
