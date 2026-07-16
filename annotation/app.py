"""Minimal, zero-dependency stance-annotation UI for the DeBERTa validation set.

One screen: the issue proposition, the response text, a 0-100 slider anchored
against<->for the proposition, and a Next arrow. Ratings append to
annotation/ratings_<annotator>.csv the moment each item is submitted, so the CSV
grows live and an interrupted session resumes where it left off.

Pure standard library (http.server) so it runs in any Python without pip installs.

Run (on Brains):  python annotation/app.py --items annotation/items.csv --port 8000
Then open        http://<host>:<port>/
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

_LOCK = threading.Lock()
ITEMS: list[dict] = []
OUT_DIR = Path("annotation")
RATING_FIELDS = ["item_id", "annotator", "score", "unratable", "ts_iso", "dwell_ms"]


def ratings_path(annotator: str) -> Path:
    safe = "".join(c for c in annotator if c.isalnum() or c in "-_").lower() or "anon"
    return OUT_DIR / f"ratings_{safe}.csv"


def done_ids(annotator: str) -> set[str]:
    path = ratings_path(annotator)
    if not path.exists():
        return set()
    with path.open(newline="") as fh:
        return {r["item_id"] for r in csv.DictReader(fh)}


def append_rating(row: dict) -> None:
    path = ratings_path(row["annotator"])
    with _LOCK:
        new = not path.exists()
        with path.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=RATING_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
            fh.flush()


PAGE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Stance annotation</title><style>
 :root{color-scheme:light dark}
 body{font:16px/1.55 -apple-system,system-ui,sans-serif;max-width:760px;margin:0 auto;padding:24px}
 #gate{text-align:center;margin-top:15vh}
 input,button{font:inherit;padding:8px 12px;border-radius:8px;border:1px solid #8888}
 button{cursor:pointer}
 .bar{height:4px;background:#8883;border-radius:2px;margin:4px 0 20px}
 .bar>i{display:block;height:100%;background:#4a90d9;border-radius:2px;transition:width .2s}
 .prop{font-weight:600;margin:6px 0 14px}
 .prop small{font-weight:400;opacity:.6}
 .resp{white-space:pre-wrap;border:1px solid #8884;border-radius:10px;padding:14px 16px;
   max-height:46vh;overflow:auto;background:#88881a}
 .slwrap{margin:22px 0 6px}
 input[type=range]{width:100%}
 .ends{display:flex;justify-content:space-between;font-size:13px;opacity:.7}
 .val{text-align:center;font-size:28px;font-weight:700;margin:10px 0}
 .row{display:flex;align-items:center;gap:14px;margin-top:14px}
 .row label{font-size:14px;opacity:.8}
 .next{margin-left:auto;font-size:18px;padding:10px 22px;background:#4a90d9;color:#fff;border:none}
 #done{text-align:center;margin-top:15vh}
 .hint{font-size:13px;opacity:.6;margin-top:18px}
</style></head><body>

<div id=gate>
 <h2>Stance annotation</h2>
 <p>Enter your name / initials to begin.</p>
 <input id=name placeholder="e.g. CA" autofocus>
 <button onclick=start()>Start &rarr;</button>
</div>

<div id=app style=display:none>
 <div class=bar><i id=fill></i></div>
 <div id=count style="font-size:13px;opacity:.6"></div>
 <p class=prop>Does the text argue for or against:<br>
   <span id=prop></span></p>
 <div class=resp id=resp></div>
 <div class=slwrap>
   <div class=val id=val>50</div>
   <input type=range id=slider min=0 max=100 value=50>
   <div class=ends><span>0 &mdash; fully against</span><span>50 &mdash; neutral</span><span>100 &mdash; fully for</span></div>
 </div>
 <div class=row>
   <label><input type=checkbox id=unratable> refusal / off-topic (unratable)</label>
   <button class=next id=next onclick=submit()>Next &rarr;</button>
 </div>
 <div class=hint>&larr;/&rarr; move slider &middot; Enter = submit &amp; next. Judge only the stance of the text on the proposition above.</div>
</div>

<div id=done style=display:none><h2>All done &mdash; thank you!</h2><p id=donemsg></p></div>

<script>
let A="", cur=null, shownAt=0, busy=false;
function start(){A=document.getElementById('name').value.trim(); if(!A)return;
  document.getElementById('gate').style.display='none';
  document.getElementById('app').style.display='block'; load();}
async function load(){
  const r=await fetch('/api/next?a='+encodeURIComponent(A)); const d=await r.json();
  if(d.done){finish(d);return;}
  cur=d.item;
  document.getElementById('prop').textContent=cur.stance_target;
  document.getElementById('resp').textContent=cur.response_text;
  document.getElementById('resp').scrollTop=0;
  document.getElementById('count').textContent=d.n_done+' / '+d.total+' rated';
  document.getElementById('fill').style.width=(100*d.n_done/d.total)+'%';
  const s=document.getElementById('slider'); s.value=50; setval(50);
  document.getElementById('unratable').checked=false;
  shownAt=Date.now();
}
function setval(v){document.getElementById('val').textContent=v;}
async function submit(){
  if(busy||!cur)return; busy=true;
  const s=document.getElementById('slider');
  await fetch('/api/rate',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({a:A,item_id:cur.item_id,score:+s.value,
      unratable:document.getElementById('unratable').checked?1:0,
      dwell_ms:Date.now()-shownAt})});
  busy=false; load();
}
function finish(d){document.getElementById('app').style.display='none';
  document.getElementById('done').style.display='block';
  document.getElementById('donemsg').textContent=d.n_done+' items rated. You can close this tab.';}
document.getElementById('slider').addEventListener('input',e=>setval(e.target.value));
document.addEventListener('keydown',e=>{
  if(document.getElementById('app').style.display=='none')return;
  if(e.key=='Enter'){submit();}
});
document.getElementById('name').addEventListener('keydown',e=>{if(e.key=='Enter')start();});
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self) -> None:
        u = urlparse(self.path)
        if u.path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif u.path == "/api/next":
            a = (parse_qs(u.query).get("a", [""])[0]).strip()
            if not a:
                return self._json(400, {"error": "no annotator"})
            done = done_ids(a)
            for it in ITEMS:
                if it["item_id"] not in done:
                    return self._json(200, {"done": False, "total": len(ITEMS),
                                            "n_done": len(done), "item": it})
            self._json(200, {"done": True, "total": len(ITEMS), "n_done": len(done)})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/rate":
            return self._json(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", 0))
        d = json.loads(self.rfile.read(n) or b"{}")
        a = (d.get("a") or "").strip()
        if not a or "item_id" not in d:
            return self._json(400, {"error": "bad payload"})
        append_rating({
            "item_id": d["item_id"], "annotator": a, "score": d.get("score", ""),
            "unratable": d.get("unratable", 0),
            "ts_iso": dt.datetime.now().isoformat(timespec="seconds"),
            "dwell_ms": d.get("dwell_ms", ""),
        })
        self._json(200, {"ok": True})

    def log_message(self, *a) -> None:  # quiet
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="annotation/items.csv", type=Path)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    global OUT_DIR
    OUT_DIR = args.items.parent
    with args.items.open(newline="") as fh:
        ITEMS.extend(csv.DictReader(fh))
    print(f"Loaded {len(ITEMS)} items from {args.items}")
    print(f"Ratings -> {OUT_DIR}/ratings_<annotator>.csv")
    print(f"Serving on http://{args.host}:{args.port}/  (Ctrl-C to stop)")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
