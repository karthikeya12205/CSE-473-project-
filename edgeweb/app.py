"""Web UI for EdgeWebAgent.  Run:  python app.py  ->  http://127.0.0.1:5000"""
import time
from flask import Flask, jsonify, request, render_template
from agent_core import *   # VOCAB, distill, SimBrowser, Agent, Grounder, make_task, KINDS ...

app = Flask(__name__)
G = Grounder().fit("train")

def episode(kind, seed, vocab, fallback=True, p_cookie=.5, p_modal=.3):
    task = make_task(kind, seed, vocab)
    br = SimBrowser(task, p_cookie, p_modal); ag = Agent(G, use_fallback=fallback); ag.begin(task)
    steps, reason = [], "max_steps"
    for _ in range(ag.max_steps):
        html = br.html(); els = distill(html)["elements"]; G.classify(els)
        t0 = time.perf_counter(); a = ag.plan(html); ms = (time.perf_counter() - t0) * 1000
        step = dict(html=html, ms=round(ms, 2),
                    elements=[dict(id=e["id"], label=e["label"], role=e["role"], conf=round(e["conf"], 2)) for e in els])
        if a is None: reason = "stuck"; step["action"] = None; steps.append(step); break
        if a["op"] == "done": reason = "done"; step["action"] = dict(op="done"); steps.append(step); break
        ok, msg = br.act(a); ag.feedback(a, ok, msg)
        step["action"] = dict(op=a["op"], id=a["id"], value=a["value"], role=a["role"], ok=ok, msg=msg)
        steps.append(step)
    return dict(goal=task["goal"], slots=task["slots"], steps=steps, success=br.is_success(), reason=reason,
                misclicks=br.misclicks, errors=br.errors, final_html=br.html())

@app.get("/")
def index(): return render_template("index.html", kinds=KINDS)

@app.post("/api/run")
def api_run():
    d = request.json
    return jsonify(episode(d["kind"], int(d["seed"]), d["vocab"], d.get("fallback", True), float(d.get("p_cookie", .5)), float(d.get("p_modal", .3))))

@app.post("/api/benchmark")
def api_bench():
    d = request.json; n = min(int(d.get("n", 100)), 500); rows = []
    for vocab in ("train", "unseen"):
        for fb in (False, True):
            res = [Agent(G, use_fallback=fb).run(SimBrowser(t), t) for t in (make_task(KINDS[i % 3], 1000 + i, vocab) for i in range(n))]
            rows.append(dict(vocab=vocab, fallback=fb, success=sum(r["success"] for r in res) / n,
                             steps=sum(r["steps"] for r in res) / n, p50_ms=sorted(m for r in res for m in r["step_ms"])[len(res) * 3]))
    return jsonify(rows)

@app.post("/api/inspect")
def api_inspect():
    els = distill(request.json["html"])["elements"]; G.classify(els)
    return jsonify([dict(tag=e["tag"], type=e["type"], label=e["label"], role=e["role"], conf=round(e["conf"], 2)) for e in els])

if __name__ == "__main__": app.run(debug=False, port=5000)
