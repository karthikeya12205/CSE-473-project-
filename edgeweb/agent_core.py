# Auto-extracted from EdgeWebAgent.ipynb (sections 1-6)
import os, re, sys, json, time, math, zlib, string, pickle, random, asyncio, statistics, tracemalloc
from html import escape
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np, pandas as pd
from lxml import html as lhtml
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

SEED = 7
random.seed(SEED); np.random.seed(SEED)
print("CPU cores available:", os.cpu_count())

VOCAB = {
 # ---- form fields
 "username":    dict(train=["Username","User name","Login ID","Account name","User ID"], unseen=["Handle","Sign-in name","Your login"]),
 "password":    dict(train=["Password","Passcode","Account password"], unseen=["Pass phrase","Secret word"]),
 "email":       dict(train=["Email","E-mail address","Email address","Contact email"], unseen=["Mail","Where can we reach you (email)"]),
 "first_name":  dict(train=["First name","Given name","Forename"], unseen=["Your first name","Name (first)"]),
 "last_name":   dict(train=["Last name","Surname","Family name"], unseen=["Name (last)","Your surname"]),
 "country":     dict(train=["Country","Country/Region","Nation"], unseen=["Where do you live","Place of residence"]),
 "agree_terms": dict(train=["I agree to the Terms","Accept terms and conditions","I accept the terms of service"], unseen=["I have read and accept the conditions","Terms: I consent"]),
 "newsletter":  dict(train=["Subscribe to newsletter","Send me promotional emails","Email me offers"], unseen=["Keep me posted with news","Yes, send updates and deals"]),
 "search_query":dict(train=["Search","Search products","What are you looking for?"], unseen=["Find an item","Type to look up"]),
 # ---- actions
 "submit":      dict(train=["Sign in","Log in","Create account","Register","Submit"], unseen=["Let me in","Finish sign-up","Go"]),
 "next":        dict(train=["Next","Continue","Next step"], unseen=["Proceed","On to step two"]),
 "search_btn":  dict(train=["Search","Find","Go search"], unseen=["Look it up","Show results"]),
 "accept_cookies": dict(train=["Accept all cookies","Accept cookies","Allow all"], unseen=["Got it, sounds good","I'm fine with cookies"]),
 "dismiss_modal":  dict(train=["No thanks","Close","Dismiss"], unseen=["Maybe later","Not now"]),
 "add_to_cart": dict(train=["Add to cart","Add to basket","Buy"], unseen=["Put in my bag","Add this item"]),
 "view_cart":   dict(train=["View cart","Cart","Go to basket"], unseen=["Open my bag","Review order"]),
 "checkout":    dict(train=["Checkout","Place order","Pay now"], unseen=["Complete purchase","Confirm and pay"]),
 # ---- distractors (all collapse into class "distractor")
 "d_link":  dict(train=["Forgot password?","Privacy policy","Help center","Contact us","Cancel","Reset form","Sort by price","Continue shopping","Back to login"],
                 unseen=["Trouble signing in?","Legal notice","Support","Start over","Order by cost","Keep browsing","Return to sign-in"]),
 "d_field": dict(train=["Promo code (optional)","Referral code (optional)","Middle name (optional)"], unseen=["Coupon (optional)","Nickname (optional)"]),
 "d_check": dict(train=["Remember me","Keep me signed in"], unseen=["Stay logged in"]),
 "d_nav":   dict(train=["Home","Products","About us","Blog","Careers"], unseen=["Main page","Catalog","Who we are"]),
 "d_reject":dict(train=["Reject all","Manage preferences","Decline"], unseen=["No thanks to cookies","Only essential"]),
 "d_join":  dict(train=["Subscribe now","Join","Get offers"], unseen=["Sign me up for emails","Yes, count me in"]),
}
CLASS_OF = {k: "distractor" for k in VOCAB if k.startswith("d_")}
FIELD_KEYS = {"username","password","email","first_name","last_name","country","agree_terms","newsletter","search_query"}
BTN = ["tag_button type_submit","tag_input type_submit"]
META = {
 "username":["tag_input type_text"], "password":["tag_input type_password"],
 "email":["tag_input type_email","tag_input type_text"], "first_name":["tag_input type_text"], "last_name":["tag_input type_text"],
 "country":["tag_select type_select"], "agree_terms":["tag_input type_checkbox"], "newsletter":["tag_input type_checkbox"],
 "search_query":["tag_input type_search","tag_input type_text"],
 "submit":BTN, "next":BTN, "search_btn":BTN, "checkout":BTN,
 "accept_cookies":["tag_button type_button dialog"], "dismiss_modal":["tag_button type_button dialog"],
 "add_to_cart":["tag_button type_button"], "view_cart":["tag_a type_link"],
 "d_link":["tag_a type_link","tag_button type_button"], "d_field":["tag_input type_text"], "d_check":["tag_input type_checkbox"],
 "d_nav":["tag_a type_link"], "d_reject":["tag_button type_button dialog"], "d_join":["tag_button type_button dialog"],
}

def augment(rng, s, field=False):
    """Realistic label noise: case, required-star, 'please enter', typos."""
    r = rng.random()
    if r < .2: s = s.lower()
    elif r < .32: s = s.upper()
    if rng.random() < .25: s += " *"
    if field and rng.random() < .15: s = "Please enter your " + s.lower()
    if rng.random() < .08 and len(s) > 4:
        i = rng.randrange(len(s)); s = s[:i] + s[i+1:]
    return s

def pick_label(rng, key, split):
    if split == "mixed": split = "train" if rng.random() < .5 else "unseen"
    return augment(rng, rng.choice(VOCAB[key][split]), key in FIELD_KEYS)

print(f"{len(VOCAB)} label sources, {sum(len(v['train']) for v in VOCAB.values())} train phrases, {sum(len(v['unseen']) for v in VOCAB.values())} unseen phrases")

def is_hidden(n):
    for a in [n, *n.iterancestors()]:
        st = (a.get("style") or "").replace(" ", "").lower()
        if "display:none" in st or "visibility:hidden" in st or a.get("hidden") is not None or a.get("aria-hidden") == "true":
            return True
    return (n.get("type") or "") == "hidden"

def _txt(n): return " ".join(n.text_content().split())

def _context(n, own, maxlen=160):
    cur = n.getparent()
    while cur is not None and cur.tag not in ("body", "html"):
        t = _txt(cur)
        if len(t) > len(own) + 12: return t[:maxlen]
        cur = cur.getparent()
    return ""

def distill(html_str, max_elems=60):
    tree = lhtml.fromstring(html_str)
    for_map = {l.get("for"): _txt(l) for l in tree.iter("label") if l.get("for")}
    els = []
    for n in tree.iter("input", "button", "select", "textarea", "a"):
        if is_hidden(n): continue
        tag = n.tag
        typ = (n.get("type") or ("select" if tag == "select" else "link" if tag == "a" else "submit" if tag == "button" else "text")).lower()
        eid = n.get("id") or ""
        label = n.get("aria-label") or for_map.get(eid)
        if not label:
            for a in n.iterancestors():
                if a.tag == "label":
                    label = (a.text or "").strip() or (n.tail or "").strip() or _txt(a); break
        if not label and n.get("placeholder"): label = n.get("placeholder")
        if not label:
            if tag in ("button", "a"): label = _txt(n)
            elif typ in ("submit", "button"): label = n.get("value") or ""
        if not label: label = n.get("name") or ""
        value, checked, options = "", False, []
        if tag == "select":
            opts = [(_txt(o), o.get("selected") is not None) for o in n.iter("option")]
            options = [t for t, _ in opts]; value = next((t for t, s in opts if s), "")
        elif typ == "checkbox": checked = n.get("checked") is not None
        elif typ not in ("submit", "button", "link"): value = n.get("value") or ""
        in_dialog = any(a.get("role") == "dialog" for a in n.iterancestors())
        clickable = tag in ("button", "a") or typ in ("submit", "button")
        els.append(dict(id=eid, tag=tag, type=typ, label=" ".join(label.split()), value=value, checked=checked, options=options,
                        ctx=_context(n, label) if clickable else "", in_dialog=in_dialog,
                        meta=f"tag_{tag} type_{typ}" + (" dialog" if in_dialog else "")))
        if len(els) >= max_elems: break
    alerts = [_txt(a) for a in tree.xpath('//*[@role="alert" or @role="status"]')]
    return dict(elements=els, alerts=alerts, success=bool(tree.xpath('//*[@data-state="success"]')))

def to_prompt(obs):
    """Ultra-compact text view of a page (for LLM fallback / token accounting)."""
    lines = []
    for i, e in enumerate(obs["elements"]):
        extra = f' value="{e["value"]}"' if e["value"] else (" checked" if e["checked"] else "")
        extra += f' options={e["options"][:6]}' if e["options"] else ""
        lines.append(f'[{i}] {e["tag"]}/{e["type"]} "{e["label"]}"{extra}' + (f' ctx="{e["ctx"][:60]}"' if e["ctx"] else "") + (" (dialog)" if e["in_dialog"] else ""))
    return "\n".join(lines + [f"ALERT: {a}" for a in obs["alerts"]])

COUNTRIES = ["India","United States","Germany","Japan","Brazil","Canada","France","Australia","Kenya","Mexico","Norway","Singapore"]
FIRST = ["Aarav","Maya","Liam","Sofia","Kenji","Zara","Noah","Ines","Omar","Lena"]
LAST  = ["Sharma","Patel","Garcia","Muller","Tanaka","Silva","Okafor","Novak","Rossi","Kaur"]
BRANDS = ["Aero","Nova","Terra","Zenith","Orbit"]
NOUNS  = ["Trail Backpack","Sport Bottle","Travel Mug","Desk Lamp","Wireless Mouse","Noise Headphones"]
VARS   = ["20L","30L","Blue","Black","Pro","Mini","500ml","1L"]
KINDS  = ["login", "register", "shop"]

def rid(rng, used):
    while True:
        s = "e%d" % rng.randrange(10000, 999999)
        if s not in used: used.add(s); return s

class Site:
    def __init__(self, task):
        self.task = task; self.rng = random.Random(task["seed"] * 31 + 7); self.split = task["vocab"]
        self.used = set(); self._ids = {}; self._lab = {}; self._style = {}; self.decoy_ids = set()
    def id(self, key):
        if key not in self._ids: self._ids[key] = rid(self.rng, self.used)
        return self._ids[key]
    def lab(self, role, key=None):
        k = key or role
        if k not in self._lab: self._lab[k] = pick_label(self.rng, role, self.split)
        return self._lab[k]
    # ---------- HTML builders (no <form> tags so real Chromium never navigates)
    def field(self, key, role, typ="text", value="", checked=False):
        eid = self.id(key); lab = escape(self.lab(role, key))
        st = self._style.setdefault(key, self.rng.choice((0, 3) if typ == "checkbox" else (0, 1, 2, 3)))
        base = f'id="{eid}" name="{eid}" type="{typ}"' + (' checked="checked"' if checked else "") + (f' value="{escape(value)}"' if value else "")
        if typ == "checkbox":
            return (f'<div class="row"><input {base}><label for="{eid}">{lab}</label></div>' if st == 0
                    else f'<div class="row"><label><input {base}> {lab}</label></div>')
        if st == 0: return f'<div class="row"><label for="{eid}">{lab}</label><input {base}></div>'
        if st == 1: return f'<div class="row"><input {base} placeholder="{lab}"></div>'
        if st == 2: return f'<div class="row"><input {base} aria-label="{lab}"></div>'
        return f'<div class="row"><label>{lab} <input {base}></label></div>'
    def select(self, key, role, options):
        eid = self.id(key); lab = escape(self.lab(role, key)); st = self._style.setdefault(key, self.rng.choice((0, 2, 3)))
        opts = '<option value="">-- select --</option>' + "".join(f'<option value="{escape(o)}">{escape(o)}</option>' for o in options)
        if st == 0: return f'<div class="row"><label for="{eid}">{lab}</label><select id="{eid}" name="{eid}">{opts}</select></div>'
        if st == 2: return f'<div class="row"><select id="{eid}" name="{eid}" aria-label="{lab}">{opts}</select></div>'
        return f'<div class="row"><label>{lab} <select id="{eid}" name="{eid}">{opts}</select></label></div>'
    def button(self, key, role, primary=False, link=False):
        eid = self.id(key); lab = escape(self.lab(role, key))
        if link: return f'<a id="{eid}" href="#">{lab}</a>'
        st = self._style.setdefault(key, self.rng.choice((0, 1)) if primary else 0)
        if primary and st == 1: return f'<input id="{eid}" type="submit" value="{lab}">'
        return f'<button id="{eid}" type="{"submit" if primary else "button"}">{lab}</button>'
    def decoy(self, key, role="d_link"):
        eid = self.id(key); self.decoy_ids.add(eid)
        st = self._style.setdefault(key, self.rng.choice((0, 1)))
        return self.button(key, role, link=(st == 0))
    def nav(self):
        return '<div class="nav">' + " ".join(self._navlink(i) for i in range(3)) + "</div>"
    def _navlink(self, i):
        eid = self.id(f"nav{i}"); self.decoy_ids.add(eid)
        return f'<a id="{eid}" href="#">{escape(self.lab("d_nav", f"nav{i}"))}</a>'
    def wrap(self, inner):
        depth = self._style.setdefault("depth", self.rng.randrange(1, 4))
        for _ in range(depth): inner = f'<div class="box">{inner}</div>'
        return inner

class LoginSite(Site):
    def initial_state(self): return {"page": "login", "fails": 0}
    def render(self, st):
        s = self.task["slots"]
        if st["page"] == "dash":
            return f'<h1>Welcome, {escape(s["username"])}</h1><div data-state="success">Signed in successfully</div>'
        rows = [self.field("user", "username"), self.field("pw", "password", "password")]
        if self._style.setdefault("flip", self.rng.random() < .2): rows.reverse()
        if self._style.setdefault("remember", self.rng.random() < .5): rows.append(self.field("rem", "d_check", "checkbox"))
        hp = self.id("hp"); honeypot = f'<div style="display:none"><input id="{hp}" name="{hp}" type="text"></div>'
        alert = '<div role="alert">Invalid username or password</div>' if st["fails"] else ""
        body = alert + "<h2>Sign in</h2><div class='form'>" + "".join(rows) + honeypot + self.button("submit", "submit", primary=True) + "</div>" + self.decoy("forgot") 
        return self.nav() + self.wrap(body)
    def on_click(self, st, eid, form):
        s = self.task["slots"]
        if eid == self.id("submit"):
            ok = form.get(self.id("user")) == s["username"] and form.get(self.id("pw")) == s["password"] and not form.get(self.id("hp"))
            return ({"page": "dash", "fails": 0} if ok else {"page": "login", "fails": st["fails"] + 1}), "submit"
        return st, ("decoy" if eid in self.decoy_ids else "noop")
    def is_success(self, st): return st["page"] == "dash"

class RegisterSite(Site):
    def initial_state(self): return {"page": "s1", "err": ""}
    def render(self, st):
        if st["page"] == "done":
            return '<h1>Account created</h1><div data-state="success">Welcome aboard</div>'
        alert = f'<div role="alert">{escape(st["err"])}</div>' if st.get("err") else ""
        if st["page"] == "s1":
            rows = [self.field("fn", "first_name"), self.field("ln", "last_name"), self.field("em", "email")]
            order = self._style.setdefault("order", self.rng.sample(range(3), 3)); rows = [rows[i] for i in order]
            rows.insert(self._style.setdefault("mid_at", self.rng.randrange(4)), self.field("mid", "d_field"))
            body = "<h2>Step 1 of 2</h2><div class='form'>" + "".join(rows) + self.button("next", "next", primary=True) + "</div>" + self.decoy("back")
        else:
            news_pre = self._style.setdefault("news_pre", self.rng.random() < .6)   # trap: newsletter pre-ticked
            rows = [self.select("country", "country", COUNTRIES), self.field("agree", "agree_terms", "checkbox"),
                    self.field("news", "newsletter", "checkbox", checked=news_pre), self.field("pw", "password", "password")]
            body = "<h2>Step 2 of 2</h2><div class='form'>" + "".join(rows) + self.button("create", "submit", primary=True) + "</div>"
        return self.nav() + self.wrap(alert + body)
    def on_click(self, st, eid, form):
        s = self.task["slots"]
        if st["page"] == "s1" and eid == self.id("next"):
            ok = all(form.get(self.id(k)) == s[r] for k, r in (("fn", "first_name"), ("ln", "last_name"), ("em", "email")))
            return ({"page": "s2", "err": ""} if ok else {"page": "s1", "err": "Please check your details"}), "submit"
        if st["page"] == "s2" and eid == self.id("create"):
            ok = (form.get(self.id("country")) == s["country"] and form.get(self.id("agree")) is True
                  and form.get(self.id("news")) is False and form.get(self.id("pw")) == s["password"])
            return ({"page": "done", "err": ""} if ok else {"page": "s2", "err": "Please complete all required fields"}), "submit"
        return st, ("decoy" if eid in self.decoy_ids else "noop")
    def is_success(self, st): return st["page"] == "done"

def _toks(s): return set(re.findall(r"[a-z0-9]+", s.lower()))

class ShopSite(Site):
    def __init__(self, task):
        super().__init__(task)
        self.add_ids = {self.id("add_" + p["name"]): p["name"] for p in task["catalog"]}
    def initial_state(self): return {"page": "home", "cart": [], "results": [], "status": ""}
    def render(self, st):
        t = self.task
        if st["page"] == "done":
            return '<h1>Order confirmed</h1><div data-state="success">Thank you for your purchase</div>'
        top = self.nav() + self.button("vc", "view_cart", link=True)
        if st["page"] == "home":
            body = "<h2>Shop</h2><div class='form'>" + self.field("q", "search_query", "search") + self.button("sbtn", "search_btn", primary=True) + "</div>"
        elif st["page"] == "results":
            lab = escape(self.lab("add_to_cart", "add")); items = ""
            for n in st["results"]:
                p = next(c for c in t["catalog"] if c["name"] == n)
                items += (f'<li class="item"><span class="name">{escape(n)}</span> <span class="price">${p["price"]}</span> '
                          f'<button id="{self.id("add_" + n)}" type="button">{lab}</button></li>')
            status = f'<div role="status">{escape(st["status"])}</div>' if st["status"] else ""
            body = status + ("<ul>" + items + "</ul>" if items else "<p>No results</p>") + self.decoy("sort")
        else:
            items = "".join(f"<li>{escape(n)}</li>" for n in st["cart"])
            body = "<h2>Your cart</h2><ul>" + items + "</ul>" + self.button("checkout", "checkout", primary=True) + self.decoy("cont")
        return top + self.wrap(body)
    def on_click(self, st, eid, form):
        t = self.task
        if st["page"] == "home" and eid == self.id("sbtn"):
            q = _toks(form.get(self.id("q"), ""))
            hits = [p["name"] for p in t["catalog"] if _toks(p["name"]) & q]
            random.Random(t["seed"] + len(q)).shuffle(hits)
            return dict(st, page="results", results=hits, status="" if hits else "none"), "submit"
        if st["page"] == "results" and eid in self.add_ids:
            return dict(st, cart=st["cart"] + [self.add_ids[eid]], status="Item added to cart"), "add"
        if eid == self.id("vc"): return dict(st, page="cart"), "nav"
        if st["page"] == "cart" and eid == self.id("checkout"): return dict(st, page="done"), "submit"
        return st, ("decoy" if eid in self.decoy_ids else "noop")
    def is_success(self, st): return st["page"] == "done" and st["cart"] == [self.task["target"]]

SITES = {"login": LoginSite, "register": RegisterSite, "shop": ShopSite}

def make_task(kind, seed, vocab="train"):
    r = random.Random(seed * 1009 + KINDS.index(kind) * 17 + 3)
    pw = "".join(r.choice(string.ascii_letters + string.digits) for _ in range(9))
    task = dict(kind=kind, seed=seed, vocab=vocab, target=None, catalog=[])
    if kind == "login":
        u = r.choice(FIRST).lower() + str(r.randrange(10, 99))
        task.update(slots={"username": u, "password": pw}, goal=f"Log in as {u}")
    elif kind == "register":
        fn, ln = r.choice(FIRST), r.choice(LAST)
        task.update(slots={"first_name": fn, "last_name": ln, "email": f"{fn.lower()}.{ln.lower()}@example.com",
                           "country": r.choice(COUNTRIES), "agree_terms": True, "newsletter": False, "password": pw},
                    goal=f"Register {fn} {ln}, do NOT subscribe to the newsletter")
    else:
        b, n, v = r.choice(BRANDS), r.choice(NOUNS), r.choice(VARS); target = f"{b} {n} {v}"
        names = {target}
        while len(names) < 4:
            cand = f"{b} {n} {r.choice(VARS)}" if r.random() < .6 else f"{r.choice(BRANDS)} {n} {v}"
            names.add(cand)
        while len(names) < 7:
            names.add(f"{r.choice(BRANDS)} {r.choice(NOUNS)} {r.choice(VARS)}")
        catalog = [dict(name=x, price=r.randrange(9, 199)) for x in sorted(names)]
        task.update(slots={"search_query": target}, target=target, catalog=catalog, goal=f"Buy exactly one '{target}'")
    return task

class SimBrowser:
    """Live lxml DOM + site logic + blocking overlays. Same interface a Playwright wrapper would expose."""
    def __init__(self, task, p_cookie=.5, p_modal=.3):
        self.task = task; self.site = SITES[task["kind"]](task); self.rng = random.Random(task["seed"] * 131 + 11)
        self.state = self.site.initial_state()
        self.n_actions = 0; self.errors = 0; self.misclicks = 0
        self.modal_armed = self.rng.random() < p_modal; self.modal_fired = False
        self.ov = {k: rid(self.rng, self.site.used) for k in ("cookie", "acc", "rej", "modal", "dis", "join")}
        self._ovlab = {}
        self._render()
        if self.rng.random() < p_cookie: self._add_overlay("cookie")
    def _l(self, key):
        if key not in self._ovlab: self._ovlab[key] = escape(pick_label(self.rng, key, self.task["vocab"]))
        return self._ovlab[key]
    def _render(self):
        body = self.site.render(self.state)
        self.tree = lhtml.fromstring(f"<html><head><title>{self.task['kind']}</title></head><body>{body}</body></html>")
    def _add_overlay(self, kind):
        i = self.ov
        if kind == "cookie":
            h = (f'<div id="{i["cookie"]}" data-overlay="1" role="dialog"><p>We use cookies to improve your experience.</p>'
                 f'<button id="{i["acc"]}" type="button">{self._l("accept_cookies")}</button><button id="{i["rej"]}" type="button">{self._l("d_reject")}</button></div>')
        else:
            h = (f'<div id="{i["modal"]}" data-overlay="1" role="dialog"><p>Join our newsletter for 10% off!</p>'
                 f'<button id="{i["join"]}" type="button">{self._l("d_join")}</button><button id="{i["dis"]}" type="button">{self._l("dismiss_modal")}</button></div>')
        self.tree.find("body").append(lhtml.fragment_fromstring(h))
    def html(self): return lhtml.tostring(self.tree, encoding="unicode")
    def is_success(self): return self.site.is_success(self.state)   # ground truth (never shown to the agent)
    def form_values(self):
        d = {}
        for n in self.tree.iter("input", "select", "textarea"):
            k = n.get("name") or n.get("id"); t = (n.get("type") or "text")
            if t == "checkbox": d[k] = n.get("checked") is not None
            elif n.tag == "select":
                sel = [o for o in n.iter("option") if o.get("selected") is not None]
                d[k] = (sel[0].get("value") or _txt(sel[0])) if sel else ""
            else: d[k] = n.get("value") or ""
        return d
    def _after_edit(self):
        if self.modal_armed and not self.modal_fired:
            self.modal_fired = True; self._add_overlay("modal")
    def act(self, a):
        self.n_actions += 1
        nodes = self.tree.xpath("//*[@id=$i]", i=a.get("id", ""))
        if not nodes: self.errors += 1; return False, "not_found"
        n = nodes[0]
        if is_hidden(n): self.errors += 1; return False, "not_visible"
        ov = n.xpath("ancestor-or-self::*[@data-overlay]")
        if self.tree.xpath("//*[@data-overlay]") and not ov: self.errors += 1; return False, "blocked"
        op = a.get("op")
        typ = (n.get("type") or ("link" if n.tag == "a" else "submit" if n.tag == "button" else "select" if n.tag == "select" else "text")).lower()
        if ov:
            if op != "click": self.errors += 1; return False, "bad_op"
            if n.get("id") == self.ov["join"]: self.misclicks += 1
            ov[0].getparent().remove(ov[0]); return True, "overlay"
        if op == "type":
            if n.tag not in ("input", "textarea") or typ in ("checkbox", "submit", "button"): self.errors += 1; return False, "bad_op"
            n.set("value", str(a["value"])); self._after_edit(); return True, "ok"
        if op == "select":
            if n.tag != "select": self.errors += 1; return False, "bad_op"
            hit = False
            for o in n.iter("option"):
                if _txt(o) == a["value"]: o.set("selected", "selected"); hit = True
                elif o.get("selected") is not None: del o.attrib["selected"]
            if not hit: self.errors += 1; return False, "no_option"
            self._after_edit(); return True, "ok"
        if op == "check":
            if typ != "checkbox": self.errors += 1; return False, "bad_op"
            if a["value"]: n.set("checked", "checked")
            elif n.get("checked") is not None: del n.attrib["checked"]
            self._after_edit(); return True, "ok"
        if op == "click":
            if typ == "checkbox":
                if n.get("checked") is not None: del n.attrib["checked"]
                else: n.set("checked", "checked")
                return True, "ok"
            if n.tag in ("select", "textarea") or typ in ("text", "password", "email", "search"): return True, "focus"
            self.state, msg = self.site.on_click(self.state, n.get("id"), self.form_values())
            if msg == "decoy": self.misclicks += 1
            self._render(); return True, msg
        self.errors += 1; return False, "bad_op"



ROLE_GROUP = {**{r: "text" for r in ("username","password","email","first_name","last_name","search_query")},
              "country": "select", "agree_terms": "checkbox", "newsletter": "checkbox",
              **{r: "click" for r in ("submit","next","search_btn","accept_cookies","dismiss_modal","add_to_cart","view_cart","checkout")},
              "distractor": "any"}

def group_of(e):
    if e["tag"] == "select": return "select"
    if e["type"] == "checkbox": return "checkbox"
    if e["tag"] in ("button", "a") or e["type"] in ("submit", "button", "link"): return "click"
    return "text"

class Grounder:
    def fit(self, split="train", n_aug=14, seed=0):
        rng = random.Random(seed); X, y = [], []
        for key, d in VOCAB.items():
            for phrase in d[split]:
                for _ in range(n_aug):
                    X.append(f"{augment(rng, phrase, key in FIELD_KEYS)} || {rng.choice(META[key])}"); y.append(CLASS_OF.get(key, key))
        self.pipe = Pipeline([
            ("f", FeatureUnion([("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)),
                                ("w", TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"[A-Za-z_]+"))])),
            ("clf", LogisticRegression(C=20, max_iter=3000))]).fit(X, y)
        self.classes = list(self.pipe.classes_); self.col = {c: i for i, c in enumerate(self.classes)}
        self.mask = {g: np.array([1.0 if ROLE_GROUP[c] in (g, "any") else 0.0 for c in self.classes]) for g in ("text", "select", "checkbox", "click")}
        return self
    def classify(self, els):
        if not els: return
        P = self.pipe.predict_proba([f"{e['label']} || {e['meta']}" for e in els])
        for e, row in zip(els, P):
            row = row * self.mask[group_of(e)]; s = row.sum(); row = row / s if s > 0 else row
            j = int(row.argmax()); e["role"], e["conf"], e["p"] = self.classes[j], float(row[j]), row

GROUNDER = Grounder().fit("train")

def grounder_accuracy(split, n_pages=300):
    ok = tot = 0
    for s in range(n_pages):
        t = make_task(KINDS[s % 3], 90000 + s, split); b = SimBrowser(t, p_cookie=1, p_modal=0)
        els = distill(b.html())["elements"]; GROUNDER.classify(els)
        site_roles = {v: k for k, v in b.site._ids.items()}
        for e in els:
            if e["role"] is None: continue
            # ground-truth role for elements we know (primary actions / fields); skip the rest
            key = site_roles.get(e["id"], "")
            truth = {"user":"username","pw":"password","fn":"first_name","ln":"last_name","em":"email","country":"country","agree":"agree_terms","news":"newsletter",
                     "q":"search_query","submit":"submit","create":"submit","next":"next","sbtn":"search_btn","checkout":"checkout","vc":"view_cart"}.get(key)
            if truth is None and (key.startswith("nav") or key in ("forgot","back","sort","cont","rem","mid")): truth = "distractor"
            if truth is None and e["id"] == b.ov["acc"]: truth = "accept_cookies"
            if truth is None and e["id"] == b.ov["rej"]: truth = "distractor"
            if truth: tot += 1; ok += (e["role"] == truth)
    return ok / tot


def _norm(s): return " ".join(str(s).lower().split())
def match_score(target, ctx):
    T, C = _toks(target), _toks(ctx); return len(T & C) / max(1, len(T | C))
def best_option(options, want):
    cands = [o for o in options if o and not o.startswith("--")]
    return next((o for o in cands if _norm(o) == _norm(want)), None) or (max(cands, key=lambda o: match_score(want, o), default=None))

class Agent:
    def __init__(self, grounder, max_steps=30, thr=0.30, use_fallback=True):
        self.g, self.max_steps, self.thr, self.fallback = grounder, max_steps, thr, use_fallback
    def begin(self, task):
        self.task = task; self.mem = dict(added=False, hist=[], tabu=set(), sig=())
    def plan(self, html):
        obs = distill(html); els = obs["elements"]; m = self.mem; t = self.task; slots = t["slots"]
        if obs["success"]: return {"op": "done"}
        self.g.classify(els)
        sig = m["sig"] = tuple(sorted(e["id"] for e in els))
        ok = lambda e: (sig, e["id"]) not in m["tabu"]
        act = lambda op, e, v=None: {"op": op, "id": e["id"], "value": v, "role": e.get("role")}
        good = lambda e: e.get("conf", 0) >= self.thr and ok(e)
        # 0) blocking dialogs (cookie banner / newsletter pop-up)
        dlg = [e for e in els if e["in_dialog"] and ok(e) and group_of(e) == "click"]
        if dlg:
            hit = [e for e in dlg if e["role"] in ("accept_cookies", "dismiss_modal") and good(e)]
            if hit: return act("click", max(hit, key=lambda e: e["conf"]))
            if self.fallback:
                p = lambda e: e["p"][self.g.col["accept_cookies"]] + e["p"][self.g.col["dismiss_modal"]]
                return act("click", max(dlg, key=p))
        # 1) fill / fix fields
        for e in els:
            r = e.get("role")
            if r in slots and good(e) and not e["in_dialog"]:
                want = slots[r]
                if r == "country":
                    if _norm(e["value"]) != _norm(want):
                        o = best_option(e["options"], want)
                        if o: return act("select", e, o)
                elif r in ("agree_terms", "newsletter"):
                    if bool(e["checked"]) != bool(want): return act("check", e, bool(want))
                elif e["value"] != str(want): return act("type", e, str(want))
        # 2) shopping: choose the right product row
        pick = lambda role: [e for e in els if e.get("role") == role and good(e) and not e["in_dialog"]]
        if t["kind"] == "shop" and not m["added"]:
            adds = pick("add_to_cart")
            if not adds and self.fallback:   # semantic fallback: any button whose row mentions the target
                adds = [e for e in els if e["type"] == "button" and e["tag"] == "button" and ok(e) and not e["in_dialog"] and match_score(t["target"], e["ctx"]) >= .4]
            if adds: return act("click", max(adds, key=lambda e: match_score(t["target"], e["ctx"])))
        # 3) progression button
        order = (["next", "submit"] if t["kind"] == "register" else ["submit"] if t["kind"] == "login"
                 else (["checkout", "view_cart"] if m["added"] else ["search_btn", "submit"]))
        for role in order:
            c = pick(role)
            if c: return act("click", max(c, key=lambda e: e["conf"]))
        if self.fallback and t["kind"] == "shop" and m["added"]:   # relative ranking: most cart/checkout-like control on the page
            cc = [e for e in els if group_of(e) == "click" and ok(e) and not e["in_dialog"]]
            sc = lambda e: e["p"][self.g.col["view_cart"]] + e["p"][self.g.col["checkout"]]
            if cc: return act("click", max(cc, key=sc))
        if self.fallback:   # DOM prior: the form's primary submit control
            c = [e for e in els if e["type"] == "submit" and ok(e) and not e["in_dialog"]]
            if c: return act("click", c[0])
        return None
    def feedback(self, a, ok, msg):
        m = self.mem; key = (a["op"], a["id"], a.get("value")); m["hist"].append(key)
        if a["op"] == "click" and ok and msg == "add": m["added"] = True
        if not ok and msg in ("not_found", "not_visible", "no_option", "bad_op"): m["tabu"].add((m["sig"], a["id"]))
        if len(m["hist"]) >= 2 and m["hist"][-1] == m["hist"][-2]: m["tabu"].add((m["sig"], a["id"]))   # no progress → ban
    def run(self, br, task):
        self.begin(task); claimed, reason, step_ms = False, "max_steps", []
        for _ in range(self.max_steps):
            t0 = time.perf_counter(); a = self.plan(br.html()); step_ms.append((time.perf_counter() - t0) * 1000)
            if a is None: reason = "stuck"; break
            if a["op"] == "done": claimed, reason = True, "done"; break
            ok, msg = br.act(a); self.feedback(a, ok, msg)
        return dict(kind=task["kind"], vocab=task["vocab"], success=br.is_success(), claimed=claimed, steps=br.n_actions,
                    errors=br.errors, misclicks=br.misclicks, plan_ms=sum(step_ms), step_ms=step_ms, reason=reason)

