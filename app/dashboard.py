"""BidPilot dashboard.  streamlit run app/dashboard.py

Tabs: Store | Run a day | 14-day race | Consultant | Ask the data | Speed
Results are cached in out/ so the demo does not recompute unless asked.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from agent.analyst import PRESET_QUESTIONS, REAL_ADS_QUESTIONS, ask, latest_for, load_source  # noqa: E402
from agent.citations import TAG_RE  # noqa: E402
from agent.consultant import decide_recommendations, latest_report, run_consultant  # noqa: E402
from agent.llm import make_llm, run_sync  # noqa: E402
from agent.orchestrator import finish_day, load_pending_actions, prepare_day_async, run_agent_day  # noqa: E402
from agent.research import load_cached  # noqa: E402
from analysis import real_ads as RA  # noqa: E402
from analysis.metrics import account_totals, stock_metrics  # noqa: E402
from data.generate import generate  # noqa: E402
from settings import load_config, out_dir  # noqa: E402
from sim.platform import AdPlatform, reset  # noqa: E402

st.set_page_config(page_title="BidPilot", page_icon="📈", layout="wide")
CFG = load_config()
OUT = out_dir()
if not (OUT / "products.csv").exists():
    generate(CFG, OUT)
OBJ_PATH = OUT / "objectives.json"          # seller-chosen objectives (Store tab)
if OBJ_PATH.exists():
    CFG["objectives"] = json.loads(OBJ_PATH.read_text(encoding="utf-8"))
from agent import experiments as XP  # noqa: E402
from agent.orchestrator import live_cfg  # noqa: E402

CFG = live_cfg(CFG, OUT)                    # measured incrementality factors from concluded experiments
STORE_META = json.loads((OUT / "store_meta.json").read_text(encoding="utf-8")) if (OUT / "store_meta.json").exists() \
    else {"source": "synthetic"}


def platform() -> AdPlatform:
    return AdPlatform(CFG, OUT, persist=True)


def llm_client():
    if "llm" not in st.session_state:
        st.session_state.llm = make_llm(CFG, OUT)
    return st.session_state.llm


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def rs(x) -> str:
    return f"Rs {x:,.2f}"


# ---------------- citations ----------------
def w_sources() -> dict:
    rep = latest_report(OUT)
    ws = {s["id"]: s for s in (rep or {}).get("sources", {}).get("W", [])}
    if not ws:
        ws = {s["id"]: s for s in load_cached(CFG, OUT).get("sources", [])}
    return ws


def show_source(sid: str, w_map: dict, key: str):
    with st.popover(f"[{sid}]"):
        if sid.startswith("D"):
            s = load_source(sid, OUT)
            if not s:
                st.error(f"{sid} not found")
                return
            st.markdown(f"**{sid}: {s['question']}**")
            st.caption(f"code by {s['code_author']} · ran in the {s['sandbox_backend']} sandbox · {s['created_at']}"
                       f" · data through day {s.get('data_through_day')}")
            st.code(s["code"], language="python")
            st.dataframe(pd.DataFrame(s["result_table"]), width="stretch", hide_index=True)
            if s.get("result_summary"):
                st.json(s["result_summary"])
        else:
            s = w_map.get(sid)
            if not s:
                st.error(f"{sid} not found")
                return
            st.markdown(f"**{sid}: [{s['title']}]({s['url']})**")
            st.markdown(f"> {s['quote']}")
            st.caption(f"Fact: {s['fact']}")
            st.caption(f"Retrieved {s['retrieved_at']} · quote verified against the page text: {s['verified']}")


def citation_chips(text_or_ids, w_map: dict, key: str):
    ids = text_or_ids if isinstance(text_or_ids, list) else sorted(set(TAG_RE.findall(text_or_ids)))
    if not ids:
        return
    cols = st.columns(min(len(ids), 8))
    for i, sid in enumerate(ids):
        with cols[i % len(cols)]:
            show_source(sid, w_map, f"{key}_{sid}")


# ---------------- sidebar ----------------
p = platform()
llm = llm_client()
with st.sidebar:
    st.title("BidPilot")
    st.caption("AI bid manager + consultant · practice ad account (simulated)")
    st.metric("Account day", p.day)
    st.write("LLM:", "✅ connected" if llm else "⚪ off (--no-llm mode: templates, presets only)")
    st.caption(f"{CFG['llm']['model_planner']} @ {CFG['llm']['base_url']}")
    st.write("Research mode:", CFG["research"]["mode"])
    st.write("Rules profile:", CFG["platform_profile"]["label"])
    if STORE_META["source"] == "uci_online_retail":
        d0 = STORE_META["dates"].get(str(p.day), "")
        st.success(f"Real-data store: UCI Online Retail (real products, prices, daily demand). Day {p.day} = {d0}")
    else:
        st.caption("Store: synthetic demo shop")
    src = st.radio("Store data", ["Synthetic demo shop", "Real retailer (UCI Online Retail)"],
                   index=1 if STORE_META["source"] == "uci_online_retail" else 0)
    if st.button("Load this store (resets the account)"):
        new = "uci_online_retail" if src.startswith("Real") else "synthetic"
        gcfg = {**CFG, "store": {**CFG["store"], "source": new}}
        generate(gcfg, OUT)
        for f in list(OUT.glob("day_*_*.json")) + [OUT / "pending_consultant_actions.json", OUT / "race.csv",
                                                   OUT / "experiments.json", OUT / "incrementality.json"]:
            f.unlink(missing_ok=True)
        st.rerun()
    st.caption(f"Primary KPI {CFG['platform_profile']['primary_kpi'].upper()} · conversion lag "
               f"{CFG['guardrails']['conversion_lag_days']}d · attribution window "
               f"{CFG['platform_profile']['attribution_window_days']}d")
    if st.button("Re-check LLM"):
        st.session_state.pop("llm", None)
        st.rerun()
    st.divider()
    if st.button("Reset account to day 0"):
        reset(CFG, OUT)
        for f in list(OUT.glob("day_*_*.json")) + [OUT / "pending_consultant_actions.json"]:
            f.unlink(missing_ok=True)
        st.rerun()

tabs = st.tabs(["Store", "Run a day", "14-day race", "Consultant", "Ask the data", "Experiments",
                "Real ads audit", "Speed"])

# ---------------- 1. Store ----------------
with tabs[0]:
    products = p.get_products()
    ents = p.get_entities()
    sm = stock_metrics(p.get_stock_history(), p.day)
    t7 = account_totals(p.get_stats(), p.day, 7)
    c = st.columns(6)
    c[0].metric("Products", len(products))
    c[1].metric("Keywords", int((ents.entity_type == "keyword").sum()))
    c[2].metric("Spend (7d)", rs(t7["spend"]))
    c[3].metric("Revenue (7d)", rs(t7["revenue"]))
    c[4].metric("Profit (7d)", rs(t7["profit"]))
    low = sm[sm.days_of_stock < CFG["guardrails"]["low_stock_days"]]
    c[5].metric("Low stock", len(low))
    objectives = CFG.get("objectives") or {}
    view = products[["product_id", "name", "category", "price", "gross_margin_pct", "fee_pct", "return_rate",
                     "margin_pct"]].merge(
        sm.reset_index()[["product_id", "stock_units", "avg_daily_sales", "days_of_stock"]], on="product_id")
    if "source_id" in products:
        view.insert(1, "real_stock_code", products.source_id.astype(str))
    view["objective"] = view.product_id.map(
        lambda pid: f"growth (ACOS <= {objectives[pid]['max_acos']:.0%})"
        if (objectives.get(pid) or {}).get("mode") == "growth" else "profit")
    view["status"] = view.days_of_stock.map(
        lambda d: "OUT" if d <= 0 else "LOW" if d < CFG["guardrails"]["low_stock_days"]
        else "OVERSTOCK" if d > CFG["guardrails"]["overstock_days"] else "")

    def hl(row):
        color = {"OUT": "background-color: #f8d7da", "LOW": "background-color: #fff3cd",
                 "OVERSTOCK": "background-color: #d1ecf1"}.get(row.status, "")
        return [color] * len(row)
    st.caption("Contribution margin = gross margin - marketplace/fulfilment fees - return costs. All profit math "
               "uses contribution margin.")
    st.dataframe(view.rename(columns={"margin_pct": "contribution_margin"}).style.apply(hl, axis=1).format(
        {"price": "{:,.0f}", "gross_margin_pct": "{:.0%}", "fee_pct": "{:.0%}", "return_rate": "{:.0%}",
         "contribution_margin": "{:.0%}", "avg_daily_sales": "{:.2f}", "days_of_stock": "{:.1f}"}),
        width="stretch", hide_index=True, height=520)
    with st.expander("Seller objectives: choose products for a controlled growth push"):
        st.caption("Default objective is profit. A growth product buys the most sales whose expected ACOS stays "
                   "under your cap (e.g. a launch or clearing overstock). Bids still pass every guardrail.")
        names = dict(zip(products.product_id, products.name))
        cur_growth = [k for k, v in objectives.items() if (v or {}).get("mode") == "growth"]
        sel = st.multiselect("Growth products", list(products.product_id), default=cur_growth,
                             format_func=lambda x: f"{x} - {names[x]}")
        cur_cap = float(next(iter(objectives.values())).get("max_acos", 0.35)) if objectives else 0.35
        cap = st.slider("Max ACOS for growth products", 0.10, 0.80, cur_cap, 0.05)
        if st.button("Save objectives"):
            OBJ_PATH.write_text(json.dumps({pid: {"mode": "growth", "max_acos": cap} for pid in sel}, indent=1),
                                encoding="utf-8")
            st.rerun()

# ---------------- 2. Run a day ----------------
with tabs[1]:
    nxt = p.day + 1
    plan_path, report_path = OUT / f"day_{nxt}_plan.json", OUT / f"day_{nxt}_report.json"
    plan = read_json(plan_path) if not report_path.exists() else None
    pending = load_pending_actions(OUT)
    c1, c2, c3 = st.columns([1, 1, 2])
    if c1.button(f"Plan day {nxt}", type="primary"):
        with st.spinner("Workflows, planner and explanations..."):
            plan, _ = run_sync(prepare_day_async(p, CFG, llm, pending, OUT))
        st.rerun()
    ff = c2.number_input("Fast-forward days (auto-approve)", 1, 14, 1, label_visibility="collapsed")
    if c3.button(f"Fast-forward {ff} day(s) with auto-approve"):
        with st.spinner("Running..."):
            for _ in range(int(ff)):
                run_agent_day(p, auto_approve=True, pending_consultant_actions=load_pending_actions(OUT), cfg=CFG,
                              llm=llm, out=OUT)
        st.rerun()
    if pending:
        st.info(f"{len(pending)} accepted consultant action(s) will be included in the next plan.")

    if plan:
        st.subheader(f"Day {plan['day']} plan")
        st.markdown(f"**{len(plan['changes'])} changes, {plan['n_need_approval']} need your approval.**  "
                    f"{plan['day_note']}")
        if plan["focus"]:
            st.caption("Focus: " + " · ".join(plan["focus"]))
        if plan["held"]:
            with st.expander(f"{len(plan['held'])} change(s) held by the planner"):
                st.dataframe(pd.DataFrame(plan["held"]), hide_index=True, width="stretch")
        if plan["flags"]:
            with st.expander(f"Daily check-up: {len(plan['flags'])} mover(s)"):
                st.dataframe(pd.DataFrame(plan["flags"]), hide_index=True, width="stretch")
        ch = pd.DataFrame(plan["changes"])
        need = ch[ch.needs_approval]
        st.markdown("#### Needs approval")
        decisions = {}
        for r in need.itertuples():
            cols = st.columns([3, 1.3, 1.6, 5, 2])
            cols[0].markdown(f"**{r.name}**  \n`{r.entity_id}` · {r.workflow} · {getattr(r, 'role', '')} · "
                             f"{getattr(r, 'confidence', '')} confidence")
            cols[1].markdown(f"{r.action}  \n{r.old_bid:.2f} → {r.new_bid:.2f}")
            cols[2].caption(r.approval_reason)
            cols[3].write(r.reason)
            decisions[r.entity_id] = cols[4].radio("decision", ["approve", "reject", "undecided"], index=2,
                                                   key=f"dec_{plan['day']}_{r.entity_id}", horizontal=False,
                                                   label_visibility="collapsed")
        st.markdown("#### Applies automatically")
        auto = ch[~ch.needs_approval][[c for c in ["name", "workflow", "role", "confidence", "action", "old_bid",
                                                    "new_bid", "change_pct", "reason"] if c in ch.columns]]
        st.dataframe(auto.assign(change_pct=(auto.change_pct * 100).round(1)), hide_index=True,
                     width="stretch", height=300)
        if st.button("Apply decisions and run the day", type="primary"):
            dec = {k: v for k, v in decisions.items() if v != "undecided"}
            (OUT / f"day_{plan['day']}_decisions.json").write_text(json.dumps(dec, indent=1), encoding="utf-8")
            with st.spinner("Applying, simulating the day..."):
                finish_day(p, plan, cfg=CFG, llm=llm, out=OUT)
            st.rerun()
    last = read_json(OUT / f"day_{p.day}_report.json") if p.day > 0 else None
    if last:
        st.divider()
        st.subheader(f"Day {last['day']} report")
        st.write(last["report"])
        st.json(last["totals"], expanded=False)

# ---------------- 3. Race ----------------
with tabs[2]:
    race_path = OUT / "race.csv"
    strategies = ["fixed", "rules", "agent"] if llm else ["fixed", "rules", "agent_nollm"]
    if st.button(f"Run the 14-day race ({', '.join(strategies)})"):
        from sim.runner import compare
        with st.spinner("Simulating 14 days per strategy on fresh copies of the store..."):
            compare(strategies, cfg=CFG, out=OUT, llm=llm)
        st.rerun()
    if race_path.exists():
        from sim.runner import summarize
        race = pd.read_csv(race_path)
        metric = st.radio("Profit view", ["Total contribution profit (ads + organic sales)",
                                          "Ad-attributed contribution profit"], horizontal=True)
        y = "cum_total_profit" if metric.startswith("Total") and "cum_total_profit" in race else "cum_profit"
        fig = px.line(race, x="day", y=y, color="strategy", markers=True,
                      labels={y: "Cumulative profit (Rs)", "day": "Simulated day"},
                      title=f"14-day cumulative {metric.lower()} (simulation)")
        st.plotly_chart(fig, width="stretch")
        if "total_sales" in race:
            rows = pd.DataFrame([summarize(race[race.strategy == s]) for s in race.strategy.unique()])
            rows = rows[["strategy", "total_contribution_profit", "profit", "total_sales", "attributed_sales",
                         "spend", "acos", "roas", "cvr", "stock_availability", "stockout_ad_spend"]]
            st.dataframe(rows.rename(columns={"profit": "attributed_profit"}).style.format({
                "total_contribution_profit": "{:,.2f}", "attributed_profit": "{:,.2f}", "total_sales": "{:,.2f}",
                "attributed_sales": "{:,.2f}", "spend": "{:,.2f}", "acos": "{:.1%}", "roas": "{:.2f}",
                "cvr": "{:.2%}", "stock_availability": "{:.1%}", "stockout_ad_spend": "{:,.2f}"}),
                width="stretch", hide_index=True)
        st.caption("Same seed and common random numbers: every strategy sees identical market luck. This is a "
                   "simulated account; it shows the method works, not a guaranteed live-account result. "
                   "The goal is total contribution profit, not the lowest ACOS.")
    else:
        st.info("No race yet. Click the button above.")

# ---------------- 4. Consultant ----------------
with tabs[3]:
    rep = latest_report(OUT)
    w_map = w_sources()
    c1, c2 = st.columns([1, 3])
    if c1.button("Run consultant now", disabled=llm is None or p.day < 1):
        with st.spinner("Brief → plan → analytics + research in parallel → recommendations → citation check"):
            rep = run_consultant(p, p.day, CFG, llm, out=OUT)
        st.rerun()
    if llm is None:
        c2.caption("The consultant needs an LLM (skipped in --no-llm mode). Reports already in out/consult are shown.")
    if not rep:
        st.info("No consultant report yet. It runs automatically on days "
                f"{CFG['consultant']['run_on_days']} or on request.")
    else:
        st.subheader(f"Consultant report · day {rep['day']}")
        st.caption(f"Generated {rep['created_at']} · research: {rep['research']['mode']} mode, recorded "
                   f"{rep['research'].get('created_at')} · timings {rep['timings']}")
        decisions = {}
        for r in sorted(rep["recommendations"], key=lambda r: r.get("rank", 0)):
            verified = r["status"] == "verified"
            with st.container(border=True):
                rank = f"#{r['rank']} · " if r.get("rank") else ""
                st.markdown(f"### {rank}{r['title']}" + ("" if verified else "  ·  ⚠️ unverified"))
                st.caption(f"{r['id']} · priority {r.get('priority', '-')} · {r.get('confidence', '-')} confidence · "
                           f"{r['type']}" + (f" · action: {r['bid_action']}" if r.get("bid_action") else "")
                           + ("" if verified else f" · check failed: {r['check_errors']}"))
                body = (f"**Finding:** {r['finding']}\n\n**Recommendation:** {r['recommendation']}\n\n"
                        f"**Expected impact:** {r.get('expected_impact') or '-'}\n\n"
                        f"**Risk:** {r.get('risk') or '-'}\n\n**Next action:** {r.get('next_action') or '-'}")
                st.markdown(body if verified else f":gray[{body}]")
                d_ids = [c for c in r["citations"] if c.startswith("D")]
                w_ids = [c for c in r["citations"] if c.startswith("W")]
                if d_ids:
                    st.caption("Account data (sandbox analysis)")
                    citation_chips(d_ids, w_map, f"rec_{r['id']}_d")
                if w_ids:
                    st.caption("External evidence (web, quote verified)")
                    citation_chips(w_ids, w_map, f"rec_{r['id']}_w")
                if r.get("decision"):
                    st.success(f"Decision: {r['decision']}")
                else:
                    a, b, _ = st.columns([1, 1, 6])
                    if a.button("Accept", key=f"acc_{rep['day']}_{r['id']}"):
                        decide_recommendations(rep["day"], {r["id"]: "accept"}, OUT)
                        st.rerun()
                    if b.button("Reject", key=f"rej_{rep['day']}_{r['id']}"):
                        decide_recommendations(rep["day"], {r["id"]: "reject"}, OUT)
                        st.rerun()
        todos = read_json(OUT / "todos.json") or []
        st.subheader("Owner to-do list")
        if todos:
            st.dataframe(pd.DataFrame(todos), hide_index=True, width="stretch")
        else:
            st.caption("Nothing yet. Accepted owner to-dos (restock, pricing) appear here.")

# ---------------- 5. Ask the data ----------------
with tabs[4]:
    st.caption("The AI writes pandas code, a policy check approves it, it runs in the sandbox, and the result "
               "becomes a citable [D#] source.")
    w_map = w_sources()
    cols = st.columns(2)
    for i, q in enumerate(PRESET_QUESTIONS):
        if cols[i % 2].button(q, key=f"preset_{i}"):
            st.session_state.ask_q = q
            st.session_state.ask_force = False
    if RA.available():
        st.caption("Questions on REAL ad data (1,143 real Facebook ads from another company):")
        rcols = st.columns(2)
        for i, q in enumerate(REAL_ADS_QUESTIONS):
            if rcols[i % 2].button(q, key=f"real_{i}"):
                st.session_state.ask_q = q
                st.session_state.ask_force = False
    free = st.text_input("Or ask your own question" + ("" if llm else " (needs the LLM)"), disabled=llm is None)
    if st.button("Ask", disabled=not free):
        st.session_state.ask_q, st.session_state.ask_force = free, True
    q = st.session_state.get("ask_q")
    if q:
        src = None if st.session_state.get("ask_force") else latest_for(q, OUT)
        if src and src.get("data_through_day") != int(p.get_stats().day.max()):
            src = None
        if src is None:
            with st.spinner("Writing code → policy check → sandbox..."):
                src = ask(q, RA.with_real_ads(p.public_data()), CFG, llm=llm, out=OUT)
            st.session_state.ask_force = False
        st.markdown(f"**Q: {q}**")
        if not src.get("ok"):
            st.error(src.get("error"))
        else:
            st.markdown(src["answer"])
            citation_chips(src["answer"], w_map, "ask")
            st.caption(f"[{src['id']}] code by {src['code_author']} · {src['sandbox_backend']} sandbox · "
                       f"{src['created_at']}")
            st.code(src["code"], language="python")
            st.dataframe(pd.DataFrame(src["result_table"]), width="stretch", hide_index=True)

# ---------------- 6. Experiments ----------------
with tabs[5]:
    st.caption("Ad platforms credit every ad sale to the ad, but some of those sales would have happened anyway. "
               "A switchback test turns a product's ads ON and OFF on alternating days and compares total sales "
               "(ads + organic). Tests pause ads, so you approve them first.")
    exps = XP.load_experiments(OUT)
    prods = p.get_products()
    names = dict(zip(prods.product_id, prods.name))
    # suggestion: products with the most attributed ad sales AND strong organic sales are the likeliest to cannibalize
    st_last = p.get_stats()
    st_last = st_last[st_last.day > p.day - 14].merge(p.get_entities()[["entity_id", "product_id"]], on="entity_id")
    ad_units = st_last.groupby("product_id").conversions.sum()
    sk = p.get_stock_history()
    total_units = sk[sk.day > p.day - 14].groupby("product_id").units_sold.sum()
    sugg = pd.DataFrame({"ad_units_14d": ad_units, "total_units_14d": total_units}).fillna(0)
    sugg["organic_share"] = (1 - sugg.ad_units_14d / sugg.total_units_14d.where(sugg.total_units_14d > 0)).round(2)
    sugg = sugg[sugg.ad_units_14d >= 14].sort_values(["organic_share", "ad_units_14d"], ascending=False).head(5)
    c1, c2 = st.columns([3, 1])
    choice = c1.selectbox("Product to test", list(prods.product_id), format_func=lambda x: f"{x} - {names[x]}",
                          index=list(prods.product_id).index(sugg.index[0]) if len(sugg) else 0)
    if c2.button("Propose test"):
        try:
            XP.propose(choice, CFG, OUT, reason="proposed from dashboard")
        except ValueError as e:
            st.warning(str(e))
        st.rerun()
    if len(sugg):
        st.caption("Suggested (most ad sales with a high organic share): "
                   + ", ".join(f"{pid} ({names[pid]})" for pid in sugg.index))
    for e in reversed(exps):
        with st.container(border=True):
            st.markdown(f"**{e['id']} · {e['product_id']} {names.get(e['product_id'], '')}** · {e['status']}"
                        + (f" · starts day {e['start_day']}, {e['days']} days" if e.get("start_day") else ""))
            if e["status"] == "proposed":
                a, b, _ = st.columns([1, 1, 6])
                if a.button("Approve", key=f"xa_{e['id']}"):
                    XP.approve(e["id"], p.day + 1, CFG, OUT)
                    st.rerun()
                if b.button("Cancel", key=f"xc_{e['id']}"):
                    XP.cancel(e["id"], OUT)
                    st.rerun()
            elif e["status"] == "running":
                if st.button("Cancel test", key=f"xc_{e['id']}"):
                    XP.cancel(e["id"], OUT)
                    st.rerun()
            r = e.get("result") or {}
            if r.get("status") == "ok":
                st.markdown(f"Units/day with ads ON **{r['units_per_day_on']}** vs OFF **{r['units_per_day_off']}**; "
                            f"ads were credited with {r['attributed_per_day_on']}/day. Incrementality "
                            f"**{r['incrementality']:.0%}** (90% CI {r['ci90'][0]:.0%} to {r['ci90'][1]:.0%}) - "
                            f"{r.get('verdict', '')}.")
                if r.get("factor") is not None:
                    st.caption(f"Factor used by the optimizer: {r['factor']} (CI upper bound: conservative)")
            elif r:
                st.caption(f"Result so far: {r.get('status')} ({r.get('days_on', 0)} ON / {r.get('days_off', 0)} OFF days)")
            if e.get("schedule"):
                st.caption("Schedule: " + " ".join(f"d{k}:{v.upper()}" for k, v in sorted(e["schedule"].items(),
                                                                                     key=lambda kv: int(kv[0]))))
    factors = XP.load_factors(OUT)
    if factors:
        st.subheader("Measured incrementality in use")
        st.dataframe(pd.DataFrame(factors).T, width="stretch")

# ---------------- 7. Real ads audit ----------------
with tabs[6]:
    if not RA.available():
        st.info("No real ad export found (data/real/KAG_conversion_data.csv).")
    else:
        st.caption("Read-only audit of a REAL ad account export, using BidPilot's waste rule. Nothing is changed in "
                   "any account. Source: " + RA.SOURCE + ". Currency as in the source.")
        c = st.columns(3)
        min_spend = c[0].number_input("Min spend per ad", 0.0, 1000.0, 20.0, 5.0)
        min_clicks = c[1].number_input("Min clicks per ad", 0, 1000, 20, 5)
        conf = c[2].slider("Confidence that zero purchases is not bad luck", 0.80, 0.99, 0.95, 0.01)
        a = RA.audit(RA.load_real_ads(), min_spend, int(min_clicks), conf)
        m = st.columns(5)
        m[0].metric("Real ads", a["ads"])
        m[1].metric("Total spend", f"{a['total_spend']:,.2f}")
        m[2].metric("Purchases", a["total_purchases"])
        m[3].metric("Confident waste", f"{a['waste_spend']:,.2f}", f"{a['waste_share_pct']}% of spend", delta_color="off")
        m[4].metric("Too little data to judge", f"{a['low_data_spend']:,.2f}", f"{a['low_data_ads']} ads", delta_color="off")
        st.markdown(f"**Rule:** {a['rule']}. Ads with zero purchases but too little data are NOT flagged "
                    "(low-data protection).")
        st.subheader("By campaign")
        st.dataframe(a["by_campaign"], hide_index=True, width="stretch")
        fig = px.bar(a["by_campaign"], x="campaign_id", y="cost_per_purchase", text="cost_per_purchase",
                     title="Cost per purchase by campaign (real data)")
        st.plotly_chart(fig, width="stretch")
        st.subheader(f"Recommended to pause: {len(a['waste_ads'])} ads")
        cols = [c for c in ["ad_id", "campaign_id", "age", "gender", "interest", "impressions", "clicks", "spend",
                            "enquiries", "purchases", "p_zero"] if c in a["waste_ads"]]
        st.dataframe(a["waste_ads"][cols].round({"p_zero": 4}), hide_index=True, width="stretch")

# ---------------- 8. Speed ----------------
with tabs[7]:
    res = read_json(ROOT / "bench" / "results.json")
    if not res:
        st.info("No speed results yet. Run `python bench/latency.py` against the AMD GPU endpoint.")
    else:
        st.caption(f"Measured {res.get('measured_at')} · GPU: {res.get('gpu')} · model: {res.get('model')} · "
                   f"median of {res.get('runs')} runs")
        rows = pd.DataFrame(res["results"])
        st.dataframe(rows, hide_index=True, width="stretch")
        fig = px.bar(rows, x="scenario", y="wall_s", color="variant", barmode="group",
                     labels={"wall_s": "Wall time (s)"}, title="Naive vs BidPilot")
        st.plotly_chart(fig, width="stretch")
    if (OUT / "speed_chart.png").exists():
        st.image(str(OUT / "speed_chart.png"))
