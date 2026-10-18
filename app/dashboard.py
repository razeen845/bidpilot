"""BidPilot dashboard.  streamlit run app/dashboard.py

Pages: Your store | Today's bids | Results | Weekly advice | Ask a question | Experiments | Real ads check | Speed
Simple view by default (plain words); "Detailed view" in the sidebar adds IDs, code, statistics and full tables.
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
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from agent.analyst import PRESET_QUESTIONS, REAL_ADS_QUESTIONS, ask, latest_for, load_source  # noqa: E402
from agent.citations import TAG_RE  # noqa: E402
from agent.consultant import decide_recommendations, latest_report, run_consultant  # noqa: E402
from agent.llm import make_llm, run_sync  # noqa: E402
from agent.orchestrator import finish_day, load_pending_actions, prepare_day_async, run_agent_day  # noqa: E402
from agent.research import load_cached  # noqa: E402
from analysis import real_ads as RA  # noqa: E402
from analysis.metrics import account_totals, stock_metrics  # noqa: E402
from app import theme as T  # noqa: E402
from data.generate import generate  # noqa: E402
from settings import load_config, out_dir  # noqa: E402
from sim.platform import AdPlatform, reset  # noqa: E402

st.set_page_config(page_title="BidPilot", page_icon=":material/storefront:", layout="wide")
T.inject()

CFG = load_config()
OUT = out_dir()
if not (OUT / "products.csv").exists():
    generate(CFG, OUT)
OBJ_PATH = OUT / "objectives.json"          # seller-chosen objectives (Your store page)
if OBJ_PATH.exists():
    CFG["objectives"] = json.loads(OBJ_PATH.read_text(encoding="utf-8"))
from agent import experiments as XP  # noqa: E402
from agent.orchestrator import live_cfg  # noqa: E402

CFG = live_cfg(CFG, OUT)                    # measured incrementality factors from concluded experiments
STORE_META = json.loads((OUT / "store_meta.json").read_text(encoding="utf-8")) if (OUT / "store_meta.json").exists() \
    else {"source": "synthetic"}
SIM_DAYS = int(CFG["store"]["sim_days"])
LOW, OVER = CFG["guardrails"]["low_stock_days"], CFG["guardrails"]["overstock_days"]
WHY = {"pause": "pauses an ad", "top": "top spender", "proven": "cuts a best seller", "bid": "big change"}


def platform() -> AdPlatform:
    return AdPlatform(CFG, OUT, persist=True)


def llm_client():
    if "llm" not in st.session_state:
        st.session_state.llm = make_llm(CFG, OUT)
    return st.session_state.llm


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def plain_why(reason: str | None) -> list[str]:
    """approval_reason ("bid change 20% > 10%; top-10 spender") -> plain tags."""
    r = (reason or "").lower()
    out = []
    if "pause" in r:
        out.append("pauses an ad")
    if "bid change" in r:
        out.append("big change")
    if "spender" in r:
        out.append("top spender")
    if "proven" in r:
        out.append("cuts a best seller")
    return out or ["needs a check"]


def clear_store_files():
    for f in list(OUT.glob("day_*_*.json")) + [OUT / "pending_consultant_actions.json", OUT / "race.csv",
                                               OUT / "experiments.json", OUT / "incrementality.json"]:
        f.unlink(missing_ok=True)


# ---------------- sources ([D#] data, [W#] web) ----------------
def w_sources() -> dict:
    rep = latest_report(OUT)
    ws = {s["id"]: s for s in (rep or {}).get("sources", {}).get("W", [])}
    if not ws:
        ws = {s["id"]: s for s in load_cached(CFG, OUT).get("sources", [])}
    return ws


def show_source(sid: str, w_map: dict, detailed: bool):
    with st.popover(f"[{sid}]", help="Open the proof behind this claim"):
        if sid.startswith("D"):
            s = load_source(sid, OUT)
            if not s:
                st.error(f"Source {sid} is missing from out/analysis. Re-run the question to rebuild it.")
                return
            st.markdown(f"**{sid}: {s['question']}**")
            st.caption("Computed from your account data in a locked sandbox.")
            st.dataframe(pd.DataFrame(s["result_table"]), width="stretch", hide_index=True)
            with st.expander("Show the code", expanded=detailed):
                st.code(s["code"], language="python")
                st.caption(f"Code by: {s['code_author']}. Sandbox: {s['sandbox_backend']}. Ran {s['created_at']}, "
                           f"data up to day {s.get('data_through_day')}.")
        else:
            s = w_map.get(sid)
            if not s:
                st.error(f"Source {sid} is missing from the research cache.")
                return
            st.markdown(f"**{sid}: [{s['title']}]({s['url']})**")
            st.markdown(f"> {s['quote']}")
            st.caption(f"Fact: {s['fact']}")
            st.caption(f"Page read on {s['retrieved_at'][:10]}. The exact quote was found on the page.")


def citation_chips(text_or_ids, w_map: dict, detailed: bool):
    ids = text_or_ids if isinstance(text_or_ids, list) else sorted(set(TAG_RE.findall(text_or_ids)))
    if not ids:
        return
    cols = st.columns(min(len(ids), 8))
    for i, sid in enumerate(ids):
        with cols[i % len(cols)]:
            show_source(sid, w_map, detailed)


# ---------------- state shared by the banner and the pages ----------------
p = platform()
llm = llm_client()
nxt = p.day + 1
report_next = OUT / f"day_{nxt}_report.json"
plan = read_json(OUT / f"day_{nxt}_plan.json") if not report_next.exists() else None
waiting = sum(c["needs_approval"] for c in plan["changes"]) if plan else 0
rep = latest_report(OUT)
real_store = STORE_META["source"] == "uci_online_retail"

# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown("## BidPilot")
    st.caption("AI bid manager and consultant for product and keyword ads. Runs on a practice ad account, "
               "so no real money is spent.")
    st.metric("Account day", p.day, help=f"The practice account moves one day forward each time you apply a day's "
                                         f"plan. The demo covers {SIM_DAYS} days.")
    detailed = st.toggle("Detailed view", value=False,
                         help="Off: plain words and the essentials. On: IDs, code, statistics and full tables.")
    if llm:
        st.markdown(f"{T.tag('AI connected', 'up')}", unsafe_allow_html=True)
        st.caption(f"Model {CFG['llm']['model_planner']}")
    else:
        st.markdown(f"{T.tag('AI off', 'pause')}", unsafe_allow_html=True)
        st.caption("Bids still work. Explanations use plain templates and weekly advice waits for the AI.")
    if real_store:
        st.caption(f"Store: a real UK retailer's products, prices and daily demand (UCI Online Retail). "
                   f"Day {p.day} is {STORE_META['dates'].get(str(p.day), '')}.")
    else:
        st.caption("Store: demo home-textiles shop.")
    with st.expander("What the words mean"):
        for term, meaning in T.GLOSSARY.items():
            st.markdown(f"**{term}**: {meaning}")
    with st.expander("Settings"):
        if st.button("Check the AI connection again", width="stretch"):
            st.session_state.pop("llm", None)
            st.rerun()
        src = st.radio("Store data", ["Demo shop", "Real retailer (UCI Online Retail)"],
                       index=1 if real_store else 0)
        with st.popover("Switch to this store", width="stretch"):
            st.warning("This rebuilds the store and puts the account back to day 0. Plans, race results and "
                       "experiments are cleared.")
            if st.button("Yes, switch store", type="primary"):
                new = "uci_online_retail" if src.startswith("Real") else "synthetic"
                generate({**CFG, "store": {**CFG["store"], "source": new}}, OUT)
                clear_store_files()
                st.rerun()
        with st.popover("Reset account to day 0", width="stretch"):
            st.warning("This undoes every applied day. The store itself stays the same.")
            if st.button("Yes, reset", type="primary"):
                reset(CFG, OUT)
                for f in list(OUT.glob("day_*_*.json")) + [OUT / "pending_consultant_actions.json"]:
                    f.unlink(missing_ok=True)
                st.rerun()
        if detailed:
            st.caption(f"Rules profile: {CFG['platform_profile']['label']}. Main KPI "
                       f"{CFG['platform_profile']['primary_kpi'].upper()}, conversion lag "
                       f"{CFG['guardrails']['conversion_lag_days']} days. Research: {CFG['research']['mode']} mode. "
                       f"AI endpoint: {CFG['llm']['base_url']}")

# ---------------- banner: system status at a glance ----------------
chips = [T.chip(f"Day {p.day} of {SIM_DAYS}", "ok"),
         T.chip("AI connected" if llm else "AI off (bids still work)", "ok" if llm else "off"),
         T.chip("Real retailer data" if real_store else "Demo shop", "ok"),
         T.chip(f"{waiting} change{'s' if waiting != 1 else ''} waiting for you" if waiting else "Nothing waiting",
                "warn" if waiting else "ok")]
T.hero("BidPilot", "Sets your ad bids with maths, explains every change in plain words, and asks you before "
                   "anything big. Every number links to its proof.", chips)

if plan and waiting:
    T.next_step(f"<b>Next step:</b> {waiting} change{'s' if waiting != 1 else ''} for day {plan['day']} "
                f"{'need' if waiting != 1 else 'needs'} your decision on <b>Today's bids</b>.")
elif not plan and p.day < SIM_DAYS:
    T.next_step(f"<b>Next step:</b> open <b>Today's bids</b> and plan day {nxt}.")
elif rep is None and p.day >= 7 and llm:
    T.next_step("<b>Next step:</b> your weekly advice is ready to write on <b>Weekly advice</b>.")

tabs = st.tabs([":material/storefront: Your store", ":material/event_available: Today's bids",
                ":material/trending_up: Results", ":material/lightbulb: Weekly advice",
                ":material/forum: Ask a question", ":material/science: Experiments",
                ":material/fact_check: Real ads check", ":material/speed: Speed"])

# ======================= 1. Your store =======================
with tabs[0]:
    st.markdown("### Your store")
    T.lead("Your shop at a glance: what sells, what is running out, and what you spend on ads.")
    products, ents = p.get_products(), p.get_entities()
    sm = stock_metrics(p.get_stock_history(), p.day)
    t7 = account_totals(p.get_stats(), p.day, 7)
    low = sm[sm.days_of_stock < LOW]
    c = st.columns(3)
    c[0].metric("Products", len(products))
    c[1].metric("Keywords", int((ents.entity_type == "keyword").sum()), help="Search words your ads appear for.")
    c[2].metric("Running low", len(low), help=f"Products with less than {LOW:g} days of stock left.")
    c = st.columns(3)
    c[0].metric("Ad spend, last 7 days", T.rs(t7["spend"]))
    c[1].metric("Ad sales, last 7 days", T.rs(t7["revenue"]))
    c[2].metric("Profit, last 7 days", T.rs(t7["profit"]), help=T.GLOSSARY["Contribution profit"])

    objectives = CFG.get("objectives") or {}

    def status(d):
        if d <= 0:
            return "🔴 Out of stock"
        if d < LOW:
            return "🟠 Running low"
        if d > OVER:
            return "🔵 Too much stock"
        return "🟢 Healthy"

    view = products.drop(columns=["stock_units"], errors="ignore").merge(   # current stock, not the starting stock
        sm.reset_index()[["product_id", "stock_units", "avg_daily_sales", "days_of_stock"]], on="product_id")
    view["Status"] = view.days_of_stock.map(status)
    view["Goal"] = view.product_id.map(lambda pid: f"More sales (ad cost up to {objectives[pid]['max_acos']:.0%})"
                                       if (objectives.get(pid) or {}).get("mode") == "growth" else "Most profit")
    show = view.rename(columns={"name": "Product", "category": "Category", "price": "Price (Rs)",
                                "stock_units": "Units in stock", "days_of_stock": "Days of stock"})
    cols = ["Product", "Category", "Price (Rs)", "Units in stock", "Days of stock", "Status", "Goal"]
    if detailed:
        show = show.rename(columns={"product_id": "ID", "gross_margin_pct": "Gross margin", "fee_pct": "Fees",
                                    "return_rate": "Returns", "margin_pct": "Contribution margin",
                                    "avg_daily_sales": "Sold per day"})
        cols = ["ID"] + (["source_id"] if "source_id" in show else []) + cols[:4] + [
            "Sold per day", "Days of stock", "Gross margin", "Fees", "Returns", "Contribution margin", "Status", "Goal"]
    st.dataframe(show[cols], width="stretch", hide_index=True, height=460, column_config={
        "Days of stock": st.column_config.ProgressColumn("Days of stock", help=T.GLOSSARY["Days of stock"],
                                                         min_value=0, max_value=60, format="%.1f"),
        "Price (Rs)": st.column_config.NumberColumn(format="%,.0f"),
        "Gross margin": st.column_config.NumberColumn(format="percent"),
        "Fees": st.column_config.NumberColumn(format="percent"),
        "Returns": st.column_config.NumberColumn(format="percent"),
        "Contribution margin": st.column_config.NumberColumn(format="percent"),
        "source_id": st.column_config.TextColumn("Real stock code"),
    })
    with st.expander("Push a product for more sales"):
        st.markdown("By default BidPilot bids for the **most profit**. Pick products you want to grow (a launch, or "
                    "too much stock) and the highest ad cost you accept; BidPilot then buys the most sales within "
                    "that limit. Every safety rule still applies.")
        names = dict(zip(products.product_id, products.name))
        cur_growth = [k for k, v in objectives.items() if (v or {}).get("mode") == "growth"]
        sel = st.multiselect("Growth products", list(products.product_id), default=cur_growth,
                             format_func=lambda x: f"{names[x]} ({x})")
        cur_cap = float(next(iter(objectives.values())).get("max_acos", 0.35)) if objectives else 0.35
        cap = st.slider("Highest ad cost you accept, as a share of ad sales (ACOS)", 0.10, 0.80, cur_cap, 0.05,
                        format="%.2f", help=T.GLOSSARY["ACOS"])
        if st.button("Save objectives"):
            OBJ_PATH.write_text(json.dumps({pid: {"mode": "growth", "max_acos": cap} for pid in sel}, indent=1),
                                encoding="utf-8")
            st.toast(f"Saved: {len(sel)} growth product{'s' if len(sel) != 1 else ''}.")
            st.rerun()

# ======================= 2. Today's bids =======================
with tabs[1]:
    st.markdown("### Today's bids")
    T.lead("Each day BidPilot proposes new bids. Small changes apply by themselves; big ones wait for your OK.")
    pending = load_pending_actions(OUT)
    if pending:
        st.info(f"{len(pending)} weekly-advice action{'s' if len(pending) != 1 else ''} you accepted will be "
                f"included in the next plan.")
    if not plan:
        if p.day >= SIM_DAYS:
            st.success(f"All {SIM_DAYS} demo days are done. See **Results**, or reset the account in Settings.")
        else:
            T.stitch("<b>What happens when you plan a day:</b> the maths checks every ad's results, proposes the "
                     "most profitable bid within safe limits, protects products that are running out, and the AI "
                     "writes a one-line reason for each change. Nothing is applied until you press Apply.")
            if st.button(f"Plan day {nxt}", type="primary"):
                with st.spinner("Checking every ad, choosing bids and writing the reasons..."):
                    run_sync(prepare_day_async(p, CFG, llm, pending, OUT))
                st.rerun()
        with st.expander("Skip ahead several days"):
            ff = st.number_input("Days to run", 1, SIM_DAYS, 1)
            st.caption("Every change is approved automatically. Use this to reach the weekly advice on day 7.")
            if st.button(f"Run {ff} day{'s' if ff != 1 else ''} automatically", disabled=p.day >= SIM_DAYS):
                with st.spinner("Running..."):
                    for _ in range(int(min(ff, SIM_DAYS - p.day))):
                        run_agent_day(p, auto_approve=True, pending_consultant_actions=load_pending_actions(OUT),
                                      cfg=CFG, llm=llm, out=OUT)
                st.rerun()
    else:
        ch = pd.DataFrame(plan["changes"])
        need = ch[ch.needs_approval] if len(ch) else ch
        auto = ch[~ch.needs_approval] if len(ch) else ch
        m = st.columns(4)
        m[0].metric("Changes proposed", len(ch))
        m[1].metric("Waiting for you", len(need), help=T.GLOSSARY["Approval"])
        m[2].metric("Apply automatically", len(auto))
        m[3].metric("Held back by the AI", len(plan["held"]),
                    help="Changes the AI planner chose to hold because the data looked thin.")
        T.stitch(f"<b>Plan for day {plan['day']}:</b> {T.esc(plan['day_note'])}")
        if detailed and (plan["focus"] or plan["flags"] or plan["held"]):
            if plan["focus"]:
                st.caption("AI focus: " + "; ".join(plan["focus"]))
            if plan["held"]:
                with st.expander(f"{len(plan['held'])} held by the AI planner"):
                    st.dataframe(pd.DataFrame(plan["held"]), hide_index=True, width="stretch")
            if plan["flags"]:
                with st.expander(f"Daily check-up: {len(plan['flags'])} ads moved more than 30% week on week"):
                    st.dataframe(pd.DataFrame(plan["flags"]), hide_index=True, width="stretch")

        st.markdown(f"#### Waiting for you ({len(need)})")
        keys = {r.entity_id: f"dec_{plan['day']}_{r.entity_id}" for r in need.itertuples()}

        def set_all(value):
            for k in keys.values():
                st.session_state[k] = value

        if len(need):
            b = st.columns([1, 1, 1, 3])
            b[0].button("Approve all", on_click=set_all, args=("approve",), width="stretch")
            b[1].button("Reject all", on_click=set_all, args=("reject",), width="stretch")
            b[2].button("Clear choices", on_click=set_all, args=("undecided",), width="stretch")
        decisions = {}
        labels = {"approve": "Approve", "reject": "Reject", "undecided": "Decide later"}
        for r in need.itertuples():
            with st.container(border=True):
                top = st.columns([3, 2])
                tags = "".join(T.tag(w, "you") for w in plain_why(r.approval_reason))
                conf = getattr(r, "confidence", "")
                if conf:
                    tags += T.tag(f"{conf} confidence", "info")
                top[0].markdown(f"**{T.esc(r.name)}** {T.tag(r.entity_type, 'pause')}<br>{tags}",
                                unsafe_allow_html=True)
                top[0].markdown(T.bid_move(r.action, r.old_bid, r.new_bid, r.change_pct), unsafe_allow_html=True)
                top[0].write(r.reason)
                if keys[r.entity_id] not in st.session_state:
                    st.session_state[keys[r.entity_id]] = "undecided"
                decisions[r.entity_id] = top[1].radio(
                    f"Decision for {r.name}", list(labels), format_func=labels.get, key=keys[r.entity_id],
                    horizontal=True, label_visibility="collapsed")
                if detailed:
                    top[1].caption(f"ID {r.entity_id}, workflow {r.workflow}, role {getattr(r, 'role', '-')}. "
                                   f"Why approval: {r.approval_reason}")
                    with top[1].expander("Evidence"):
                        st.json(r.evidence)
        n_app = sum(v == "approve" for v in decisions.values())
        n_rej = sum(v == "reject" for v in decisions.values())
        n_und = len(decisions) - n_app - n_rej
        with st.expander(f"{len(auto)} small changes apply automatically"):
            if len(auto):
                small = pd.DataFrame({
                    "Ad": auto.name,
                    "Change": [("Pause" if a == "pause" else "Resume" if a == "resume"
                                else f"{'Raise' if n > o else 'Lower'} {abs(cp) * 100:.0f}%")
                               for a, o, n, cp in zip(auto.action, auto.old_bid, auto.new_bid, auto.change_pct)],
                    "From (Rs)": auto.old_bid, "To (Rs)": auto.new_bid, "Why": auto.reason})
                if detailed:
                    small["Workflow"] = auto.workflow.values
                    small["Confidence"] = auto["confidence"].values if "confidence" in auto else ""
                st.dataframe(small, hide_index=True, width="stretch", height=300)
        st.caption(f"{n_app} approved, {n_rej} rejected, {n_und} to decide later. Changes left undecided are "
                   f"skipped today.")
        if st.button(f"Apply decisions and run day {plan['day']}", type="primary"):
            dec = {k: v for k, v in decisions.items() if v != "undecided"}
            (OUT / f"day_{plan['day']}_decisions.json").write_text(json.dumps(dec, indent=1), encoding="utf-8")
            with st.spinner("Applying your decisions and running the day..."):
                finish_day(p, plan, cfg=CFG, llm=llm, out=OUT)
            st.toast(f"Day {plan['day']} done: {len(auto) + n_app} changes applied.")
            st.rerun()

    last = read_json(OUT / f"day_{p.day}_report.json") if p.day > 0 else None
    if last:
        st.divider()
        st.markdown(f"#### How day {last['day']} went")
        t = last["totals"]
        m = st.columns(4)
        m[0].metric("Ad spend", T.rs(t["spend"]))
        m[1].metric("Ad sales", T.rs(t["revenue"]))
        m[2].metric("Profit from ads", T.rs(t["profit"]), help=T.GLOSSARY["Contribution profit"])
        m[3].metric("Spent on out-of-stock products", T.rs(t.get("stockout_ad_spend", 0)))
        st.write(last["report"])
        if detailed:
            st.json(last, expanded=False)

# ======================= 3. Results =======================
with tabs[2]:
    st.markdown("### Results")
    T.lead("The same shop, the same 14 days and the same luck, bid three ways: changing nothing, simple rules, "
           "and BidPilot.")
    race_path = OUT / "race.csv"
    strategies = ["fixed", "rules", "agent"] if llm else ["fixed", "rules", "agent_nollm"]
    if st.button("Run the 14-day race again" if race_path.exists() else "Run the 14-day race"):
        from sim.runner import compare
        with st.spinner("Running 14 days three times on fresh copies of the store..."):
            compare(strategies, cfg=CFG, out=OUT, llm=llm)
        st.rerun()
    if race_path.exists():
        from sim.runner import summarize
        race = pd.read_csv(race_path)
        race["Strategy"] = race.strategy.map(T.STRATEGY_NAMES)
        sums = {T.STRATEGY_NAMES[s]: summarize(race[race.strategy == s]) for s in race.strategy.unique()}
        bp, fx, rl = sums.get("BidPilot"), sums.get("No changes (fixed bids)"), sums.get("Simple rules")
        if bp and fx and rl:
            gain_fx = bp["total_contribution_profit"] - fx["total_contribution_profit"]
            gain_rl = bp["total_contribution_profit"] - rl["total_contribution_profit"]
            st.markdown(f'<div class="bp-title" style="font-size:1.45rem;margin:6px 0 2px">BidPilot earned '
                        f'{T.rs(gain_rl)} more than simple rules and {T.rs(gain_fx)} more than changing nothing.'
                        f'</div>', unsafe_allow_html=True)
            st.caption(f"Total profit over 14 days: BidPilot {T.rs(bp['total_contribution_profit'])}, simple rules "
                       f"{T.rs(rl['total_contribution_profit'])}, no changes {T.rs(fx['total_contribution_profit'])}. "
                       f"Ad money spent on out-of-stock products: BidPilot {T.rs(bp['stockout_ad_spend'])}.")
        view_ = st.radio("Show profit from", ["All sales (ads and organic)", "Ad sales only"], horizontal=True,
                         help=T.GLOSSARY["Contribution profit"])
        y = "cum_total_profit" if view_.startswith("All") and "cum_total_profit" in race else "cum_profit"
        fig = px.line(race, x="day", y=y, color="Strategy", markers=True, color_discrete_map=T.STRATEGY_COLORS,
                      labels={y: "Profit so far (Rs)", "day": "Day"})
        fig.update_traces(line_width=2.5)
        fig.for_each_trace(lambda tr: tr.update(line_width=4) if tr.name == "BidPilot" else None)
        st.plotly_chart(T.style_fig(fig, 400), width="stretch")
        rows = pd.DataFrame([{"Strategy": k, **v} for k, v in sums.items()])
        simple_cols = {"Strategy": "Strategy", "total_contribution_profit": "Total profit (Rs)",
                       "spend": "Ad spend (Rs)", "stockout_ad_spend": "Spent on out-of-stock (Rs)",
                       "stock_availability": "Products in stock"}
        more = {"profit": "Profit from ads (Rs)", "total_sales": "All sales (Rs)", "attributed_sales": "Ad sales (Rs)",
                "acos": "ACOS", "roas": "ROAS", "cvr": "Conversion rate"}
        cols = {**simple_cols, **(more if detailed else {})}
        st.dataframe(rows[list(cols)].rename(columns=cols), hide_index=True, width="stretch", column_config={
            c: st.column_config.NumberColumn(format="%,.0f") for c in cols.values() if "(Rs)" in c} | {
            "Products in stock": st.column_config.NumberColumn(format="percent"),
            "ACOS": st.column_config.NumberColumn(format="percent", help=T.GLOSSARY["ACOS"]),
            "Conversion rate": st.column_config.NumberColumn(format="percent"),
            "ROAS": st.column_config.NumberColumn(format="%.2f", help=T.GLOSSARY["ROAS"])})
        st.caption("This is a practice account: it shows the method works, not a promise for a real account. "
                   "BidPilot sells a little less than changing nothing because it stops buying sales that lose money.")
    else:
        st.info("No results yet. Press **Run the 14-day race** to compare the three ways of bidding.")

# ======================= 4. Weekly advice =======================
with tabs[3]:
    st.markdown("### Weekly advice")
    T.lead("Once a week BidPilot studies your data and the web and suggests what to do next. Every claim links "
           "to its proof; anything it cannot prove is greyed out.")
    w_map = w_sources()
    if st.button("Write this week's advice now", disabled=llm is None or p.day < 1,
                 help="Needs the AI and at least one finished day."):
        with st.spinner("Studying your data and the web at the same time, then checking every claim..."):
            rep = run_consultant(p, p.day, CFG, llm, out=OUT)
        st.rerun()
    if llm is None:
        st.caption("The AI is off, so new advice cannot be written. Earlier advice is still shown below.")
    if not rep:
        st.info(f"No advice yet. It is written automatically on days {', '.join(map(str, CFG['consultant']['run_on_days']))}, "
                f"or press the button above.")
    else:
        st.caption(f"Written on day {rep['day']}." + (f" Generated {rep['created_at']}; research {rep['research']['mode']} "
                                                       f"mode; timings {rep['timings']}." if detailed else ""))
        for r in sorted(rep["recommendations"], key=lambda r: r.get("rank", 0)):
            verified = r["status"] == "verified"
            with st.container(border=True):
                st.markdown(f"#### {r.get('rank', '')}. {r['title']}")
                tags = T.tag("Bid change BidPilot can make" if r["type"] == "bid_action" else "To-do for you", "info")
                tags += T.tag(f"{r.get('confidence', 'medium')} confidence", "info")
                if not verified:
                    tags += T.tag("Not proven: for reference only", "down")
                st.markdown(tags, unsafe_allow_html=True)
                body = (f"**What we found:** {r['finding']}\n\n**What to do:** {r['recommendation']}\n\n"
                        f"**Expected effect:** {r.get('expected_impact') or 'not stated'}\n\n"
                        f"**Risk:** {r.get('risk') or 'none stated'}\n\n**Next step:** {r.get('next_action') or '-'}")
                st.markdown(body if verified else f":gray[{body}]")
                d_ids = [c for c in r["citations"] if c.startswith("D")]
                w_ids = [c for c in r["citations"] if c.startswith("W")]
                if d_ids:
                    st.caption("Proof from your data")
                    citation_chips(d_ids, w_map, detailed)
                if w_ids:
                    st.caption("Proof from the web (exact quote checked)")
                    citation_chips(w_ids, w_map, detailed)
                if detailed:
                    st.caption(f"{r['id']}, priority {r.get('priority', '-')}"
                               + (f", bid action {r['bid_action']}" if r.get("bid_action") else "")
                               + ("" if verified else f", check failed: {r['check_errors']}"))
                if r.get("decision"):
                    (st.success if r["decision"] == "accept" else st.info)(
                        "Accepted" if r["decision"] == "accept" else "Rejected")
                else:
                    a, b_, _ = st.columns([1, 1, 5])
                    if a.button("Accept", key=f"acc_{rep['day']}_{r['id']}", type="primary" if verified else "secondary"):
                        decide_recommendations(rep["day"], {r["id"]: "accept"}, OUT)
                        st.toast("Accepted. Bid changes join tomorrow's plan; other advice goes to your to-do list.")
                        st.rerun()
                    if b_.button("Reject", key=f"rej_{rep['day']}_{r['id']}"):
                        decide_recommendations(rep["day"], {r["id"]: "reject"}, OUT)
                        st.rerun()
        todos = read_json(OUT / "todos.json") or []
        st.markdown("#### Your to-do list")
        if todos:
            td = pd.DataFrame(todos)
            st.dataframe(td if detailed else td[[c for c in ["title", "todo", "next_action"] if c in td]].rename(
                columns={"title": "Topic", "todo": "What to do", "next_action": "Next step"}),
                hide_index=True, width="stretch")
        else:
            st.caption("Empty. Advice you accept that BidPilot cannot do itself (restock, change a price) lands here.")

# ======================= 5. Ask a question =======================
with tabs[4]:
    st.markdown("### Ask a question")
    T.lead("Ask about your ads in plain words. Every answer comes with the table, and the code, that produced it.")
    w_map = w_sources()
    st.caption("Ready-made questions about your shop")
    cols = st.columns(2)
    for i, q in enumerate(PRESET_QUESTIONS):
        if cols[i % 2].button(q, key=f"preset_{i}", width="stretch"):
            st.session_state.ask_q, st.session_state.ask_force = q, False
    if RA.available():
        st.caption("Questions about real ad data (1,143 real Facebook ads from another company)")
        rcols = st.columns(2)
        for i, q in enumerate(REAL_ADS_QUESTIONS):
            if rcols[i % 2].button(q, key=f"real_{i}", width="stretch"):
                st.session_state.ask_q, st.session_state.ask_force = q, False
    with st.form("ask_form", clear_on_submit=False, border=False):
        free = st.text_input("Or type your own question", disabled=llm is None,
                             placeholder="For example: which 5 keywords cost the most per click last week?"
                             if llm else "Your own questions need the AI (it is off)")
        if st.form_submit_button("Ask", disabled=llm is None) and free.strip():
            st.session_state.ask_q, st.session_state.ask_force = free.strip(), True
    q = st.session_state.get("ask_q")
    if q:
        src = None if st.session_state.get("ask_force") else latest_for(q, OUT)
        if src and src.get("data_through_day") != int(p.get_stats().day.max()):
            src = None
        if src is None:
            with st.spinner("Working it out in a locked sandbox..."):
                src = ask(q, RA.with_real_ads(p.public_data()), CFG, llm=llm, out=OUT)
            st.session_state.ask_force = False
        with st.container(border=True):
            st.markdown(f"**{q}**")
            if not src.get("ok"):
                st.error(f"No answer: {src.get('error')}. Try rephrasing, or pick a ready-made question.")
            else:
                st.markdown(src["answer"])
                citation_chips(src["answer"], w_map, detailed)
                st.dataframe(pd.DataFrame(src["result_table"]), width="stretch", hide_index=True)
                with st.expander("Show the code", expanded=detailed):
                    st.code(src["code"], language="python")
                    st.caption(f"[{src['id']}] code by {src['code_author']}, {src['sandbox_backend']} sandbox, "
                               f"{src['created_at']}.")

# ======================= 6. Experiments =======================
with tabs[5]:
    st.markdown("### Experiments")
    T.lead("Ad platforms give the ad credit for every sale, but some buyers would have bought anyway. An experiment "
           "finds out how many ad sales are really extra.")
    T.stitch("<b>How a test works:</b> for 14 days one product's ads switch on and off on alternating days. "
             "BidPilot compares total sales on the two kinds of day. If the ads add fewer sales than the platform "
             "claims, BidPilot bids less on that product. Ads are off on some days, so you approve every test first.")
    exps = XP.load_experiments(OUT)
    prods = p.get_products()
    names = dict(zip(prods.product_id, prods.name))
    st_last = p.get_stats()
    st_last = st_last[st_last.day > p.day - 14].merge(p.get_entities()[["entity_id", "product_id"]], on="entity_id")
    ad_units = st_last.groupby("product_id").conversions.sum()
    sk = p.get_stock_history()
    total_units = sk[sk.day > p.day - 14].groupby("product_id").units_sold.sum()
    sugg = pd.DataFrame({"ad_units_14d": ad_units, "total_units_14d": total_units}).fillna(0)
    sugg["organic_share"] = (1 - sugg.ad_units_14d / sugg.total_units_14d.where(sugg.total_units_14d > 0)).round(2)
    sugg = sugg[sugg.ad_units_14d >= 14].sort_values(["organic_share", "ad_units_14d"], ascending=False).head(5)
    c1, c2 = st.columns([3, 1])
    choice = c1.selectbox("Product to test", list(prods.product_id), format_func=lambda x: f"{names[x]} ({x})",
                          index=list(prods.product_id).index(sugg.index[0]) if len(sugg) else 0)
    c2.write("")
    if c2.button("Propose test", width="stretch"):
        try:
            XP.propose(choice, CFG, OUT, reason="proposed from dashboard")
            st.toast(f"Test proposed for {names[choice]}. Approve it below to start.")
        except ValueError as e:
            st.warning(str(e))
        st.rerun()
    if len(sugg):
        st.caption("Good candidates (most ad sales and many buyers who also find them without ads): "
                   + ", ".join(names[pid] for pid in sugg.index))
    for e in reversed(exps):
        with st.container(border=True):
            label = {"proposed": ("Waiting for your approval", "you"), "running": ("Running", "up"),
                     "done": ("Finished", "info"), "cancelled": ("Cancelled", "pause")}[e["status"]]
            st.markdown(f"**{T.esc(names.get(e['product_id'], e['product_id']))}** {T.tag(*label)}",
                        unsafe_allow_html=True)
            if e.get("schedule"):
                cells = "".join(
                    f'<span title="Day {d}: ads {s.upper()}" style="display:inline-block;width:20px;height:20px;'
                    f'margin:0 3px 3px 0;border-radius:4px;'
                    + ("background:#1E2A78;" if s == "on" else "background:repeating-linear-gradient(45deg,#fff 0 3px,"
                       "#C9CEE6 3px 6px);border:1px solid #C9CEE6;")
                    + ("opacity:.45;" if int(d) > p.day else "") + '"></span>'
                    for d, s in sorted(e["schedule"].items(), key=lambda kv: int(kv[0])))
                st.markdown(f'<div>{cells}</div><div class="bp-lead" style="font-size:.85rem;margin:0">Solid = ads on, '
                            f'striped = ads off, faded = days still to come</div>', unsafe_allow_html=True)
            if e["status"] == "proposed":
                a, b_, _ = st.columns([1, 1, 5])
                if a.button("Approve", key=f"xa_{e['id']}", type="primary"):
                    XP.approve(e["id"], p.day + 1, CFG, OUT)
                    st.rerun()
                if b_.button("Cancel", key=f"xc_{e['id']}"):
                    XP.cancel(e["id"], OUT)
                    st.rerun()
            elif e["status"] == "running" and st.button("Stop this test", key=f"xc_{e['id']}"):
                XP.cancel(e["id"], OUT)
                st.rerun()
            r = e.get("result") or {}
            if r.get("status") == "ok":
                st.markdown(f"With ads on it sold **{r['units_per_day_on']:.1f}** a day; with ads off "
                            f"**{r['units_per_day_off']:.1f}** a day. The ads were credited with "
                            f"{r['attributed_per_day_on']:.1f} sales a day, but only about "
                            f"**{max(r['incrementality'], 0):.0%}** of them were really extra.")
                if detailed:
                    st.caption(f"90% confidence interval {r['ci90'][0]:.0%} to {r['ci90'][1]:.0%}; {r.get('verdict', '')}; "
                               f"factor used by the optimizer: {r.get('factor')} (the interval's upper end, to be safe).")
            elif r:
                st.caption(f"Not enough days yet: {r.get('days_on', 0)} with ads on and {r.get('days_off', 0)} with "
                           f"ads off so far.")
    factors = XP.load_factors(OUT)
    if factors:
        st.markdown("#### Measured results in use")
        st.dataframe(pd.DataFrame(factors).T, width="stretch")

# ======================= 7. Real ads check =======================
with tabs[6]:
    st.markdown("### Real ads check")
    T.lead("BidPilot's waste rule applied to 1,143 real Facebook ads from another company. Read-only: nothing is "
           "changed in any account.")
    if not RA.available():
        st.info("The real ad file is missing (data/real/KAG_conversion_data.csv).")
    else:
        with st.expander("Adjust the waste rule", expanded=detailed):
            c = st.columns(3)
            min_spend = c[0].number_input("Minimum spend per ad", 0.0, 1000.0, 20.0, 5.0)
            min_clicks = c[1].number_input("Minimum clicks per ad", 0, 1000, 20, 5)
            conf = c[2].slider("How sure before calling it waste", 0.80, 0.99, 0.95, 0.01,
                               help="0.95 means: zero purchases would be bad luck less than 5% of the time.")
        a = RA.audit(RA.load_real_ads(), min_spend, int(min_clicks), conf)
        m = st.columns(5)
        m[0].metric("Real ads", a["ads"])
        m[1].metric("Total spend", f"{a['total_spend']:,.2f}")
        m[2].metric("Purchases", a["total_purchases"])
        m[3].metric("Clearly wasted", f"{a['waste_spend']:,.2f}", f"{a['waste_share_pct']}% of spend",
                    delta_color="off", delta_arrow="off")
        m[4].metric("Too early to judge", f"{a['low_data_spend']:,.2f}", f"{a['low_data_ads']} ads", delta_color="off",
                    delta_arrow="off",
                    help="Ads with no purchases but too few clicks to be sure. BidPilot leaves these alone.")
        T.stitch(f"<b>In plain words:</b> {a['waste_share_pct']}% of this company's ad money went to "
                 f"{len(a['waste_ads'])} ads that should have sold something by now and did not. Another "
                 f"{a['low_data_spend']:,.2f} went to ads with no sales yet, but too little data to be sure, so they "
                 f"are not flagged. Currency is as in the source file.")
        bc = a["by_campaign"].copy()
        bc["campaign_id"] = "Campaign " + bc.campaign_id.astype(str)
        fig = px.bar(bc, x="campaign_id", y="cost_per_purchase", text="cost_per_purchase",
                     labels={"campaign_id": "", "cost_per_purchase": "Cost per purchase"},
                     title="What one purchase cost, by campaign")
        fig.update_traces(marker_color=T.INDIGO, texttemplate="%{text:,.2f}", textposition="outside")
        st.plotly_chart(T.style_fig(fig, 340), width="stretch")
        st.markdown(f"#### Ads to pause ({len(a['waste_ads'])})")
        cols = ["ad_id", "campaign_id", "age", "gender", "clicks", "spend", "purchases"] + (
            ["interest", "impressions", "enquiries", "p_zero"] if detailed else [])
        st.dataframe(a["waste_ads"][[c for c in cols if c in a["waste_ads"]]].rename(columns={
            "ad_id": "Ad", "campaign_id": "Campaign", "age": "Age", "gender": "Gender", "clicks": "Clicks",
            "spend": "Spend", "purchases": "Purchases", "p_zero": "Chance it is bad luck"}),
            hide_index=True, width="stretch")
        if detailed:
            st.caption(f"Rule: {a['rule']}. Source: {a['source']}.")

# ======================= 8. Speed =======================
with tabs[7]:
    st.markdown("### Speed")
    T.lead("How long BidPilot's AI work takes: measured on this AMD laptop, and estimated on an AMD Instinct MI300X "
           "server running vLLM.")
    res = read_json(ROOT / "bench" / "results.json")
    if not res:
        st.info("No speed test yet. Run `python -m cli bench` with the AI server running.")
    else:
        by = {(r["scenario"], r["variant"]): r for r in res["results"]}
        nv, bpv = by[("bidding", "naive")], by[("bidding", "bidpilot")]
        st.markdown("#### Measured on this laptop")
        st.caption(res.get("label") or f"{res.get('model')} on {res.get('gpu')}")
        m = st.columns(3)
        m[0].metric("Daily bidding, one AI call per ad", f"{nv['wall_s'] / 60:.1f} min",
                    help=nv.get("note", "All ads, one AI call each, one after another."))
        m[1].metric("Daily bidding with BidPilot", f"{bpv['wall_s'] / 60:.1f} min")
        m[2].metric("Faster by", f"{res['speedup_bidding']:.1f}x")
        st.caption("The laptop's built-in graphics chip works on about one AI request at a time, so running calls "
                   "side by side barely helps here. " + (nv.get("note", "") + "." if detailed else ""))

        from bench.estimate import estimate
        est = estimate(res)
        lo, hi = est["range"]["low"], est["range"]["high"]
        st.markdown("#### Estimated on AMD Instinct MI300X with vLLM")
        st.markdown(T.tag("Estimate, not a measurement", "you") + T.tag("Qwen2.5-32B, the stronger model", "info"),
                    unsafe_allow_html=True)
        m = st.columns(4)
        m[0].metric("Daily bidding, one AI call per ad", f"{hi['bidding_naive_s']:.0f}-{lo['bidding_naive_s']:.0f} s")
        m[1].metric("Daily bidding with BidPilot", f"{hi['bidding_bidpilot_s']:.0f}-{lo['bidding_bidpilot_s']:.0f} s")
        m[2].metric("Faster by", f"{min(lo['speedup_bidding'], hi['speedup_bidding']):.0f}-"
                                 f"{max(lo['speedup_bidding'], hi['speedup_bidding']):.0f}x")
        m[3].metric("Weekly analysis, side by side", f"{hi['consultant_parallel_s']:.1f}-{lo['consultant_parallel_s']:.1f} s",
                    help=f"One by one it would take {hi['consultant_sequential_s']:.1f}-"
                         f"{lo['consultant_sequential_s']:.1f} s.")
        names_ = ["One AI call per ad", "BidPilot"]
        mids = [(lo["bidding_naive_s"] + hi["bidding_naive_s"]) / 2, (lo["bidding_bidpilot_s"] + hi["bidding_bidpilot_s"]) / 2]
        errs = [(lo["bidding_naive_s"] - hi["bidding_naive_s"]) / 2, (lo["bidding_bidpilot_s"] - hi["bidding_bidpilot_s"]) / 2]
        fig = go.Figure(go.Bar(x=names_, y=mids, error_y=dict(type="data", array=errs, color=T.INK, thickness=1.5),
                               marker_color=["#9AA0B4", T.INDIGO], text=[f"{v:.0f} s" for v in mids],
                               textposition="outside"))
        fig.update_layout(title="Daily bidding for all ads on one MI300X (estimated range)", yaxis_title="Seconds")
        st.plotly_chart(T.style_fig(fig, 340), width="stretch")
        T.stitch("<b>Why the big GPU is so much faster:</b> writing each word of an answer means reading the whole "
                 "AI model from memory. The MI300X reads memory about 100 times faster than this laptop (5.3 TB/s), "
                 "and vLLM lets dozens of requests share each read, so BidPilot's explanations are written together "
                 "instead of one by one. The maths that sets the bids already takes about 2 seconds on any computer.")
        if detailed:
            st.caption(f"Method: {est['method']}. Single request about {lo['single_stream_tokens_per_s']:.0f}-"
                       f"{hi['single_stream_tokens_per_s']:.0f} tokens/s; 32 requests together about "
                       f"{lo['batch32_total_tokens_per_s']:,.0f}-{hi['batch32_total_tokens_per_s']:,.0f} tokens/s; "
                       f"prompt reading {lo['prefill_tokens_per_s']:,.0f}-{hi['prefill_tokens_per_s']:,.0f} tokens/s. "
                       f"Confirm with: {est['how_to_confirm']}.")
            st.dataframe(pd.DataFrame(est["range"]).T, width="stretch")
        if (OUT / "speed_chart.png").exists() and detailed:
            st.image(str(OUT / "speed_chart.png"), caption="Measured on the laptop")
