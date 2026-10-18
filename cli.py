"""BidPilot command line.  python -m cli <command> [options]

Account (practice ad account, persisted in out/):
  generate            create a fresh store (fake catalogue, hidden market, 30 days of history)
  reset               put the live account back to day 0
  status              account day, LLM status, pending approvals / actions / experiments
  day                 plan the next day (--plan), apply decisions (--apply), or plan+apply automatically (--auto)
  decide              record approve/reject decisions for the planned day
  race                14-day comparison: fixed vs rules vs agent (on fresh copies; never touches the live account)
Consultant and analytics:
  consult             run the weekly consultant now (needs the LLM)
  recs                accept / reject recommendations of the latest consultant report
  ask                 ask the data (a preset number or free text; free text needs the LLM)
Incrementality:
  experiment          list | propose PID | approve XID | cancel XID   (switchback holdout tests)
Real data:
  generate --source uci_online_retail   store built from a real retailer's products, prices and daily demand
  audit-ads [CSV]     read-only waste audit of a real ad export (default: 1,143 real Facebook ads)
Ops:
  validate-config     check config.yaml and print the effective platform profile
  check-llm           one chat + one JSON-schema request against the configured endpoint
  bench               speed test (on the AMD box); --dry-run uses a fake server
  dashboard           start the Streamlit dashboard
  demo                prepare demo data (race, preset analyses, day-7 report if the LLM is up) and start the dashboard
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from settings import ConfigError, load_config, out_dir, read_json, write_json  # noqa: E402


def _platform(cfg):
    from sim.platform import AdPlatform
    o = out_dir()
    if not (o / "products.csv").exists():
        from data.generate import generate
        generate(cfg, o)
    return AdPlatform(cfg, o, persist=True)


def _llm(cfg, args):
    if getattr(args, "no_llm", False):
        return None
    from agent.llm import make_llm
    return make_llm(cfg, out_dir())


def _print(obj):
    print(json.dumps(obj, indent=1, default=str))


def cmd_generate(cfg, args):
    from data.generate import generate
    if args.source:
        cfg = {**cfg, "store": {**cfg["store"], "source": args.source}}
    paths = generate(cfg, out_dir())
    print(f"store generated in {paths['products'].parent}")


def cmd_reset(cfg, args):
    from sim.platform import reset
    reset(cfg, out_dir())
    o = out_dir()
    for f in list(o.glob("day_*_*.json")) + [o / "pending_consultant_actions.json"]:
        f.unlink(missing_ok=True)
    print("account reset to day 0")


def cmd_status(cfg, args):
    from agent.experiments import load_experiments
    from agent.orchestrator import load_pending_actions
    p = _platform(cfg)
    o = out_dir()
    plan = read_json(o / f"day_{p.day + 1}_plan.json")
    _print({
        "account_day": p.day,
        "platform_profile": cfg["platform_profile"]["label"],
        "llm": "connected" if _llm(cfg, args) else "off (--no-llm behaviour)",
        "planned_next_day": bool(plan),
        "awaiting_approval": [c["entity_id"] for c in (plan or {}).get("changes", []) if c["needs_approval"]],
        "pending_consultant_actions": len(load_pending_actions(o)),
        "experiments": [{k: e[k] for k in ("id", "product_id", "status")} for e in load_experiments(o)],
    })


def cmd_day(cfg, args):
    from agent.llm import run_sync
    from agent.orchestrator import finish_day, load_pending_actions, prepare_day_async, run_agent_day
    p = _platform(cfg)
    llm = _llm(cfg, args)
    o = out_dir()
    if args.auto:
        for _ in range(args.days):
            res = run_agent_day(p, auto_approve=True, pending_consultant_actions=load_pending_actions(o), cfg=cfg,
                                llm=llm, out=o)
            print(f"day {res['day']}: {res['report']}")
        return
    if args.apply:
        plan = read_json(o / f"day_{p.day + 1}_plan.json")
        if not plan:
            sys.exit("no plan for the next day; run `day --plan` first")
        res = finish_day(p, plan, cfg=cfg, llm=llm, out=o)
        print(res["report"])
        if res["applied"]["not_decided"]:
            print(f"{len(res['applied']['not_decided'])} approval rows had no decision and were NOT applied")
        return
    plan, _ = run_sync(prepare_day_async(p, cfg, llm, load_pending_actions(o), o))
    print(f"day {plan['day']}: {len(plan['changes'])} changes, {plan['n_need_approval']} need approval. {plan['day_note']}")
    for c in plan["changes"]:
        if c["needs_approval"]:
            print(f"  [{c['entity_id']}] {c['name']}: {c['action']} {c['old_bid']} -> {c['new_bid']} "
                  f"({c['approval_reason']}) - {c['reason']}")
    print("record decisions with `python -m cli decide --approve ID ... --reject ID ...`, then `day --apply`")


def cmd_decide(cfg, args):
    p = _platform(cfg)
    o = out_dir()
    path = o / f"day_{p.day + 1}_decisions.json"
    dec = read_json(path, default={})
    dec.update({eid: "approve" for eid in args.approve})
    dec.update({eid: "reject" for eid in args.reject})
    if args.approve_all or args.reject_all:
        plan = read_json(o / f"day_{p.day + 1}_plan.json", default={"changes": []})
        for c in plan["changes"]:
            if c["needs_approval"] and c["entity_id"] not in dec:
                dec[c["entity_id"]] = "approve" if args.approve_all else "reject"
    write_json(path, dec)
    print(f"{len(dec)} decisions saved for day {p.day + 1}")


def cmd_race(cfg, args):
    from sim.runner import compare, summarize
    llm = _llm(cfg, args) if args.llm else None
    strategies = ["fixed", "rules", "agent" if llm else "agent_nollm"]
    race = compare(strategies, cfg=cfg, out=out_dir(), llm=llm)
    import pandas as pd
    pd.set_option("display.width", 200)
    print(pd.DataFrame([summarize(race[race.strategy == s]) for s in strategies]).to_string(index=False))


def cmd_consult(cfg, args):
    from agent.consultant import run_consultant
    llm = _llm(cfg, args)
    if llm is None:
        sys.exit("the consultant needs the LLM; set LLM_BASE_URL (see README)")
    p = _platform(cfg)
    rep = run_consultant(p, p.day, cfg, llm, out=out_dir())
    print((out_dir() / "consult" / f"day_{rep['day']}.md").read_text(encoding="utf-8"))


def cmd_recs(cfg, args):
    from agent.consultant import decide_recommendations, latest_report
    rep = latest_report(out_dir())
    if not rep:
        sys.exit("no consultant report yet")
    dec = {r: "accept" for r in args.accept} | {r: "reject" for r in args.reject}
    if dec:
        res = decide_recommendations(rep["day"], dec, out_dir())
        print(f"pending bid actions: {len(res['pending_actions'])}, owner to-dos: {len(res['todos'])}")
    for r in sorted(rep["recommendations"], key=lambda r: r.get("rank", 0)):
        print(f"#{r.get('rank')} {r['id']} [{r['status']}, {r.get('confidence')}] {r['title']} "
              f"-> {r.get('decision') or 'undecided'}")


def cmd_ask(cfg, args):
    from agent.analyst import PRESET_QUESTIONS, ask
    q = args.question
    if q.isdigit():
        q = PRESET_QUESTIONS[int(q) - 1]
    p = _platform(cfg)
    src = ask(q, p.public_data(), cfg, llm=_llm(cfg, args), out=out_dir())
    if not src.get("ok"):
        sys.exit(src.get("error"))
    print(f"Q: {q}\nA: {src['answer']}\n\n# [{src['id']}] code by {src['code_author']}\n{src['code']}")
    import pandas as pd
    print(pd.DataFrame(src["result_table"]).to_string(index=False))


def cmd_experiment(cfg, args):
    from agent import experiments as X
    o = out_dir()
    if args.action == "list":
        _print([{k: e.get(k) for k in ("id", "product_id", "status", "start_day", "result")} for e in X.load_experiments(o)])
    elif args.action == "propose":
        _print(X.propose(args.target, cfg, o, reason="proposed from CLI"))
    elif args.action == "approve":
        p = _platform(cfg)
        _print(X.approve(args.target, p.day + 1, cfg, o))
    elif args.action == "cancel":
        X.cancel(args.target, o)
        print(f"{args.target} cancelled")


def cmd_audit_ads(cfg, args):
    from analysis.real_ads import audit, load_real_ads
    a = audit(load_real_ads(args.file), args.min_spend, args.min_clicks, args.confidence)
    print(a["source"])
    print(f"{a['ads']} ads, spend {a['total_spend']:,.2f}, purchases {a['total_purchases']}")
    print(f"confident waste: {a['waste_spend']:,.2f} ({a['waste_share_pct']}% of spend) in {len(a['waste_ads'])} ads")
    print(f"too little data to judge: {a['low_data_spend']:,.2f} in {a['low_data_ads']} ads (not flagged)")
    print(a["by_campaign"].to_string(index=False))
    print(a["waste_ads"][["ad_id", "clicks", "spend", "purchases", "p_zero"]].head(10).to_string(index=False))


def cmd_validate(cfg, args):
    _print({"ok": True, "platform_profile": cfg["platform_profile"], "seed": cfg["seed"],
            "sandbox": cfg["sandbox"]["backend"], "research_mode": cfg["research"]["mode"]})


def cmd_check_llm(cfg, args):
    sys.exit(subprocess.call([sys.executable, str(ROOT / "scripts" / "check_llm.py")]))


def cmd_bench(cfg, args):
    extra = ["--dry-run"] if args.dry_run else []
    sys.exit(subprocess.call([sys.executable, str(ROOT / "bench" / "latency.py"), "--runs", str(args.runs), *extra]))


def cmd_dashboard(cfg, args):
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "dashboard.py"),
                              "--server.port", str(args.port), "--server.headless", "true"]))


def cmd_demo(cfg, args):
    rc = subprocess.call([sys.executable, str(ROOT / "scripts" / "prepare_demo.py")])
    if rc:
        sys.exit(rc)
    cmd_dashboard(cfg, args)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m cli", description="BidPilot: AI bid manager + consultant "
                                 "(practice ad account)", formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n", 1)[1])
    ap.add_argument("--config", help="path to config.yaml (default: ./config.yaml or $BIDPILOT_CONFIG)")
    ap.add_argument("--no-llm", action="store_true", help="never call the LLM (templates, presets only)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    gp = sub.add_parser("generate")
    gp.add_argument("--source", choices=["synthetic", "uci_online_retail"], help="store data (default: config)")
    gp.set_defaults(fn=cmd_generate)
    sub.add_parser("reset").set_defaults(fn=cmd_reset)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    d = sub.add_parser("day")
    g = d.add_mutually_exclusive_group()
    g.add_argument("--plan", action="store_true", help="plan the next day (default)")
    g.add_argument("--apply", action="store_true", help="apply recorded decisions and run the day")
    g.add_argument("--auto", action="store_true", help="plan + auto-approve + run")
    d.add_argument("--days", type=int, default=1, help="with --auto: how many days")
    d.set_defaults(fn=cmd_day)
    dc = sub.add_parser("decide")
    dc.add_argument("--approve", nargs="*", default=[])
    dc.add_argument("--reject", nargs="*", default=[])
    dc.add_argument("--approve-all", action="store_true")
    dc.add_argument("--reject-all", action="store_true")
    dc.set_defaults(fn=cmd_decide)
    r = sub.add_parser("race")
    r.add_argument("--llm", action="store_true", help="use the LLM agent instead of agent_nollm")
    r.set_defaults(fn=cmd_race)
    sub.add_parser("consult").set_defaults(fn=cmd_consult)
    rc = sub.add_parser("recs")
    rc.add_argument("--accept", nargs="*", default=[])
    rc.add_argument("--reject", nargs="*", default=[])
    rc.set_defaults(fn=cmd_recs)
    a = sub.add_parser("ask")
    a.add_argument("question", help="free text, or 1-6 for a preset question")
    a.set_defaults(fn=cmd_ask)
    x = sub.add_parser("experiment")
    x.add_argument("action", choices=["list", "propose", "approve", "cancel"])
    x.add_argument("target", nargs="?", help="product id (propose) or experiment id (approve/cancel)")
    x.set_defaults(fn=cmd_experiment)
    au = sub.add_parser("audit-ads", help="read-only waste audit of a real ad export CSV")
    au.add_argument("file", nargs="?", help="CSV (default: data/real/KAG_conversion_data.csv)")
    au.add_argument("--min-spend", type=float, default=20.0)
    au.add_argument("--min-clicks", type=int, default=20)
    au.add_argument("--confidence", type=float, default=0.95)
    au.set_defaults(fn=cmd_audit_ads)
    sub.add_parser("validate-config").set_defaults(fn=cmd_validate)
    sub.add_parser("check-llm").set_defaults(fn=cmd_check_llm)
    b = sub.add_parser("bench")
    b.add_argument("--dry-run", action="store_true")
    b.add_argument("--runs", type=int, default=3)
    b.set_defaults(fn=cmd_bench)
    for name, fn in (("dashboard", cmd_dashboard), ("demo", cmd_demo)):
        sp = sub.add_parser(name)
        sp.add_argument("--port", type=int, default=8501)
        sp.set_defaults(fn=fn)
    return ap


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):      # Windows consoles default to cp1252
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    args = build_parser().parse_args(argv)
    if args.cmd == "experiment" and args.action != "list" and not args.target:
        build_parser().error("experiment propose/approve/cancel needs a target id")
    try:
        cfg = load_config(args.config)
        if args.no_llm:
            cfg["llm"]["enabled"] = False
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 2
    args.fn(cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
