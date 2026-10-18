"""BidPilot look and feel: a block-print textile identity for a shop that sells home textiles.

Palette (taken from indigo / madder / marigold block-printed cloth):
  indigo #1E2A78 main, deep indigo #141C52, marigold #F4A51C = needs your attention, madder #B3263E = loss / pause,
  leaf #1F8A5B = profit / healthy, loom white #F6F7FB background, ink #1A1D33 text, thread #5B6178 secondary text.
Patterns: a block-print lattice in the top banner (the one loud element), a faint woven gingham in the sidebar,
and stitched (dashed) borders on the "How this works" boxes. Everything else stays quiet.
"""
from __future__ import annotations

import html

import streamlit as st

INDIGO, DEEP, MARIGOLD, MADDER, LEAF = "#1E2A78", "#141C52", "#F4A51C", "#B3263E", "#1F8A5B"
LOOM, INK, THREAD = "#F6F7FB", "#1A1D33", "#5B6178"
STRATEGY_NAMES = {"fixed": "No changes (fixed bids)", "rules": "Simple rules", "agent": "BidPilot",
                  "agent_nollm": "BidPilot"}
STRATEGY_COLORS = {"No changes (fixed bids)": "#9AA0B4", "Simple rules": MARIGOLD, "BidPilot": INDIGO}

GLOSSARY = {
    "Bid": "The most you agree to pay when someone clicks your ad.",
    "ROAS": "Return on ad spend: ad sales for every Rs 1 spent on ads. 5.0 means Rs 5 of sales per Rs 1.",
    "ACOS": "Ad cost of sales: ad spend as a share of ad sales. Lower is cheaper. ACOS = 1 / ROAS.",
    "Conversion rate": "The share of clicks that turn into a sale.",
    "Contribution profit": "What is left after product cost, marketplace fees, returns and ad spend.",
    "Days of stock": "How many days the current stock lasts at the recent selling speed.",
    "Approval": "Big changes (more than 10%, pauses, top spenders, cuts on best sellers) wait for you.",
    "Sources [D#] and [W#]": "[D#] is a result computed from your data (click to see the table and code). "
                             "[W#] is a fact from a web page (click to see the link and the exact quote).",
    "Incrementality": "The share of ad sales that would NOT have happened without the ad.",
    "Confidence": "How much data supports a change: high, medium or low.",
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Public+Sans:wght@400;500;600;700&display=swap');
:root {{
  --indigo:{INDIGO}; --deep:{DEEP}; --marigold:{MARIGOLD}; --madder:{MADDER}; --leaf:{LEAF};
  --loom:{LOOM}; --ink:{INK}; --thread:{THREAD}; --line:#DDE1EE;
}}
html, body, .stApp, .stMarkdown, button, input, textarea, select, [data-testid="stMetricValue"] {{
  font-family: 'Public Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}}
h1, h2, h3, h4, .bp-title {{ font-family: 'Bricolage Grotesque', 'Public Sans', system-ui, sans-serif;
  letter-spacing: -0.01em; color: var(--ink); }}
.stApp {{ background: var(--loom); color: var(--ink); }}
[data-testid="stMetricValue"], .bp-num {{ font-variant-numeric: tabular-nums; }}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {{
  font-family:'Public Sans', system-ui, -apple-system, 'Segoe UI', sans-serif !important;
  font-size:1.7rem !important; font-weight:600 !important; color:var(--ink); }}
[data-testid="stMetricLabel"] * {{ color:var(--thread); }}

/* sidebar: faint woven gingham */
[data-testid="stSidebar"] {{
  background-color: #EEF0F8;
  background-image:
    repeating-linear-gradient(0deg, rgba(30,42,120,.045) 0 6px, transparent 6px 14px),
    repeating-linear-gradient(90deg, rgba(30,42,120,.045) 0 6px, transparent 6px 14px);
}}

/* banner with block-print lattice */
.bp-hero {{ display:grid; grid-template-columns:minmax(0,1fr) 240px; border-radius:18px; overflow:hidden;
  background:var(--indigo); margin:4px 0 14px 0; }}
.bp-hero-text {{ padding:20px 24px 18px; color:#fff; }}
.bp-hero-text .bp-title {{ color:#fff; font-size:2rem; font-weight:700; line-height:1.1; margin:0 0 6px; }}
.bp-hero-text p {{ margin:0 0 12px; color:#DCE0F5; max-width:62ch; line-height:1.45; }}
.bp-pattern {{
  background-color: var(--deep);
  background-image:
    radial-gradient(circle, var(--marigold) 0 3px, transparent 3.6px),
    radial-gradient(circle, var(--madder) 0 5px, transparent 5.6px),
    repeating-linear-gradient(45deg, rgba(255,255,255,.13) 0 2px, transparent 2px 19.8px),
    repeating-linear-gradient(-45deg, rgba(255,255,255,.13) 0 2px, transparent 2px 19.8px);
  background-size: 28px 28px, 28px 28px, auto, auto;
  background-position: 0 0, 14px 14px, 0 0, 0 0;
}}
.bp-chip {{ display:inline-flex; align-items:center; gap:7px; padding:5px 11px; margin:0 6px 6px 0;
  border-radius:999px; background:rgba(255,255,255,.13); color:#fff; font-size:.88rem; }}
.bp-dot {{ width:9px; height:9px; border-radius:50%; display:inline-block; }}
.bp-dot.ok {{ background:var(--leaf); box-shadow:0 0 0 2px rgba(255,255,255,.55); }}
.bp-dot.warn {{ background:var(--marigold); }}
.bp-dot.bad {{ background:var(--madder); box-shadow:0 0 0 2px rgba(255,255,255,.55); }}
.bp-dot.off {{ background:#9AA0B4; }}

/* next-step hint: marigold = something for you to do */
.bp-next {{ border:1px solid var(--marigold); background:#FFF5E0; border-radius:12px; padding:10px 14px;
  margin:0 0 12px; color:var(--ink); }}
.bp-next b {{ color:#8A5300; }}

/* stitched explainer boxes */
.bp-stitch {{ border:2px dashed rgba(30,42,120,.32); border-radius:14px; padding:12px 16px; margin:4px 0 12px;
  background:#FBFBFE; color:var(--ink); line-height:1.5; }}
.bp-lead {{ color:var(--thread); font-size:1.02rem; margin:2px 0 12px; max-width:75ch; line-height:1.5; }}

/* bid change line */
.bp-move {{ font-size:1.05rem; }}
.bp-move .old {{ color:var(--thread); text-decoration:line-through; }}
.bp-move .new {{ font-weight:700; }}
.bp-tag {{ display:inline-block; padding:2px 9px; border-radius:999px; font-size:.82rem; font-weight:600;
  margin-right:6px; border:1px solid transparent; }}
.bp-tag.up {{ background:#E3F4EC; color:#14603F; border-color:#BFE3D0; }}
.bp-tag.down {{ background:#FBE7EA; color:#8E1C30; border-color:#F2C6CE; }}
.bp-tag.pause {{ background:#ECEDF3; color:#3B4060; border-color:#D5D8E5; }}
.bp-tag.you {{ background:#FFF0CC; color:#7A4A00; border-color:#F7D58A; }}
.bp-tag.info {{ background:#E8EBF8; color:var(--indigo); border-color:#CDD3EE; }}

/* calm containment for metrics and bordered groups */
[data-testid="stMetric"] {{ background:#fff; border:1px solid var(--line); border-radius:12px; padding:10px 14px; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius:14px; }}
.stTabs [data-baseweb="tab-list"] {{ gap:4px; flex-wrap:wrap; }}
.stTabs [data-baseweb="tab"] {{ font-weight:600; }}
.stTabs [aria-selected="true"] {{ color:var(--indigo); }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color:var(--marigold); height:3px; }}

/* keyboard focus must always be visible */
button:focus-visible, [role="tab"]:focus-visible, input:focus-visible, a:focus-visible,
[data-baseweb="select"]:focus-within {{ outline:3px solid var(--marigold) !important; outline-offset:2px; }}

@media (max-width: 760px) {{
  .bp-hero {{ grid-template-columns:1fr; }}
  .bp-pattern {{ height:22px; order:-1; }}
  .bp-hero-text .bp-title {{ font-size:1.6rem; }}
}}
@media (prefers-reduced-motion: reduce) {{ * {{ animation:none !important; transition:none !important; }} }}
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(s) -> str:
    return html.escape(str(s))


def chip(text: str, state: str = "ok") -> str:
    return f'<span class="bp-chip"><span class="bp-dot {state}"></span>{esc(text)}</span>'


def hero(title: str, subtitle: str, chips: list[str]) -> None:
    st.markdown(f'<div class="bp-hero"><div class="bp-hero-text"><div class="bp-title">{esc(title)}</div>'
                f'<p>{esc(subtitle)}</p>{"".join(chips)}</div><div class="bp-pattern" aria-hidden="true"></div>'
                f'</div>', unsafe_allow_html=True)


def next_step(text_html: str) -> None:
    st.markdown(f'<div class="bp-next">{text_html}</div>', unsafe_allow_html=True)


def lead(text: str) -> None:
    """One plain sentence under a page title: what this page is for."""
    st.markdown(f'<div class="bp-lead">{esc(text)}</div>', unsafe_allow_html=True)


def stitch(text_html: str) -> None:
    st.markdown(f'<div class="bp-stitch">{text_html}</div>', unsafe_allow_html=True)


def tag(text: str, kind: str = "info") -> str:
    return f'<span class="bp-tag {kind}">{esc(text)}</span>'


def rs(x, decimals: int = 0) -> str:
    return f"Rs {x:,.{decimals}f}"


def bid_move(action: str, old: float, new: float, change_pct: float) -> str:
    if action == "pause":
        return f'<span class="bp-move">{tag("Pause ads", "pause")} bid stays Rs {old:,.2f} while paused</span>'
    if action == "resume":
        return f'<span class="bp-move">{tag("Resume ads", "up")} at Rs {new:,.2f}</span>'
    kind = "up" if new > old else "down"
    word = "Raise" if new > old else "Lower"
    sign = "+" if change_pct > 0 else "-"
    return (f'<span class="bp-move">{tag(f"{word} bid {sign}{abs(change_pct) * 100:.0f}%", kind)}'
            f'<span class="old">Rs {old:,.2f}</span> to <span class="new">Rs {new:,.2f}</span></span>')


def style_fig(fig, height: int = 380):
    fig.update_layout(font_family="Public Sans, system-ui, sans-serif", font_color=INK, height=height,
                      plot_bgcolor="#FFFFFF", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
                      title_font_family="Bricolage Grotesque, Public Sans, sans-serif")
    fig.update_xaxes(gridcolor="#EEF0F6", zeroline=False)
    fig.update_yaxes(gridcolor="#EEF0F6", zeroline=False)
    return fig
