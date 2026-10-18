"""BidPilot look and feel: formal and minimal, with one quiet textile reference.

Palette: navy #13294B main, deep navy #0B1A33, muted gold #A16207 = needs your attention, oxblood #9F1D2B =
loss / pause, forest #1E6B4E = profit / healthy, paper #F5F6F8 background, ink #0F172A text, slate #475569
secondary text. Pattern: a fine, low-contrast herringbone weave in the banner (the one textile reference) and a
faint pinstripe in the sidebar. Cards are flat white with a hairline border, no fills, no shadows.
Type: IBM Plex Serif for headings, IBM Plex Sans for everything else.
"""
from __future__ import annotations

import html

import streamlit as st

INDIGO, DEEP, MARIGOLD, MADDER, LEAF = "#13294B", "#0B1A33", "#A16207", "#9F1D2B", "#1E6B4E"
LOOM, INK, THREAD = "#F5F6F8", "#0F172A", "#475569"
STRATEGY_NAMES = {"fixed": "No changes (fixed bids)", "rules": "Simple rules", "agent": "BidPilot",
                  "agent_nollm": "BidPilot"}
STRATEGY_COLORS = {"No changes (fixed bids)": "#A3AAB8", "Simple rules": "#B08D57", "BidPilot": INDIGO}

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
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@500;600&display=swap');
:root {{
  --indigo:{INDIGO}; --deep:{DEEP}; --gold:{MARIGOLD}; --madder:{MADDER}; --leaf:{LEAF};
  --paper:{LOOM}; --ink:{INK}; --slate:{THREAD}; --line:#E2E5EA;
}}
html, body, .stApp, .stMarkdown, button, input, textarea, select {{
  font-family: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}}
h1, h2, h3, h4, .bp-title {{ font-family: 'IBM Plex Serif', Georgia, serif; font-weight:600;
  letter-spacing:-0.005em; color: var(--ink); }}
.stApp {{ background: var(--paper); color: var(--ink); }}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {{
  font-family:'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif !important;
  font-size:1.6rem !important; font-weight:500 !important; color:var(--indigo); font-variant-numeric: tabular-nums; }}
[data-testid="stMetricLabel"] * {{ color:var(--slate); font-size:.85rem; }}

/* sidebar: faint pinstripe */
[data-testid="stSidebar"] {{
  background-color: #F0F2F5;
  background-image: repeating-linear-gradient(90deg, rgba(19,41,75,.035) 0 1px, transparent 1px 9px);
  border-right: 1px solid var(--line);
}}

/* banner: navy with a fine herringbone weave */
.bp-hero {{ display:grid; grid-template-columns:minmax(0,1fr) 220px; border-radius:8px; overflow:hidden;
  background:var(--indigo); margin:4px 0 14px 0; }}
.bp-hero-text {{ padding:22px 26px 18px; color:#fff; }}
.bp-hero-text .bp-title {{ color:#fff; font-size:1.9rem; line-height:1.15; margin:0 0 6px; }}
.bp-hero-text p {{ margin:0 0 14px; color:#C9D3E3; max-width:62ch; line-height:1.5; }}
.bp-pattern {{
  background-color: var(--deep);
  background-image:
    repeating-linear-gradient(45deg, rgba(255,255,255,.07) 0 1px, transparent 1px 8px),
    repeating-linear-gradient(-45deg, rgba(255,255,255,.07) 0 1px, transparent 1px 8px);
  background-size: 16px 32px, 16px 32px;
  background-position: 0 0, 8px 16px;
}}
.bp-chip {{ display:inline-flex; align-items:center; gap:7px; padding:4px 10px; margin:0 6px 6px 0;
  border-radius:4px; border:1px solid rgba(255,255,255,.22); color:#E6EBF3; font-size:.85rem; }}
.bp-dot {{ width:7px; height:7px; border-radius:50%; display:inline-block; }}
.bp-dot.ok {{ background:#7FC4A4; }}
.bp-dot.warn {{ background:#E3B562; }}
.bp-dot.bad {{ background:#E08A93; }}
.bp-dot.off {{ background:#A3AAB8; }}

/* next step: quiet, with a gold marker */
.bp-next {{ border:1px solid var(--line); background:#fff; border-radius:8px; padding:10px 14px; margin:0 0 14px;
  color:var(--ink); }}
.bp-next b {{ color:var(--gold); font-weight:600; }}

/* explainer boxes */
.bp-stitch {{ border:1px dashed #C5CBD6; border-radius:8px; padding:12px 16px; margin:4px 0 12px;
  background:#FBFBFC; color:var(--ink); line-height:1.55; }}
.bp-lead {{ color:var(--slate); font-size:1rem; margin:2px 0 14px; max-width:75ch; line-height:1.55; }}

/* bid change line and tags: outlined, no fills */
.bp-move {{ font-size:1rem; }}
.bp-move .old {{ color:var(--slate); text-decoration:line-through; }}
.bp-move .new {{ font-weight:600; color:var(--ink); }}
.bp-tag {{ display:inline-block; padding:1px 8px; border-radius:4px; font-size:.78rem; font-weight:500;
  margin-right:6px; border:1px solid var(--line); color:var(--slate); background:#fff; }}
.bp-tag.up {{ color:var(--leaf); border-color:#BFD8CB; }}
.bp-tag.down {{ color:var(--madder); border-color:#E8C3C8; }}
.bp-tag.pause {{ color:var(--slate); }}
.bp-tag.you {{ color:var(--gold); border-color:#E9D3A6; }}
.bp-tag.info {{ color:var(--indigo); border-color:#CBD3E1; }}

/* cards: flat white, hairline border, no shadow */
[data-testid="stMetric"] {{ background:#fff; border:1px solid var(--line); border-radius:8px; padding:12px 16px; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius:8px !important; border-color:var(--line) !important;
  background:#fff; }}
[data-testid="stExpander"] details {{ border-radius:8px; border-color:var(--line); background:#fff; }}
.stTabs [data-baseweb="tab-list"] {{ gap:2px; flex-wrap:wrap; border-bottom:1px solid var(--line); }}
.stTabs [data-baseweb="tab"] {{ font-weight:500; color:var(--slate); }}
.stTabs [aria-selected="true"] {{ color:var(--indigo); }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color:var(--indigo); height:2px; }}
.stButton button, .stFormSubmitButton button, [data-testid="stPopover"] button {{ border-radius:6px; }}

/* keyboard focus must always be visible */
button:focus-visible, [role="tab"]:focus-visible, input:focus-visible, a:focus-visible,
[data-baseweb="select"]:focus-within {{ outline:2px solid var(--gold) !important; outline-offset:2px; }}

@media (max-width: 760px) {{
  .bp-hero {{ grid-template-columns:1fr; }}
  .bp-pattern {{ height:18px; order:-1; }}
  .bp-hero-text .bp-title {{ font-size:1.55rem; }}
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
    fig.update_layout(font_family="IBM Plex Sans, system-ui, sans-serif", font_color=INK, height=height,
                      plot_bgcolor="#FFFFFF", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
                      title_font_family="IBM Plex Serif, Georgia, serif")
    fig.update_xaxes(gridcolor="#EEF0F3", zeroline=False, linecolor="#E2E5EA")
    fig.update_yaxes(gridcolor="#EEF0F3", zeroline=False)
    return fig
