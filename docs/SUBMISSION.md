# BidPilot: an AI bid manager and consultant for online sellers

## The problem
Online sellers pay for ads on every product and every search keyword. Someone has to check every day which ads
make money and which waste it, then raise or lower the bids. Large brands hire bid managers for this; small
sellers can't, so their bids stay wrong for weeks: money leaks into keywords that never sell, ads keep running on
products that are out of stock, and good products stay under-advertised.

## What BidPilot does
BidPilot does the daily work of a bid manager and the weekly work of a consultant, and keeps the seller in control.

**Every day**
- Reads yesterday's results for every product ad and keyword ad.
- Sets new bids with a **math optimizer, not the AI**: it fits a response curve per ad and picks the bid with the
  most expected profit, within safety limits (max ±20% a day).
- Applies practitioner rules: a **stock guard** (slow down ads for products about to sell out, pause when out of
  stock, resume after restock), **waste cleanup** that only pauses when zero sales is statistically unlikely
  (with a conversion-lag window so recent clicks aren't judged too early), **protection for proven winners**,
  **cautious raises** (a bid only goes up if it pays even with a pessimistic conversion rate), and **automatic
  rollback** of moves that clearly hurt profit.
- Uses the **LLM to plan the day and explain every change in plain words**. The maths writes a correct draft;
  the LLM rewrites it for a non-expert; every number it writes is checked against the data, and anything
  invented is replaced by the safe draft.
- Sends big changes (more than 10%, pauses, top spenders, cuts on best earners) to the human for approval;
  small changes apply automatically.

**Every week (or on request)**
- **Ask the data:** the AI writes pandas code, an AST safety policy approves it, it runs in an isolated sandbox
  (no network, no files), and the result becomes a citable source **[D#]** with its code and table.
- **Web research:** market facts are kept only if their exact quote is found on the real page; each becomes a
  citable source **[W#]** with link and date.
- **Consultant report:** a ranked action list (priority, confidence, risk, next step). A citation check removes
  anything not backed by a real data result or verified quote. Accepted bid ideas go into tomorrow's plan
  through the same guardrails; the rest becomes the owner's to-do list.

**Measuring what really works**
- **Incrementality experiments:** ad platforms credit every sale to the ad, but some buyers would have bought
  anyway. BidPilot runs switchback tests (ads on/off on alternating days, owner-approved) and lowers bids on
  products whose ad sales are mostly not extra.
- **Seller objectives:** profit by default, or a "growth push" on chosen products with an ACOS cap.
- **Platform rule profiles** for Amazon (ACOS, 2-day lag, negative exact keywords) and Google (ROAS, 3-day lag,
  negative phrase keywords).

## How AMD is used
- The LLM is an open model (**Qwen2.5 Instruct**) that we run ourselves on AMD hardware through an
  OpenAI-compatible API.
- **Running today on an AMD Ryzen 5 7430U laptop with AMD Radeon graphics**, using llama.cpp with the Vulkan
  backend (Qwen2.5-3B). The full agent ran on it: day plans, explanations for every bid change, analysis answers
  and a complete consultant report.
- **Built for AMD Instinct MI300X + ROCm + vLLM** (Qwen2.5-32B): a Docker Compose profile starts vLLM on ROCm and
  the app with one command.
- Speed design: the optimizer sets all bids in under a second; the LLM writes all explanations **in parallel**
  instead of one call per ad, and the consultant's data questions and web research run **at the same time**.

## Real data
- **Real retailer:** the store can be built from the **UCI Online Retail** dataset (a real UK online shop, CC BY
  4.0): 40 real home-textile products (doormats, hot-water-bottle covers, tea towels, aprons, cushions) with real
  prices and real day-by-day demand. Keywords are generated from the real product names.
- **Real ads:** a read-only audit of **1,143 real Facebook ads** (3 campaigns) applies BidPilot's waste rule:
  **7.5% of all spend** went to ads where zero purchases was statistically not bad luck, while another 8,738.60
  of zero-purchase spend was correctly left alone because the data was too thin. It also found the cheapest
  audience (men 30-34: 25.55 per purchase vs 119.94 for the most expensive group).
- Ad clicks, costs and margins in the store run on a **practice ad account** (simulated), because no public
  dataset has them. No real ad platform or real money is touched.

## Results (14-day simulation, same luck for every strategy)
| Store | Strategy | Total contribution profit | Ad-attributed profit | Ad spend on out-of-stock products |
|---|---|---|---|---|
| Demo store | Fixed bids | Rs 807,137 | Rs 128,820 | Rs 9,640 |
| Demo store | Simple rules | Rs 842,640 | Rs 155,987 | Rs 10,161 |
| Demo store | **BidPilot** | **Rs 932,013** | **Rs 233,661** | **Rs 0** |
| Real-retailer store | Simple rules | Rs 318,852 | Rs 128,759 | Rs 8,226 |
| Real-retailer store | **BidPilot** | **Rs 386,594** | Rs 189,544 | **Rs 0** |

BidPilot sells slightly less than "do nothing" but earns clearly more, because it stops buying sales that lose
money and never advertises products that are out of stock. It wins on every random seed we tested. These are
simulated results: they show the method works, not a guaranteed live-account result.

## Trust and safety
- The LLM never chooses bid amounts and never touches money.
- Every number the AI writes is checked; every claim needs a clickable source; invented numbers and quotes are
  removed (in our real-model run, 2 of 5 consultant recommendations were caught with made-up numbers and
  greyed out as unverified).
- AI-written code runs only in an isolated sandbox after a safety check.
- Works without the AI (`--no-llm`): bidding, safety rules and reviewed analyses keep running.

## Built with
Python, NumPy, SciPy, pandas, PyArrow - Qwen2.5 Instruct, llama.cpp (Vulkan), vLLM on ROCm, OpenAI-compatible
API, Pydantic - Streamlit, Plotly, Matplotlib - Docker, GitHub Actions, pytest (131 tests), ruff - httpx,
trafilatura - AMD Ryzen + Radeon (local), AMD Instinct MI300X (deployment target).

## What's next
Connect a real Google Ads / Amazon Ads account behind the same interface, starting read-only (recommendations
only) before allowing bid changes, and run the 32B model on an AMD Instinct GPU for faster, sharper reasoning.
