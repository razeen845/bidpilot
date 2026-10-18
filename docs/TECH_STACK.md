# BidPilot - tools and technologies

## AI / LLM
- **Qwen2.5 Instruct** (open-weight LLM by Alibaba Cloud): 3B (local, AMD Ryzen laptop) and 32B (AMD Instinct MI300X)
- **llama.cpp (Vulkan backend)** - runs the model on the AMD Radeon GPU of an AMD Ryzen laptop
- **vLLM on ROCm** - serves the model on AMD Instinct GPUs (MI300X) in the cloud
- **OpenAI-compatible API** (via the `openai` Python client) - one interface for both servers
- JSON-schema structured outputs, async concurrent calls (asyncio + semaphore), retry and validation (Pydantic)

## AMD hardware and software
- AMD Ryzen 5 7430U with AMD Radeon graphics (Vulkan) - local runs
- AMD Instinct MI300X + ROCm + vLLM - cloud deployment target (Docker Compose profile included)

## Decision engine (no AI in the numbers)
- **Python 3.12**
- **NumPy, SciPy** - response-curve fitting (curve_fit), Welch t-tests and confidence intervals
- **pandas, PyArrow / Parquet** - data handling and storage
- Custom bid optimizer, guardrails, stock guard, rollback rules, switchback incrementality tests

## Safety and trust
- **AST-based code policy** + isolated subprocess sandbox (Docker `--network none --read-only` option) for AI-written analysis code
- Number check and citation check on every AI sentence; quote verification for web facts
- **httpx + trafilatura** - fetching and extracting web page text for research
- Pydantic-validated configuration

## App and tooling
- **Streamlit** dashboard, **Plotly** and **Matplotlib** charts
- Command line interface (argparse)
- **Docker / Docker Compose**, **GitHub Actions** CI
- **pytest** (131 tests, 93% coverage, 25/25 planted bugs caught), **ruff** linting

## Data
- **UCI Online Retail** (real UK online retailer, CC BY 4.0) - real products, prices and daily demand
- **Facebook ad campaign data** ("Sales Conversion Optimization", Kaggle) - 1,143 real ads for the waste audit
- Simulated practice ad account (no real ad platform APIs, no real money)
