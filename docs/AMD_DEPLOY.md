# Running BidPilot on an AMD GPU server

Two supported setups. Both serve the LLM through vLLM's OpenAI-compatible API on ROCm; BidPilot only needs the
URL.

## A. AMD Instinct GPU in the cloud (MI300X class) - the hackathon setup

**1. Get a GPU machine.** AMD Developer Cloud (or any provider with MI300X / MI250 and ROCm 6.x drivers).
Choose an image with ROCm + Docker preinstalled. Open ports 8000 (vLLM) and 8501 (dashboard), or use an SSH tunnel.

**2. Check the GPU.**
```bash
rocm-smi                 # or: amd-smi list    -> should list the Instinct GPU(s)
docker --version
```

**3. Get the code.**
```bash
git clone <your-repo-url> bidpilot && cd bidpilot
```

**4a. One command (Docker Compose, recommended).** Starts vLLM on ROCm and the dashboard:
```bash
export LLM_MODEL=Qwen/Qwen2.5-32B-Instruct      # fits in one MI300X (192 GB) with room to spare
export HF_TOKEN=<optional huggingface token>
docker compose --profile amd up -d
docker compose logs -f vllm                     # wait for "Application startup complete" (first run downloads ~65 GB)
```
Dashboard: `http://<server-ip>:8501` (or `ssh -L 8501:localhost:8501 user@server` and open http://localhost:8501).

**4b. Without Compose.**
```bash
docker run -d --name vllm --device /dev/kfd --device /dev/dri --group-add video --ipc host --shm-size 16g \
  --security-opt seccomp=unconfined -p 8000:8000 -v ~/.cache/huggingface:/root/.cache/huggingface \
  rocm/vllm:latest vllm serve Qwen/Qwen2.5-32B-Instruct --port 8000 --max-model-len 16384
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
export LLM_BASE_URL=http://localhost:8000/v1
```

**5. Run the checks and produce the demo results** (inside the app container: `docker compose exec app bash`,
or in the venv):
```bash
python -m cli check-llm                          # 1 chat + 1 JSON-schema request: prints latency + JSON mode
python -m cli generate --source uci_online_retail   # optional: the real-retailer store
python scripts/prepare_demo.py                   # race with the full LLM agent + day-7 consultant report
GPU_NAME="AMD Instinct MI300X" python -m cli bench    # speed test -> bench/results.json, out/speed_chart.png
python -m cli dashboard                          # if not already running
```
Write the JSON mode printed by `check-llm` into NOTES.md.

**6. For the online demo page.** Record a 3-4 minute screen capture following MVP.md's demo script, and add
`out/race_chart.png`, `out/speed_chart.png` and a consultant-report screenshot to the submission.

## B. AMD Ryzen laptop / desktop (Radeon iGPU or GPU) - for development

Tested on an AMD Ryzen 5 7430U with Radeon graphics (Windows 11) using llama.cpp's Vulkan backend, which
exposes the same OpenAI-compatible API:
```bash
llama-server -m qwen2.5-3b-instruct-q4_k_m.gguf --port 8080 -ngl 99 -c 16384 -np 4 --jinja
export LLM_BASE_URL=http://localhost:8080/v1  LLM_MODEL=qwen2.5-3b-instruct
python -m cli check-llm
```
A 3B model is much weaker and slower than the 32B model on an MI300X; use it to develop and test, and the
Instinct GPU for the final numbers.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `check-llm` says endpoint not reachable | `curl $LLM_BASE_URL/models`; check the port and firewall |
| JSON errors / mode switches to "prompt" | normal on servers without json_schema support; outputs are still validated |
| vLLM out of memory | lower `--max-model-len`, or use Qwen2.5-14B-Instruct |
| Slow first request | the model is still loading or compiling kernels; wait for the startup log line |
