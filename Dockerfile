# BidPilot dashboard + CLI.
#   docker build -t bidpilot .
#   docker run -p 8501:8501 -e LLM_BASE_URL=http://<amd-box>:8000/v1 -v bidpilot-out:/app/out bidpilot
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BIDPILOT_OUT=/app/out \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN useradd --create-home --uid 1000 bidpilot && mkdir -p /app/out && chown -R bidpilot /app
USER bidpilot

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["python", "-m", "streamlit", "run", "app/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
