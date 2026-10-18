# Image for sandbox.backend: docker
#   docker build -t bidpilot-sandbox:latest sandbox/
# Runs with --network none --read-only --memory <mb>m; data is mounted read-only at /work.
FROM python:3.12-slim
RUN pip install --no-cache-dir pandas==2.2.3 numpy==2.1.3 pyarrow==18.1.0
USER nobody
