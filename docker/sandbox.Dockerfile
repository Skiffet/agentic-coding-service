# Minimal image used to run model-generated test commands in isolation.
# Build once: docker build -f docker/sandbox.Dockerfile -t agentic-sandbox:latest .
FROM python:3.12-slim

RUN pip install --no-cache-dir pytest

WORKDIR /workspace
