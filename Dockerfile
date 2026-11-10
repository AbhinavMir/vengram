FROM python:3.12-slim
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/* \
 && curl -fsSL "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-${TARGETARCH:-amd64}.tgz" | tar -xz -C /usr/local/bin ngrok
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir transformers safetensors anthropic huggingface_hub
WORKDIR /app
COPY brain.py engine.py server.py ui.html logo.png ./
ENV VENGRAM_DATA=/data VENGRAM_HOST=0.0.0.0 PYTHONUNBUFFERED=1
EXPOSE 8790
CMD ["python", "server.py", "--config", "/app/config.json"]
