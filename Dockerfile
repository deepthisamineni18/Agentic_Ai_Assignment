FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
# torch is only used for a small CPU-trained LSTM in the active-learning
# assignment. Without --extra-index-url, pip grabs the default Linux wheel,
# which drags in the full CUDA/cuDNN/triton toolkit (~2GB+, ~20 min build)
# even though nothing in this container ever touches a GPU. Pointing at
# PyTorch's dedicated CPU wheel index avoids all of that.
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY entrypoint.py .
COPY research_pipeline/ ./research_pipeline/
COPY tests/ ./tests/

# Regenerate the mock corpus at build time so it's always in sync with the generator.
RUN python -m research_pipeline.data.generate_mock_corpus

ENV LOG_LEVEL=INFO
ENV RESEARCH_OUTPUT_DIR=/app/output
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379

ENTRYPOINT ["python", "entrypoint.py"]
CMD ["--topic", "Recent advances in quantum computing hardware"]
