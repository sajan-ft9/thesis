FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.12.0 torchvision==0.27.0 \
    && python -m pip install -r requirements-docker.txt \
    && python -m pip install --no-deps -e .

COPY scripts ./scripts
COPY tests ./tests
COPY configs ./configs
COPY data/README.md ./data/README.md
COPY reports ./reports
COPY paper_assets ./paper_assets
COPY Makefile ./Makefile

# Real datasets, models, results, and reports are mounted at runtime. This
# image intentionally does not bake licensed medical images into the image.
VOLUME ["/app/data", "/app/models", "/app/results"]

CMD ["python", "-m", "pytest"]
