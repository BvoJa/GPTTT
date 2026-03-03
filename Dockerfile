FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_PROJECT_ENVIRONMENT=/venv

WORKDIR /build

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-cache --no-dev --no-install-project


FROM python:3.11-slim

WORKDIR /code

COPY --from=builder /venv /venv
COPY ./app ./app

ENV PATH="/venv/bin:$PATH"

EXPOSE 8080

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8080"]