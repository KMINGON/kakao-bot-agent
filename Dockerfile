FROM ghcr.io/astral-sh/uv:0.9.21@sha256:15f68a476b768083505fe1dbfcc998344d0135f0ca1b8465c4760b323904f05a AS uv
FROM python:3.12-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337

COPY --from=uv /uv /uvx /usr/local/bin/
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    LANGSMITH_TRACING=false \
    LANGCHAIN_TRACING_V2=false
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY domain_packs ./domain_packs
COPY ops ./ops
RUN uv sync --frozen --no-dev --no-editable \
    && groupadd --gid 1000 app \
    && useradd --uid 1000 --gid 1000 --create-home app
USER 1000:1000
EXPOSE 8000
CMD ["uvicorn", "kakao_bot.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]

