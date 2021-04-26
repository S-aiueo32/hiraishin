# base image
FROM python:3.8-slim AS base

ENV TZ=Asia/Tokyo \
    DEBIAN_FRONTEND=noninteractive \
    # python
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # pip
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    # poetry
    POETRY_VERSION="1.1.5" \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    # path
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv" \
    MY_MODULE_PATH="/app/src"

ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

RUN apt-get update && \
    apt-get install --no-install-recommends -y curl unzip tzdata && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# builder image
FROM base AS builder

RUN curl -sSL "https://raw.githubusercontent.com/python-poetry/poetry/${POETRY_VERSION}/get-poetry.py" | python

WORKDIR $PYSETUP_PATH

COPY poetry.lock pyproject.toml ./
RUN poetry install --no-dev

RUN echo $MY_MODULE_PATH > $(python -c 'import sys; print(sys.path)' | grep -o "[^']*site-packages")/my_module.pth


# dev image
FROM base AS dev

COPY --from=builder $POETRY_HOME $POETRY_HOME
COPY --from=builder $PYSETUP_PATH $PYSETUP_PATH

WORKDIR $PYSETUP_PATH
RUN poetry install

WORKDIR /app
COPY . .