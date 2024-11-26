FROM python:3.10.15-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    && apt-get clean

RUN curl -sSL https://install.python-poetry.org | python3 -

ENV PATH="/root/.local/bin:$PATH"

COPY pyproject.toml poetry.lock /app/

RUN poetry install --no-dev --no-interaction

COPY . /app/

# ports for chatbot and local qdrant
# EXPOSE 8501
# EXPOSE 6333

# docker-compose overwrites below
CMD ["poetry", "run", "python", "api_data_extractor.py"]
