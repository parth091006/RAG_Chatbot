FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml ./
COPY . .

RUN pip install --upgrade pip && pip install .

EXPOSE 8000

CMD ["uvicorn", "src.rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
