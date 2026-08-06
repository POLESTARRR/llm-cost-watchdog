FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY dashboard/ ./dashboard/

EXPOSE 8000

# Render (and most PaaS hosts) inject their own $PORT and expect the app to
# bind to it; hardcoding 8000 works locally but silently fails a real deploy.
# Shell form (not exec-form array CMD) so ${PORT} actually expands.
CMD uvicorn dashboard.app:app --host 0.0.0.0 --port ${PORT:-8000}
