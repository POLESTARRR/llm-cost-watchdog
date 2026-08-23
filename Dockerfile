FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Separate from requirements.txt on purpose, see the comment there. On
# linux/amd64 (Render's platform) libsql ships a prebuilt wheel and this is a
# fast, no-compile install.
#
# On arm64 there is no wheel, so pip falls back to building from source and
# starts downloading a Rust toolchain. That turned `docker build` on an Apple
# Silicon Mac into a failure, while the README advertised `docker compose up`
# as a way to run this. Turso is optional (src/turso_backend.py is only used
# when TURSO_DATABASE_URL is set), so a missing wheel must not fail the build:
# the image still runs, backed by local SQLite.
RUN pip install --no-cache-dir libsql==0.1.11 \
    || echo "libsql unavailable for this architecture; Turso disabled, SQLite still works"

COPY src/ ./src/
COPY dashboard/ ./dashboard/
# scripts/ ships too: gateway_demo.py and proof.py are how a deployed instance
# is verified from the outside, and they are useless sitting only on a laptop.
COPY scripts/ ./scripts/

EXPOSE 8000

# Render (and most PaaS hosts) inject their own $PORT and expect the app to
# bind to it; hardcoding 8000 works locally but silently fails a real deploy.
# Shell form (not exec-form array CMD) so ${PORT} actually expands.
CMD uvicorn dashboard.app:app --host 0.0.0.0 --port ${PORT:-8000}
