# Deployment Guide

## Render (Production)

The gateway is configured for deployment on Render with zero-config cold-start optimization.

### One-time setup

1. **Create a Render Blueprint from `render.yaml`:**
   - Go to https://dashboard.render.com/blueprints
   - Click "New Blueprint"
   - Paste the git URL for this repository
   - Render will auto-detect `render.yaml` and deploy both services:
     - **Web service:** the gateway itself
     - **Background worker:** keep-warm ping to prevent sleep

2. **Set environment variables** in the Render dashboard:
   ```
   WATCHDOG_GATEWAY_KEY=<your-secret-key>
   WATCHDOG_GROUP_LADDER=gemini-flash-lite-latest,gemini-flash-latest,gemini-pro-latest
   WATCHDOG_ROUTING_STRATEGY=complexity
   WATCHDOG_GUARD_MODE=block
   TURSO_DATABASE_URL=<your-turso-url>
   TURSO_AUTH_TOKEN=<your-turso-token>
   WATCHDOG_IMPORT_KEY=<optional-import-key>
   ```

### How cold-start optimization works

**Problem:** Render's free tier puts services to sleep after 15 minutes of inactivity. Waking up takes ~13 seconds (Python startup + first request).

**Solution:** The `keep-warm` background worker pings the web service every 14 minutes, keeping it from sleeping.

- Web service startup: database initialization on app launch (0.65s)
- Background worker: lightweight HTTP client, pings `/health` every 840 seconds
- Cost: negligible (a few hundred pings/month)

### Deployment workflow

```bash
# Push to your git repo
git push origin main

# Render auto-deploys both services via webhook
# The keep-warm worker starts immediately and runs continuously
# If you need to stop it, pause it in the Render dashboard
```

### Manual testing

If you're deploying to a different URL, update the `keep-warm` worker:

```bash
# Locally
python scripts/keep_warm.py --base http://localhost:8000 --interval 60

# Production
python scripts/keep_warm.py --base https://your-gateway.onrender.com --interval 840
```

## Local Development

```bash
# Start the gateway
venv/bin/uvicorn dashboard.app:app --reload --port 8000

# In another terminal, test with the proof script
venv/bin/python scripts/proof.py
```

## Docker

```bash
docker build -t llm-cost-gateway .
docker run -p 8000:8000 -e WATCHDOG_GATEWAY_KEY=test llm-cost-gateway
```

## Monitoring

Check the live ledger at:
- **Calls:** `https://your-gateway.onrender.com/calls?source=live`
- **Report:** `https://your-gateway.onrender.com/report?source=live`
- **Router status:** `https://your-gateway.onrender.com/router`
- **Cost levers:** `https://your-gateway.onrender.com/levers?source=live`
