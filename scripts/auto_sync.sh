#!/bin/bash
# Push any new Claude Code build cost to the live dashboard. Runs unattended.
#
#   bash scripts/auto_sync.sh              # sync now
#   bash scripts/auto_sync.sh --install    # install the daily launchd job
#   bash scripts/auto_sync.sh --uninstall
#   bash scripts/auto_sync.sh --status
#
# Why this runs on the laptop and not in CI: the transcripts it reads live in
# ~/.claude/projects, on this machine. A GitHub Action has no access to them, so
# a scheduled job in the repo could never do this work. The import endpoint is
# the seam that lets a local reader update a remote site.
#
# Safe to run as often as you like. import_claude_code_usage.py keeps a
# checkpoint per transcript and sends only turns that have not been sent before,
# so a repeat run is a no-op rather than a duplicate import.

set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

SITE="${WATCHDOG_SITE_URL:-https://llmcostwatchdog.onrender.com}"
LABEL="com.dhruvsharma.llmcostgateway.sync"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG="$ROOT/data/auto_sync.log"

install_job() {
  mkdir -p "$HOME/Library/LaunchAgents" "$ROOT/data"
  cat > "$PLIST" <<PLIST_END
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$ROOT/scripts/auto_sync.sh</string>
  </array>
  <!-- Once a day, plus once at load. RunAtLoad matters because a laptop is
       usually asleep at any fixed hour, and StartCalendarInterval does not
       fire for a time that passed while the machine was off. -->
  <key>StartCalendarInterval</key>
  <dict><key>Hour</key><integer>20</integer><key>Minute</key><integer>0</integer></dict>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>$LOG</string>
  <key>StandardErrorPath</key><string>$LOG</string>
</dict>
</plist>
PLIST_END
  launchctl unload "$PLIST" 2>/dev/null
  launchctl load "$PLIST" || { echo "launchctl refused to load $PLIST"; return 1; }
  echo "installed: $LABEL (daily 20:00, and on login)"
  echo "log: $LOG"

  # RunAtLoad fires immediately, so a few seconds is enough to find out whether
  # it can actually run. Reporting "installed" for a job that cannot start is
  # the same failure as a dashboard reporting a number nobody was charged.
  sleep 6
  if grep -qi "operation not permitted" "$LOG" 2>/dev/null; then
    cat <<'BLOCKED'

  BUT IT CANNOT RUN YET, and this is a macOS restriction rather than a bug.

  Background agents are denied access to ~/Desktop, ~/Documents and ~/Downloads
  unless you grant it. This project lives under ~/Desktop, so launchd is refused
  before the script starts. It runs perfectly when you run it yourself, because
  your terminal already has that permission.

  One-time fix:
    System Settings -> Privacy & Security -> Full Disk Access
    add /bin/bash   (click +, press Cmd-Shift-G, type /bin/bash)

  Then: bash scripts/auto_sync.sh --install

  Or avoid the permission entirely by moving this project out of ~/Desktop,
  for example to ~/code/llm-cost-gateway, and reinstalling.

  Until one of those, run it yourself whenever you want the site updated:
    bash scripts/auto_sync.sh
BLOCKED
    return 1
  fi
}

case "${1:-}" in
  --install)   install_job; exit $? ;;
  --uninstall) launchctl unload "$PLIST" 2>/dev/null; rm -f "$PLIST"; echo "removed $LABEL"; exit 0 ;;
  --status)    # Captured first, not piped into grep -q. `grep -q` exits the
               # moment it matches, which SIGPIPEs launchctl, and under
               # `set -o pipefail` that makes the whole pipeline report failure:
               # the job was loaded and this said it was not.
               listing="$(launchctl list 2>/dev/null || true)"
               case "$listing" in
                 *"$LABEL"*) echo "job is loaded" ;;
                 *)          echo "job is NOT loaded" ;;
               esac
               [ -f "$LOG" ] && echo "--- last run ---" && tail -n 15 "$LOG"; exit 0 ;;
esac

echo "=== $(date '+%Y-%m-%d %H:%M:%S') syncing to $SITE ==="

if [ ! -x "$ROOT/venv/bin/python" ]; then
  echo "no venv at $ROOT/venv. Rebuild it: python3 -m venv venv && venv/bin/pip install -r requirements.txt"
  exit 1
fi

# 1. Read every transcript into the local ledger. Discovery is automatic, so a
#    project started today is picked up without anything being configured.
"$ROOT/venv/bin/python" "$ROOT/scripts/import_all_projects.py"
status=$?

# 2. Publish it. This writes to the deployed database directly rather than
#    posting to the site's /import endpoint, which needs WATCHDOG_IMPORT_KEY to
#    match on both ends and does not here, so that route returns 401 and the
#    published site silently stops tracking new work. Going straight to the
#    database is the same operation with one fewer secret to keep aligned, and
#    it does not need the web service to be awake, which on a free tier it
#    usually is not.
if [ $status -eq 0 ]; then
  "$ROOT/venv/bin/python" "$ROOT/scripts/sync_to_turso.py" || status=$?
fi

# Refresh the shipped example of the tool while the transcripts are here to read.
# It is the only thing on the deployed page that shows ccost doing anything, and
# it is a file in the repo rather than something the server can recompute, so it
# goes stale silently unless something rewrites it.
"$ROOT/venv/bin/python" "$ROOT/scripts/ccost.py" --snapshot "$ROOT/data/ccost_snapshot.json" \
  && echo "refreshed data/ccost_snapshot.json (commit it to publish)"

# The findings snapshot is what the site falls back to if its hosted database
# ever lapses. Stale is survivable, absent is not, so it is refreshed here too.
"$ROOT/venv/bin/python" - <<'SNAP'
import datetime as dt, json, pathlib, sys
sys.path.insert(0, ".")
from dashboard.app import findings_endpoint
d = findings_endpoint()
if d.get("headline", {}).get("turns"):
    d.pop("is_snapshot", None)
    d["generated_at"] = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    pathlib.Path("data/findings_snapshot.json").write_text(json.dumps(d, indent=1) + "\n")
    print("refreshed data/findings_snapshot.json")
SNAP

if [ $status -ne 0 ]; then
  # A failed sync must not look like a successful one. The most common cause by
  # far is WATCHDOG_IMPORT_KEY here not matching the value set on the host.
  echo "sync FAILED (exit $status). Check TURSO_DATABASE_URL and TURSO_AUTH_TOKEN"
  echo "in .env.render, and that the local ledger is not empty."
  exit $status
fi

echo "sync complete"
