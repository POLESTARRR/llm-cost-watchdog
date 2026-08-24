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
  launchctl load "$PLIST" && echo "installed: $LABEL (daily 20:00, and on login)" \
    && echo "log: $LOG"
}

case "${1:-}" in
  --install)   install_job; exit $? ;;
  --uninstall) launchctl unload "$PLIST" 2>/dev/null; rm -f "$PLIST"; echo "removed $LABEL"; exit 0 ;;
  --status)    launchctl list | grep -q "$LABEL" && echo "job is loaded" || echo "job is NOT loaded"
               [ -f "$LOG" ] && echo "--- last run ---" && tail -n 15 "$LOG"; exit 0 ;;
esac

echo "=== $(date '+%Y-%m-%d %H:%M:%S') syncing to $SITE ==="

if [ ! -x "$ROOT/venv/bin/python" ]; then
  echo "no venv at $ROOT/venv. Rebuild it: python3 -m venv venv && venv/bin/pip install -r requirements.txt"
  exit 1
fi

"$ROOT/venv/bin/python" "$ROOT/scripts/import_all_projects.py" --remote-url "$SITE"
status=$?

if [ $status -ne 0 ]; then
  # A failed sync must not look like a successful one. The most common cause by
  # far is WATCHDOG_IMPORT_KEY here not matching the value set on the host.
  echo "sync FAILED (exit $status). If this is a 401, the local WATCHDOG_IMPORT_KEY"
  echo "does not match the one in the deployment's environment."
  exit $status
fi

echo "sync complete"
