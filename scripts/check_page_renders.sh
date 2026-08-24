#!/bin/bash
# Load the dashboard in a real browser and fail if any section came out empty.
#
#   bash scripts/check_page_renders.sh                     # against localhost:8077
#   bash scripts/check_page_renders.sh https://your.site   # against a deployment
#
# This exists because `node --check` passed on a page that rendered nothing. It
# validates syntax, and the bug was a ReferenceError: an edit removed a helper
# the render function still called. Syntax was perfect, the identifier was gone,
# the exception was caught by the page's own error handler, and every finding
# section deleted itself. The HTML looked right in every check that read it as
# text, and a visitor saw a blank page.
#
# Nothing short of executing the page catches that, so this executes the page.

set -uo pipefail
URL="${1:-http://localhost:8077}"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OUT="$(mktemp -t rendered).html"

if [ ! -x "$CHROME" ]; then
  echo "SKIP: no Chrome at $CHROME"; exit 0
fi

"$CHROME" --headless --disable-gpu --no-sandbox \
  --virtual-time-budget=25000 --dump-dom "$URL" > "$OUT" 2>/dev/null

python3 - "$OUT" <<'PY'
import re, sys, pathlib

html = pathlib.Path(sys.argv[1]).read_text()

# id -> what it is, for the message when it is empty.
REQUIRED = {
    "study-nums":        "headline figures",
    "study-src":         "provenance note",
    "finding-lede":      "the finding",
    "rw-split":          "cost split bar",
    "priciest":          "priciest request",
    "cache-lede":        "caching section",
    "findings-projects": "per-project table",
    "total-cost":        "ledger total",
}

def inner(el):
    m = re.search(r'id="%s"[^>]*>(.*?)</(?:div|tbody|p|section)>' % el, html, re.S)
    if not m:
        return None
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", m.group(1))).strip()

failures = []
for el, label in REQUIRED.items():
    v = inner(el)
    if v is None:
        failures.append(f"{label}: element #{el} not in the rendered DOM")
    elif not v:
        failures.append(f"{label}: #{el} rendered empty")
    else:
        print(f"  ok    {label:20} {v[:70]}")

# The page's own error path must not have fired. Checked against the *rendered*
# headline rather than the whole document: the handler's message is a string
# literal inside the inline script, so searching the raw HTML matches the source
# of the check itself and fails every time, error or not.
headline = inner("study-nums") or ""
if "Could not load the measurements" in headline:
    reason = re.search(r"Could not load the measurements \(([^)]*)\)", headline)
    failures.append("the page caught an error: " + (reason.group(1) if reason else "unknown"))

if failures:
    print("\nFAILED:")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("\nall sections rendered")
PY
status=$?
rm -f "$OUT"
exit $status
