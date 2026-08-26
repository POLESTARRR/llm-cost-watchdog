#!/bin/bash
# One command that says whether everything is actually working.
#
#   bash scripts/status.sh
#
# Written because "is this project fine?" was being answered by memory, and
# memory was wrong several times: a page that rendered blank in a browser while
# its HTML looked perfect, a scheduled job reporting success while macOS blocked
# it, a header contradicting the numbers below it. Every line here runs a real
# check rather than reporting a belief.
cd "$(dirname "$0")/.." || exit 1
ok() { printf "  \033[32mok\033[0m   %s\n" "$1"; }
no() { printf "  \033[31mFAIL\033[0m %s\n" "$1"; }

echo
echo "llm-cost-gateway  ($(pwd))"
echo

[ -x venv/bin/python ] && ok "venv present" || { no "no venv — run: python3 -m venv venv && venv/bin/pip install -r requirements.txt -e ."; exit 1; }

t=$(venv/bin/python -m pytest tests/ -q -p no:randomly 2>&1 | tail -1)
case "$t" in *"passed"*) ok "tests — $t";; *) no "tests — $t";; esac

venv/bin/ccost >/dev/null 2>&1 && ok "ccost runs" || no "ccost broken"

# The scheduled job deliberately runs from ~/code/ccost-sync, not from here.
# macOS refuses background agents access to ~/Desktop, and this working copy
# lives there, so the job would install, report success and never run. A probe
# agent confirmed it can read ~/.claude and ~/code but not ~/Desktop, and the
# sync only ever needed the transcripts, some code and the network.
listing="$(launchctl list 2>/dev/null || true)"
case "$listing" in
  *ccost.sync*) code=$(echo "$listing" | grep ccost.sync | awk '{print $2}')
      [ "$code" = "0" ] && ok "daily sync job loaded, last run ok (~/code/ccost-sync)" \
                        || no "daily sync job loaded but last run exited $code" ;;
  *) no "daily sync job not installed — see ~/code/ccost-sync/run-sync.sh" ;;
esac

grep -q ccost "$HOME/.claude/settings.json" 2>/dev/null \
  && ok "mid-session hook installed" \
  || no "hook not installed — run: venv/bin/python scripts/ccost.py hook --help"

git diff --quiet && git diff --cached --quiet && ok "working tree clean" || no "uncommitted changes"
[ -z "$(git log origin/main..HEAD --oneline 2>/dev/null)" ] && ok "everything pushed" || no "unpushed commits"

venv/bin/python scripts/check_live.py 2>/dev/null | tail -1 | grep -q consistent \
  && ok "live site consistent" || no "live site check failed"

echo
echo "  ccost            what this session is costing"
echo "  ccost week       the last seven days"
echo "  ccost report     a shareable HTML file"
echo
