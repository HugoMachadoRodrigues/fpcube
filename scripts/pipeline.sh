#!/bin/bash
# Self-healing driver for the Florida Pasture Cube build on a single machine.
#
#   caffeinate -i nohup bash scripts/pipeline.sh > ~/fpcube_data/pipeline.log 2>&1 &
#
# Every extraction step is resumable, so the driver simply (re)starts any step that is not done and not running,
# kills a step whose log has been silent for too long (hung network read) and restarts it, and gives up on a step
# after MAX_TRIES failures. When all extraction steps are done it runs the downstream steps once, in order:
# cube -> validation -> benchmark -> release -> paper numbers -> figures -> LaTeX.
# State: $FPCUBE_DATA/state/<step>.done | .tries ; logs: $FPCUBE_DATA/logs/<step>.log
# Stop it with:  touch ~/fpcube_data/state/STOP

set -u
DATA=${FPCUBE_DATA:-$HOME/fpcube_data}
PY=${FPCUBE_PY:-$HOME/fpcube_env/bin/python}
REPO="$(cd "$(dirname "$0")/.." && pwd)"
MS="$REPO/manuscript"
STATE="$DATA/state"; LOGS="$DATA/logs"
mkdir -p "$STATE" "$LOGS"
MAX_TRIES=30
CHECK_EVERY=120

# step name | command | stall limit in hours (log silent this long while running => kill and restart)
STEPS=(
  "prism|$PY -m fpcube.sources.prism|3"
  "radar|env FPCUBE_S1_WORKERS=4 FPCUBE_S1_BLOCK_THREADS=8 $PY -m fpcube.sources.radar|2"
  "optical|env FPCUBE_THREADS=24 $PY -m fpcube.sources.optical|3"
  "terrain|$PY -m fpcube.sources.static terrain|16"
  "soil|$PY -m fpcube.sources.static soil|3"
  "wetlands|$PY -m fpcube.sources.static wetlands|3"
  "labels|$PY -m fpcube.sources.labels usdm hurricanes|2"
)
NASS_KEY_FILE="$DATA/.nass_key"     # optional: one line with the NASS Quick Stats key

log() { echo "$(date '+%F %T') $*"; }
done_() { [ -f "$STATE/$1.done" ]; }
running() { pgrep -f -- "$2" >/dev/null 2>&1; }
pattern_of() { echo "$1" | sed -E 's/^env ([A-Z0-9_]+=[^ ]+ )+//; s#^[^ ]+/python ##'; }   # "-m fpcube.sources.x ..."

start_step() {
  local name=$1 cmd=$2
  local tries; tries=$(cat "$STATE/$name.tries" 2>/dev/null || echo 0)
  if [ "$tries" -ge "$MAX_TRIES" ]; then return; fi
  echo $((tries + 1)) > "$STATE/$name.tries"
  log "start $name (try $((tries + 1)))"
  ( cd "$HOME" && bash -c "$cmd" >> "$LOGS/$name.log" 2>&1 && touch "$STATE/$name.done" \
      && echo "$(date '+%F %T') done $name" >> "$DATA/pipeline.log" ) &
}

stalled() {   # log untouched for more than $2 hours
  local f="$LOGS/$1.log"; [ -f "$f" ] || return 1
  local age=$(( $(date +%s) - $(stat -f %m "$f") ))
  [ "$age" -gt $(( $2 * 3600 )) ]
}

downstream() {
  local s
  for s in \
    "cube|$PY -m fpcube.cube" \
    "validation|$PY -m fpcube.validation" \
    "benchmark|$PY -m fpcube.benchmark.run" \
    "release|$PY -m fpcube.release" \
    "numbers|$PY $REPO/scripts/paper_numbers.py" \
    "figures|$PY $REPO/scripts/figures.py fig1 fig2 fig3 fig4 fig5" \
    "latex|cd '$MS' && latexmk -pdf -interaction=nonstopmode -f fpcube_essd.tex"; do
    local name=${s%%|*} cmd=${s#*|}
    done_ "$name" && continue
    log "run $name"
    if (cd "$HOME" && bash -c "$cmd") >> "$LOGS/$name.log" 2>&1; then
      touch "$STATE/$name.done"; log "done $name"
    else
      log "FAILED $name (see $LOGS/$name.log); downstream stopped"; return 1
    fi
  done
  log "ALL DONE"
}

log "pipeline driver started (pid $$)"
while true; do
  [ -f "$STATE/STOP" ] && { log "STOP file found, driver exits (running steps keep running)"; exit 0; }
  all_done=1
  for s in "${STEPS[@]}"; do
    IFS='|' read -r name cmd stall <<< "$s"
    done_ "$name" && continue
    all_done=0
    pat=$(pattern_of "$cmd")
    if running "$name" "$pat"; then
      if stalled "$name" "$stall"; then
        log "$name silent for > ${stall} h: killing for restart"
        pkill -f -- "$pat"; sleep 10
      fi
      continue
    fi
    start_step "$name" "$cmd"
  done
  if [ -f "$NASS_KEY_FILE" ] && ! done_ nass && ! running nass "fpcube.sources.labels nass"; then
    start_step nass "FPCUBE_NASS_KEY=\$(cat '$NASS_KEY_FILE') $PY -m fpcube.sources.labels nass"
  fi
  if [ "$all_done" = 1 ]; then
    downstream
    exit $?
  fi
  sleep $CHECK_EVERY
done
