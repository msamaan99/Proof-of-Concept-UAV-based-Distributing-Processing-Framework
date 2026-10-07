#!/usr/bin/env bash
# Copy the code to the rig. One command per target, or all of them at once.
#
#   ./deploy.sh nano1          ./deploy.sh pi1       ./deploy.sh nanos
#   ./deploy.sh nano2          ./deploy.sh pi2       ./deploy.sh pis
#   ./deploy.sh all
#
# Run it from anywhere; paths are resolved from the script's own location.
#
# WHY A SCRIPT AND NOT A LIST OF scp LINES
# ────────────────────────────────────────
# Three deploy accidents happened on this rig, and each one produced a run whose
# numbers looked completely normal:
#
#   1. `scp swarm_net.py` to both boards overwrote Nano 2's identity with
#      Nano 1's, because NODE_ID was a hand-edited literal. Both boards then
#      booted believing they were nano1, took the same claim delay, and raced for
#      the access point. Fixed in the code (identity now comes from the
#      hostname), but the class of accident is what this script guards against.
#   2. A stale __pycache__ meant a board kept running the previous scheduler
#      after the source was replaced. Python only recompiles when the source
#      mtime is newer, and scp preserves nothing by default — so this always
#      removes the cache rather than trusting the timestamp.
#   3. One board got the new file and the other did not, so a failover mid-run
#      silently swapped schedulers. Every copy here is checksum-verified against
#      the local file and the script exits non-zero on any mismatch.
#
# It never restarts a service on its own. Deciding when a node may drop out is
# the operator's call, and the command to do it is printed at the end.

set -u
cd "$(dirname "$(readlink -f "$0")")"
REPO="$(cd .. && pwd)"
. ./hosts.env

FAIL=0
ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$1"; FAIL=1; }
info() { printf '  · %s\n' "$1"; }
head_(){ printf '\n\033[1m%s\033[0m\n' "$1"; }

# md5 of a local file, portable between coreutils and busybox.
md5of() { md5sum "$1" | awk '{print $1}'; }

# push <ssh-target> <remote-dir> <file>...
#
# Verifies every file after the copy. A silent scp failure — full disk, wrong
# permissions, a path that does not exist — is indistinguishable from success
# without this, and the consequence is a board running last week's scheduler.
push() {
    local target="$1"; shift
    local dir="$1"; shift
    ssh -o ConnectTimeout=10 "$target" "mkdir -p $dir" || { bad "$target: cannot create $dir"; return 1; }
    scp -q "$@" "$target:$dir/" || { bad "$target: scp failed"; return 1; }

    # Clear the bytecode cache before checking, so the next run cannot import a
    # stale .pyc compiled from the file we just replaced.
    ssh "$target" "rm -rf $dir/__pycache__"

    local names=() f
    for f in "$@"; do names+=("$(basename "$f")"); done
    local remote
    remote=$(ssh "$target" "cd $dir && md5sum ${names[*]} 2>/dev/null")
    for f in "$@"; do
        local base want got
        base=$(basename "$f"); want=$(md5of "$f")
        got=$(printf '%s\n' "$remote" | awk -v n="$base" '$2==n{print $1}')
        if [ "$want" = "$got" ]; then
            ok "$base  ${want:0:8}"
        else
            bad "$base  local ${want:0:8} vs remote ${got:0:8}  ← MISMATCH"
        fi
    done
}

# Units carry placeholders rather than a hardcoded user, so the same file works
# on both Jetsons (different logins) and both Pis. Substituting here, at deploy
# time, keeps one copy in git instead of four near-identical ones.
install_unit() {
    local target="$1" src="$2" name="$3"; shift 3
    local user tmp
    user=$(ssh "$target" 'whoami') || { bad "$target: cannot read login"; return 1; }
    tmp=$(mktemp)
    sed "s|NANO_USER|$user|g; s|PI_USER|$user|g" "$src" > "$tmp"
    # Extra per-unit substitutions, e.g. SENSOR_ID_VALUE=pi2
    while [ $# -ge 2 ]; do sed -i "s|$1|$2|g" "$tmp"; shift 2; done
    scp -q "$tmp" "$target:/tmp/$name" && \
      ssh "$target" "sudo install -m 0644 /tmp/$name /etc/systemd/system/$name && rm -f /tmp/$name && sudo systemctl daemon-reload" \
      && ok "unit $name installed (User=$user)" || bad "unit $name failed"
    rm -f "$tmp"
}

deploy_nano() {
    local label="$1" target="$2"
    head_ "$label  →  $target:$NANO_DIR"
    # frame_trace.py is a symlink to ../shared/. scp follows it and transfers the
    # real content, so the board gets a plain file and there is still exactly one
    # copy of the source in git.
    push "$target" "$NANO_DIR" \
        "$REPO/nano/mec_node.py" \
        "$REPO/nano/swarm_net.py" \
        "$REPO/nano/completion_time_scheduler.py" \
        "$REPO/nano/scheduler_lyapunov.py" \
        "$REPO/nano/scheduler_rr.py" \
        "$REPO/nano/scheduler_fixed.py" \
        "$REPO/nano/frame_trace.py" \
        "$REPO/nano/bench_infer.py" \
        "$REPO/nano/engine_info.py"
    if [ "${WITH_UNITS:-0}" = "1" ]; then
        install_unit "$target" "$REPO/nano/systemd/swarm_net.service" swarm_net.service
        install_unit "$target" "$REPO/nano/systemd/mec_node.service"  mec_node.service
    fi
}

deploy_pi() {
    local label="$1" target="$2" sid="$3"
    head_ "$label  →  $target:$PI_DIR   (sensor id $sid)"
    push "$target" "$PI_DIR" \
        "$REPO/pi/pi_sensor.py" \
        "$REPO/pi/nanonet_link.py"
    if [ "${WITH_UNITS:-0}" = "1" ]; then
        install_unit "$target" "$REPO/pi/systemd/nanonet-link.service" nanonet-link.service
        # SENSOR_ID_VALUE is the one setting that MUST differ between the boards.
        # Two Pis streaming as the same id interleave in one FrameStore keyspace
        # and silently overwrite each other — the run completes and every number
        # in it is wrong.
        install_unit "$target" "$REPO/pi/systemd/pi-sensor.service" pi-sensor.service \
            SENSOR_ID_VALUE "$sid"
    fi
}

case "${1:-}" in
  nano1) deploy_nano "Nano 1" "$NANO1_SSH" ;;
  nano2) deploy_nano "Nano 2" "$NANO2_SSH" ;;
  nanos) deploy_nano "Nano 1" "$NANO1_SSH"; deploy_nano "Nano 2" "$NANO2_SSH" ;;
  pi1)   deploy_pi   "Pi 1"   "$PI1_SSH" "$PI1_SENSOR_ID" ;;
  pi2)   deploy_pi   "Pi 2"   "$PI2_SSH" "$PI2_SENSOR_ID" ;;
  pis)   deploy_pi   "Pi 1"   "$PI1_SSH" "$PI1_SENSOR_ID"; deploy_pi "Pi 2" "$PI2_SSH" "$PI2_SENSOR_ID" ;;
  all)   deploy_nano "Nano 1" "$NANO1_SSH"; deploy_nano "Nano 2" "$NANO2_SSH"
         deploy_pi   "Pi 1"   "$PI1_SSH" "$PI1_SENSOR_ID"; deploy_pi "Pi 2" "$PI2_SSH" "$PI2_SENSOR_ID" ;;
  *)
    sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
    echo
    echo "  WITH_UNITS=1 also installs and reloads the systemd units:"
    echo "      WITH_UNITS=1 ./deploy.sh all"
    exit 2 ;;
esac

head_ "Result"
if [ "$FAIL" = "0" ]; then
    ok "every file verified byte-identical on every target"
    cat <<'NEXT'

  Nothing was restarted. Apply the new code when you are ready:

      # Jetsons — network layer, then processing layer
      ssh admindesktop1@10.42.0.226 'sudo systemctl restart swarm_net mec_node'
      ssh admindesktop@10.42.0.43   'sudo systemctl restart swarm_net mec_node'

      # Raspberry Pis
      ssh admin@10.42.0.31   'sudo systemctl restart nanonet-link pi-sensor'
      ssh pi02@10.42.0.128   'sudo systemctl restart nanonet-link pi-sensor'

  Restart the MASTER last. Restarting it drops the access point, which takes
  every other device off the network with it.
NEXT
else
    bad "at least one file did not verify — fix it before running anything"
fi
exit "$FAIL"
