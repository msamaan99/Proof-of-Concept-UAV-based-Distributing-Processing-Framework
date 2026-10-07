#!/usr/bin/env bash
# Verify the rig before you spend 45 minutes measuring it.
#
#   ./preflight.sh            every check
#   ./preflight.sh roles      which board holds the master address
#   ./preflight.sh clocks     chrony discipline on all four devices
#   ./preflight.sh engines    the TensorRT engine checksum on both Jetsons
#   ./preflight.sh perf       power mode and pinned clocks on both Jetsons
#   ./preflight.sh radio      Wi-Fi power save on every client
#   ./preflight.sh soak       three-minute ping soak of both sensor links
#
# Every check here exists because skipping it once cost a run that looked
# completely normal and was not usable. Each one names the failure it catches.

set -u
cd "$(dirname "$(readlink -f "$0")")"
. ./hosts.env

FAIL=0
ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$1"; FAIL=1; }
warn() { printf '  \033[33m!\033[0m %s\n' "$1"; }
info() { printf '  · %s\n' "$1"; }
head_(){ printf '\n\033[1m%s\033[0m\n' "$1"; }

sshq() { ssh -o ConnectTimeout=10 -o BatchMode=no "$@"; }

# ---------------------------------------------------------------- roles ----
#
# Master is whoever holds 192.168.50.1, which is whoever claimed the access
# point first — so it can differ between runs with nothing in the logs saying
# so. The boards are not interchangeable: their inference medians have differed
# by up to 1.7x under identical configuration, so swapping the master silently
# inverts a local-vs-offload comparison. Check it before EVERY run and write
# down which board it was.
check_roles() {
    head_ "Master role"
    local h found=""
    for h in "$NANO1_SSH:Nano 1" "$NANO2_SSH:Nano 2"; do
        local target="${h%%:*}" label="${h##*:}" addrs
        addrs=$(sshq "$target" "ip -4 -o addr show wlan0 2>/dev/null | grep -o '192\.168\.50\.[0-9]*'" 2>/dev/null | tr '\n' ' ')
        if [ -z "$addrs" ]; then
            warn "$label ($target): no NanoNet address — not on the network"
            continue
        fi
        if printf '%s' "$addrs" | grep -qw "${MASTER_ADDR##*.}"; then :; fi
        case " $addrs " in
            *" $MASTER_ADDR "*) ok "$label is MASTER — holds $MASTER_ADDR"; found="$label" ;;
            *)                  ok "$label is worker — $addrs" ;;
        esac
    done
    [ -n "$found" ] && info "record '$found was master' alongside the results" \
                    || bad "nobody holds $MASTER_ADDR — there is no master"
}

# --------------------------------------------------------------- clocks ----
#
# Every cross-device figure is a difference between two machines' timestamps, so
# a shared clock is what makes them defensible. Sub-millisecond is what this rig
# holds; anything worse goes in the write-up as the error bar.
#
# The trap is timing, not configuration: chrony cannot correct anything until a
# Nano has built the network and the others have joined, which is 15-60 s after
# power-on. Start a run inside that window and the first minutes are worthless.
check_clocks() {
    head_ "Clock discipline (reference is Nano 2)"
    local h
    for h in "$NANO1_SSH:Nano 1" "$NANO2_SSH:Nano 2" "$PI1_SSH:Pi 1" "$PI2_SSH:Pi 2"; do
        local target="${h%%:*}" label="${h##*:}" out ref sys
        out=$(sshq "$target" 'chronyc tracking 2>/dev/null' 2>/dev/null)
        if [ -z "$out" ]; then warn "$label: chrony not answering"; continue; fi
        ref=$(printf '%s' "$out" | awk -F': *' '/Reference ID/{print $2}')
        sys=$(printf '%s' "$out" | awk -F': *' '/System time/{print $2}')
        case "$ref" in
            00000000*) bad "$label: NO TIME SOURCE yet ($ref) — wait, do not start a run" ;;
            *)         ok  "$label: ref $ref  system time $sys" ;;
        esac
    done
    info "chronyc clients on Nano 2 lists everyone currently drawing time from it"
}

# -------------------------------------------------------------- engines ----
#
# Two boards on different JetPack releases cannot produce comparable engines from
# identical weights. A byte-identical checksum on both is the only way to state
# in a report that the model was the same — and a mismatched engine is a 2x
# latency difference that looks exactly like a scheduling result.
check_engines() {
    head_ "TensorRT engine"
    local a b
    a=$(sshq "$NANO1_SSH" 'md5sum ~/yolov5/yolov5n.engine 2>/dev/null | cut -c1-32')
    b=$(sshq "$NANO2_SSH" 'md5sum ~/yolov5/yolov5n.engine 2>/dev/null | cut -c1-32')
    [ -n "$a" ] && ok "Nano 1  $a" || bad "Nano 1: no engine at ~/yolov5/yolov5n.engine"
    [ -n "$b" ] && ok "Nano 2  $b" || bad "Nano 2: no engine at ~/yolov5/yolov5n.engine"
    if [ -n "$a" ] && [ -n "$b" ]; then
        [ "$a" = "$b" ] && ok "engines identical" \
          || warn "engines DIFFER — expected if each board built its own; state it in the report"
    fi
}

# ----------------------------------------------------------------- perf ----
#
# `jetson_clocks` does NOT survive a reboot or a power cycle. After booting, the
# GPU sits at its 76.8 MHz floor and ramps per frame, which inflates inference
# badly at low duty cycle — and the inflation is largest on the board doing the
# least work, i.e. exactly the worker whose cost the scheduler is estimating.
# Re-run it on both boards after every power-on, before any run you intend to
# quote.
check_perf() {
    head_ "Jetson power mode and clocks"
    local h
    for h in "$NANO1_SSH:Nano 1" "$NANO2_SSH:Nano 2"; do
        local target="${h%%:*}" label="${h##*:}" mode gpu
        mode=$(sshq "$target" 'sudo -n nvpmodel -q 2>/dev/null | tail -1' 2>/dev/null)
        gpu=$(sshq "$target" 'cat /sys/devices/gpu.0/devfreq/57000000.gpu/min_freq 2>/dev/null')
        [ -n "$mode" ] && info "$label power mode: $mode" || warn "$label: nvpmodel needs sudo — run it at the board"
        if [ -n "$gpu" ]; then
            if [ "$gpu" -ge 900000000 ] 2>/dev/null; then
                ok "$label GPU pinned at $((gpu/1000000)) MHz"
            else
                bad "$label GPU floor is $((gpu/1000000)) MHz — run: sudo jetson_clocks"
            fi
        else
            warn "$label: could not read the GPU devfreq node"
        fi
    done
    info "to fix:  ssh <board> 'sudo nvpmodel -m 0 && sudo jetson_clocks'"
}

# ---------------------------------------------------------------- radio ----
#
# Wi-Fi power save parks the radio between beacons. On this rig it showed up as
# the link dropping every 20-40 s in contiguous 120-140 frame blocks — which
# reads in the results as the scheduler shedding frames.
check_radio() {
    head_ "Wi-Fi power save (must be off on every client)"
    local h
    for h in "$PI1_SSH:Pi 1" "$PI2_SSH:Pi 2" "$NANO1_SSH:Nano 1" "$NANO2_SSH:Nano 2"; do
        local target="${h%%:*}" label="${h##*:}" ps
        ps=$(sshq "$target" 'iw dev wlan0 get power_save 2>/dev/null' 2>/dev/null | grep -o 'on\|off')
        case "$ps" in
            off) ok "$label: power save off" ;;
            on)  bad "$label: power save ON — fix: sudo iw dev wlan0 set power_save off" ;;
            *)   warn "$label: could not read power save" ;;
        esac
    done
    local laptop_if
    laptop_if=$(nmcli -t -f DEVICE,TYPE device status 2>/dev/null | awk -F: '$2=="wifi"{print $1; exit}')
    if [ -n "$laptop_if" ]; then
        local lps
        lps=$(iw dev "$laptop_if" get power_save 2>/dev/null | grep -o 'on\|off')
        [ "$lps" = "off" ] && ok "laptop ($laptop_if): power save off" \
                           || bad "laptop ($laptop_if): power save $lps — it is the other end of the GCS hop"
    fi
}

# ----------------------------------------------------------------- soak ----
#
# The gate. A measurement run over an unstable link is a wasted 45 minutes, and
# the instability is invisible in the results — it looks like drops.
# Want: 0% loss and a maximum under about 200 ms.
check_soak() {
    head_ "Link soak — 180 s per sensor, from the laptop"
    info "this is the gate: 0% loss, max under ~200 ms, or do not start"
    local t
    for t in "$PI1_NANONET:Pi 1" "$PI2_NANONET:Pi 2" "$MASTER_ADDR:master"; do
        local addr="${t%%:*}" label="${t##*:}"
        printf '  · %-8s %s ... ' "$label" "$addr"
        local out
        out=$(ping -i 0.2 -w 180 -q "$addr" 2>/dev/null | tail -3)
        local loss rtt
        loss=$(printf '%s' "$out" | grep -o '[0-9.]*% packet loss' | head -1)
        rtt=$(printf '%s' "$out"  | grep -o 'min/avg/max.*' | head -1)
        printf '\n'
        if [ -z "$loss" ]; then bad "$label unreachable at $addr"; continue; fi
        case "$loss" in
            "0% packet loss") ok "$label  $loss  $rtt" ;;
            *)                bad "$label  $loss  $rtt  ← fix the link first" ;;
        esac
    done
}

case "${1:-all}" in
  roles)   check_roles ;;
  clocks)  check_clocks ;;
  engines) check_engines ;;
  perf)    check_perf ;;
  radio)   check_radio ;;
  soak)    check_soak ;;
  all)     check_roles; check_clocks; check_engines; check_perf; check_radio
           head_ "Skipped"
           info "soak takes 9 minutes and is not run by 'all' — run ./preflight.sh soak before a measurement set" ;;
  *)       sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac

head_ "Verdict"
[ "$FAIL" = "0" ] && ok "clear to run" || bad "at least one check failed — the run will not be quotable"
exit "$FAIL"
