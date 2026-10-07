#!/bin/bash
# Swarm time sync — chrony, with Nano 2 as the reference clock.
#
#   ./chrony_setup.sh nano2     the time server        🔵
#   ./chrony_setup.sh nano1     fallback server        🟣
#   ./chrony_setup.sh pi        sensor node            🟢
#   ./chrony_setup.sh laptop    GCS (optional)         ⚪
#   ./chrony_setup.sh check     verify, any device
#
# INSTALL BEFORE JOINING NanoNet. `apt install` needs the archives, and NanoNet
# has no route to the internet. Run this on each device while it is still on
# your normal network, then move it onto the swarm.
#
# Safe to re-run: the block it writes is delimited and replaced, never doubled.

set -u

CONF="/etc/chrony/chrony.conf"
BEGIN="# >>> swarm time sync (managed by chrony_setup.sh) >>>"
END="# <<< swarm time sync (managed by chrony_setup.sh) <<<"

NANO2_CLIENT="192.168.50.55"     # Nano 2 while it is a worker
MASTER_ADDR="192.168.50.1"       # whichever Nano currently holds the AP
LAPTOP="192.168.50.23"
SUBNET="192.168.50.0/24"

ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$1"; }
info() { printf '  · %s\n' "$1"; }
head_() { printf '\n\033[1m%s\033[0m\n' "$1"; }

# ---------------------------------------------------------------- install ---

install_chrony() {
    # chronyd lives in /usr/sbin, which is NOT on a non-root user's PATH over a
    # non-login ssh command on Debian. `command -v chronyd` alone reports an
    # installed chrony as missing, and the script then wrongly blames the
    # network. Ask dpkg and the absolute path too.
    local chronyd=""
    for c in chronyd /usr/sbin/chronyd /sbin/chronyd; do
        command -v "$c" >/dev/null 2>&1 && { chronyd="$c"; break; }
        [ -x "$c" ] && { chronyd="$c"; break; }
    done
    if [ -z "$chronyd" ] && dpkg -s chrony >/dev/null 2>&1; then
        chronyd="/usr/sbin/chronyd"          # installed, just not on PATH
    fi

    if [ -n "$chronyd" ]; then
        ok "chrony already installed ($("$chronyd" -v 2>&1 | head -1 | cut -d, -f1))"
        return 0
    fi

    # ICMP is filtered on plenty of networks, so a failed ping is not proof.
    # Fall back to a TCP connect before declaring the device offline.
    if ! ping -c1 -W3 8.8.8.8 >/dev/null 2>&1 &&
       ! timeout 5 bash -c 'exec 3<>/dev/tcp/1.1.1.1/443' 2>/dev/null; then
        bad "no internet — chrony is not installed and apt cannot reach the archives"

        # The usual cause on this rig: the NanoNet profile sets
        # ipv4.gateway 192.168.50.1, which becomes the default route. NanoNet
        # has no way out, so the device is offline even while a working NAT
        # path sits on the wired interface.
        if ip route show default 2>/dev/null | grep -q '192\.168\.50\.1'; then
            info "default route points into NanoNet (192.168.50.1), which has no way out"
            info "drop the wifi, install, then bring it back:"
            info "  sudo nmcli connection down NanoNet && sudo apt-get install -y chrony; sudo nmcli connection up NanoNet"
        else
            info "connect this device to your normal network, re-run, then join NanoNet"
        fi
        return 1
    fi

    sudo apt-get update -qq && sudo apt-get install -y chrony || {
        bad "apt install chrony failed"; return 1; }
    ok "chrony installed"
}

# systemd-timesyncd and chrony both want to steer the clock. Only one may.
disable_timesyncd() {
    if systemctl list-unit-files 2>/dev/null | grep -q '^systemd-timesyncd'; then
        if systemctl is-enabled systemd-timesyncd >/dev/null 2>&1 || \
           systemctl is-active  systemd-timesyncd >/dev/null 2>&1; then
            sudo systemctl disable --now systemd-timesyncd >/dev/null 2>&1
            ok "systemd-timesyncd disabled — it would fight chrony for the clock"
        fi
    fi
}

# ------------------------------------------------------------ config write --

# write_block <<< "directives"
write_block() {
    local body; body=$(cat)

    [ -f "$CONF" ] || { bad "$CONF not found — is this really chrony?"; return 1; }

    sudo cp "$CONF" "$CONF.bak.$(date +%Y%m%d%H%M%S)"

    # Drop any previous managed block, plus the older hand-appended lines from
    # BUILD_GUIDE that pointed everything at the laptop.
    sudo sed -i "/$(printf '%s' "$BEGIN" | sed 's/[][\.*^$/]/\\&/g')/,/$(printf '%s' "$END" | sed 's/[][\.*^$/]/\\&/g')/d" "$CONF"
    sudo sed -i "\#^server ${LAPTOP}#d" "$CONF"

    printf '\n%s\n%s\n%s\n' "$BEGIN" "$body" "$END" | sudo tee -a "$CONF" >/dev/null
    ok "wrote config block to $CONF"
}

restart_chrony() {
    local svc=chrony
    systemctl list-unit-files 2>/dev/null | grep -q '^chrony\.service' || svc=chronyd
    sudo systemctl enable "$svc" >/dev/null 2>&1
    if sudo systemctl restart "$svc"; then
        ok "$svc restarted"
    else
        bad "$svc failed to restart — run: journalctl -u $svc -n 30"
        return 1
    fi
}

open_ntp_port() {
    command -v ufw >/dev/null 2>&1 || return 0
    if sudo ufw status 2>/dev/null | grep -q "^Status: active"; then
        sudo ufw allow from "$SUBNET" to any port 123 proto udp >/dev/null 2>&1
        ok "ufw: UDP 123 opened to $SUBNET"
    else
        info "ufw inactive — nothing to open"
    fi
}

# ------------------------------------------------------------------ roles ---

role_nano2() {
    head_ "🔵 Nano 2 — the swarm's reference clock"
    install_chrony || return 1
    disable_timesyncd

    # `local stratum 8` lets Nano 2 serve its own clock with no upstream at all,
    # which is the normal case: NanoNet has no internet. Stratum 8 beats the 10
    # Nano 1 advertises, so every client prefers Nano 2 whenever it is reachable.
    #
    # The laptop line is opportunistic. It is the only device with a
    # battery-backed clock, so when it is on the network Nano 2 inherits real
    # wall-clock time from it; when it is absent Nano 2 falls back to `local`
    # and the swarm still agrees with itself. Absolute time being wrong costs
    # nothing here — every cross-device latency is a difference, not an instant.
    write_block <<EOF
server ${LAPTOP} iburst minpoll 0 maxpoll 2
local stratum 8
allow ${SUBNET}
makestep 0.01 -1
EOF

    restart_chrony || return 1
    open_ntp_port

    info "clients reach this board at ${NANO2_CLIENT}, or at ${MASTER_ADDR} when it is master"
}

role_nano1() {
    head_ "🟣 Nano 1 — follows Nano 2, serves only if Nano 2 is gone"
    install_chrony || return 1
    disable_timesyncd

    # Stratum 10 is deliberately worse than Nano 2's 8. Both boards answer NTP,
    # but a client holding both sources picks the lower stratum, so Nano 1 only
    # becomes the reference once Nano 2 stops answering.
    #
    # `trust` is load-bearing, not decoration. While Nano 1 is master it holds
    # .1 and the second line below polls *itself* — a second opinion that always
    # agrees with its own clock. Without `trust`, chrony sees two sources far
    # apart, cannot form a majority, marks both 'x' and syncs to neither, which
    # is exactly the deadlock observed on 2026-09-04 (Nano 2 +3663 ms, both
    # sources 'x', reference ID 7F7F0101). `trust` assumes Nano 2 is correct so
    # the self-poll cannot veto it.
    write_block <<EOF
server ${NANO2_CLIENT} iburst prefer trust minpoll 0 maxpoll 2
server ${MASTER_ADDR} iburst minpoll 0 maxpoll 2
local stratum 10
allow ${SUBNET}
makestep 0.01 -1
EOF

    restart_chrony || return 1
    open_ntp_port
}

role_client() {
    local label="$1"
    head_ "$label — follows Nano 2"
    install_chrony || return 1
    disable_timesyncd

    # Two lines for one server. Nano 2 answers on .55 while it is a worker and
    # on .1 once it claims the AP, and nothing tells a client which is true
    # right now — so it carries both and lets chrony use whichever replies.
    #
    # `trust` on the .55 line settles which one wins. When Nano 1 is master, .1
    # is Nano 1 serving its own clock, and if the two disagree chrony refuses to
    # pick either — it marks both 'x' and syncs to nothing. Stratum alone does
    # not break that tie. `trust` says Nano 2 is correct by definition, which is
    # the whole point of designating it the reference.
    #
    # makestep 0.01 -1 steps the clock outright, at any time, for offsets over
    # 10 ms. These boards have no battery-backed RTC; slewing a boot-time offset
    # would take hours, and the run starts in minutes.
    write_block <<EOF
server ${NANO2_CLIENT} iburst prefer trust minpoll 0 maxpoll 2
server ${MASTER_ADDR} iburst minpoll 0 maxpoll 2
makestep 0.01 -1
EOF

    restart_chrony || return 1
}

# ------------------------------------------------------------------ check ---

role_check() {
    head_ "Checking time sync"

    command -v chronyc >/dev/null 2>&1 || { bad "chrony not installed here"; return 1; }

    local svc=chrony
    systemctl list-unit-files 2>/dev/null | grep -q '^chrony\.service' || svc=chronyd
    systemctl is-active "$svc" >/dev/null 2>&1 && ok "$svc running" || bad "$svc not running"

    printf '\n'
    chronyc tracking | grep -E "Reference ID|Stratum|System time|RMS offset"
    printf '\n'
    chronyc sources -v 2>/dev/null | tail -n +2

    local ref
    ref=$(chronyc tracking 2>/dev/null | awk '/Reference ID/{print $4}')
    printf '\n'
    if [ "$ref" = "00000000" ] || [ -z "$ref" ]; then
        bad "no time source yet — clock is NOT synchronised"
        info "on a client: can it reach the server?   ping -c3 ${NANO2_CLIENT}"
        info "on Nano 2:   is UDP 123 open?           sudo ufw status"
        info "chrony needs the network up first; give it 30–60 s after boot"
    else
        ok "synchronised to $ref"
    fi

    # Nano 2 can prove who is actually drawing time from it.
    if chronyc clients >/dev/null 2>&1; then
        printf '\n'
        info "clients served by this device:"
        chronyc clients 2>/dev/null | head -12
    fi
}

# ------------------------------------------------------------------- main ---

case "${1:-}" in
    nano2)  role_nano2 ;;
    nano1)  role_nano1 ;;
    pi)     role_client "🟢 Raspberry Pi" ;;
    laptop) role_client "⚪ Laptop" ;;
    check)  role_check ;;
    *)
        sed -n '2,15p' "$0" | sed 's/^# \?//'
        exit 1
        ;;
esac
