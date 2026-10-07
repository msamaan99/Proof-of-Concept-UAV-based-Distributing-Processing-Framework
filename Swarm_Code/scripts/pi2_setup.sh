#!/bin/bash
# Pi 2 sensor node setup — the Pi-1 build, with Pi 2's address and sensor id.
#
#   ./pi02_setup.sh apt     repair apt (UA-filtered upstream)   NEEDS INTERNET
#   ./pi02_setup.sh deps    install python deps + chrony        NEEDS INTERNET
#   ./pi02_setup.sh time    point chrony at Nano 2
#   ./pi02_setup.sh host    set hostname to pi2
#   ./pi02_setup.sh cam     camera check
#   ./pi02_setup.sh eth     keep ssh-over-ethernet working
#   ./pi02_setup.sh net     join NanoNet as .12                 keeps internet
#   ./pi02_setup.sh check   verify everything
#   ./pi02_setup.sh prep    apt + deps + time + host + cam + eth
#
# Order matters: everything needing the archives runs first, then `net` moves
# the board onto the swarm, which has no route out. Safe to re-run any stage.
#
# Differences from pi_setup.sh, which is Pi 1's: PI_ADDR is .12 not .11, the
# NanoNet profile carries an autoconnect priority, and the apt stage exists at
# all (Pi 2 sits behind an upstream that 403s apt's User-Agent).

set -u

SSID="NanoNet"
PSK="nanonet123"                 # real value, from pi_setup.sh — BUILD_GUIDE prints a placeholder
PI_ADDR="192.168.50.12"          # Pi 2. Pi 1 is .11
GATEWAY="192.168.50.1"
SENSOR_DIR="$HOME/camera_node"
AUTOCONNECT_PRIORITY=100         # highest of any saved profile → NanoNet always wins

ok()    { printf '  \033[32m✓\033[0m %s\n' "$1"; }
bad()   { printf '  \033[31m✗\033[0m %s\n' "$1"; }
info()  { printf '  · %s\n' "$1"; }
head_() { printf '\n\033[1m%s\033[0m\n' "$1"; }

# ------------------------------------------------------------------- apt ----

stage_apt() {
    head_ "Checking apt"

    # As of 2026-09-08 apt works over https on this board and needs no help.
    # Touch nothing unless it is actually broken — the workaround below is a
    # downgrade to plaintext and should not be applied speculatively.
    if sudo apt-get update; then
        ok "apt update clean — no repair needed"
        return 0
    fi

    bad "apt update failed — applying the known workaround"

    # Check the clock first. These boards have no battery-backed RTC, and a
    # date that is months off fails certificate validation with an error that
    # looks nothing like a clock problem.
    info "clock reads: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

    # This network 403s any request whose User-Agent looks like apt. Verified
    # from the laptop on 2026-09-08: the identical URL returned 200 as
    # curl/8.5.0 and 403 as "Debian APT-HTTP/1.3". Not the Pi, not the NAT.
    sudo tee /etc/apt/apt.conf.d/99useragent >/dev/null <<'EOF'
Acquire::http::User-Agent "Mozilla/5.0 (X11; Linux aarch64)";
Acquire::https::User-Agent "Mozilla/5.0 (X11; Linux aarch64)";
EOF
    ok "User-Agent override written"

    if sudo apt-get update; then
        ok "apt update clean with the UA override — sources left on https"
        return 0
    fi

    # Only now fall back to plaintext, and say so loudly.
    bad "still failing on https — falling back to http"
    for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; do
        [ -f "$f" ] || continue
        sudo sed -i 's|https://\(deb\.debian\.org\|archive\.raspberrypi\.com\|security\.debian\.org\)|http://\1|g' "$f"
    done
    sudo apt-get update && ok "apt update clean over http" || { bad "apt still broken"; return 1; }
}

# ------------------------------------------------------------------ deps ----

stage_deps() {
    head_ "Installing dependencies"
    mkdir -p "$SENSOR_DIR" && ok "$SENSOR_DIR ready"

    if ! ping -c1 -W3 8.8.8.8 >/dev/null 2>&1 &&
       ! timeout 5 bash -c 'exec 3<>/dev/tcp/1.1.1.1/443' 2>/dev/null; then
        bad "no internet — this stage needs the archives"
        if ip route show default 2>/dev/null | grep -q '192\.168\.50\.1'; then
            info "default route points into NanoNet, which has no way out:"
            info "  sudo nmcli connection down NanoNet   # then re-run, then bring it back up"
        fi
        return 1
    fi

    sudo apt-get install -y python3-zmq python3-opencv python3-numpy \
                            python3-picamera2 network-manager iw chrony || {
        bad "apt install failed — run the 'apt' stage first"; return 1; }

    python3 - <<'PY'
import sys
for mod in ("cv2", "zmq", "numpy"):
    try:
        m = __import__(mod)
        print("  \033[32m✓\033[0m %s %s" % (mod, getattr(m, "__version__", "")))
    except ImportError:
        print("  \033[31m✗\033[0m %s MISSING" % mod)
        sys.exit(1)
try:
    import picamera2
    print("  \033[32m✓\033[0m picamera2 (CSI camera path)")
except ImportError:
    print("  · no picamera2 — will fall back to /dev/video via OpenCV")
PY
}

# ------------------------------------------------------------------ time ----

stage_time() {
    head_ "Time sync"

    # chrony_setup.sh is the single source of truth for the swarm's chrony
    # config — the `trust` option and makestep 0.01 -1 are load-bearing and
    # explained in that file. Do not hand-write the block here.
    local d; d="$(dirname "$0")"
    for cand in "$d/chrony_setup.sh" "$SENSOR_DIR/chrony_setup.sh" "$HOME/chrony_setup.sh"; do
        if [ -x "$cand" ]; then
            "$cand" pi
            return $?
        fi
    done
    bad "chrony_setup.sh not found — copy it to $SENSOR_DIR and re-run"
    return 1
}

# ------------------------------------------------------------------ host ----

stage_host() {
    head_ "Hostname"
    local cur; cur=$(hostname)
    if [ "$cur" = "pi2" ]; then
        ok "already pi2"
    else
        sudo hostnamectl set-hostname pi2 && ok "hostname $cur → pi2"
        info "the login stays pi02 and PI_SENSOR_ID stays pi2 — three similar strings, not interchangeable"
    fi
}

# ------------------------------------------------------------------- cam ----

stage_cam() {
    head_ "Camera"

    if command -v rpicam-hello >/dev/null 2>&1; then
        rpicam-hello --list-cameras 2>&1 | sed 's/^/  · /' | head -12
    elif command -v libcamera-hello >/dev/null 2>&1; then
        libcamera-hello --list-cameras 2>&1 | sed 's/^/  · /' | head -12
    fi
    ls /dev/video* >/dev/null 2>&1 && info "v4l2 devices: $(ls /dev/video* | tr '\n' ' ')"

    # Exercise the exact path pi_sensor.py takes: Picamera2 first, V4L2 second,
    # at the resolution the sensor actually requests.
    python3 - <<'PY'
W, H = 640, 480
try:
    from picamera2 import Picamera2
    p = Picamera2()
    p.configure(p.create_preview_configuration(main={"size": (W, H), "format": "RGB888"}))
    p.start()
    import time; time.sleep(1.5)
    a = p.capture_array()
    p.stop(); p.close()
    print("  \033[32m✓\033[0m Picamera2 capture %s" % (a.shape,))
    raise SystemExit(0)
except SystemExit:
    raise
except Exception as e:
    print("  · Picamera2 unavailable: %s" % e)

import cv2
c = cv2.VideoCapture(0)
c.set(cv2.CAP_PROP_FRAME_WIDTH, W); c.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
ok, f = c.read(); c.release()
print(("  \033[32m✓\033[0m V4L2 capture %s" % (f.shape,)) if ok
      else "  \033[31m✗\033[0m NO CAMERA — both Picamera2 and V4L2 failed")
raise SystemExit(0 if ok else 1)
PY
}

# ------------------------------------------------------------------- eth ----

stage_eth() {
    head_ "Ethernet — the administration path"

    # sshd must survive a reboot. On Raspberry Pi OS it can be present but
    # disabled, in which case the board comes back after `sudo reboot` with no
    # way in except a monitor and keyboard.
    if systemctl is-enabled ssh >/dev/null 2>&1; then
        ok "ssh enabled at boot"
    else
        sudo systemctl enable --now ssh && ok "ssh enabled at boot"
    fi
    systemctl is-active ssh >/dev/null 2>&1 && ok "ssh running" || bad "ssh NOT running"

    # The wired profile must come up on its own, or a reboot strands the board.
    local wired
    wired=$(nmcli -t -f NAME,TYPE connection show 2>/dev/null \
            | awk -F: '$2=="802-3-ethernet"{print $1; exit}')
    if [ -n "$wired" ]; then
        sudo nmcli connection modify "$wired" connection.autoconnect yes
        ok "wired profile '$wired' autoconnects"
    else
        bad "no ethernet profile found"
    fi

    # Wired must win the default route. Wi-Fi and ethernet are different
    # devices so both stay active at once; what matters is only which one
    # carries the default route, and NanoNet must never be it.
    local dev
    dev=$(ip route show default 2>/dev/null | awk '{print $5; exit}')
    [ -n "$dev" ] && info "default route via $dev" || bad "no default route"
    ip -4 addr show eth0 2>/dev/null | grep -o 'inet [0-9.]*' | sed 's/^/  · /'
}

# ------------------------------------------------------------------- net ----

stage_net() {
    head_ "Joining $SSID as $PI_ADDR"

    command -v nmcli >/dev/null 2>&1 || { bad "nmcli not found"; return 1; }

    info "ethernet stays the default route — internet and ssh are preserved"
    sudo nmcli connection delete "$SSID" 2>/dev/null

    # NO ipv4.gateway, and never-default on purpose.
    #
    # pi_setup.sh sets ipv4.gateway 192.168.50.1, which installs a default
    # route over wlan0. NanoNet has no way out, so that silently kills the
    # Pi's internet AND the wired ssh path the rig is administered over —
    # the exact trap chrony_setup.sh warns about when apt cannot reach the
    # archives.
    #
    # Nothing needs that gateway. Every peer the Pi talks to — the master at
    # .1, Nano 2's chrony at .55 — is inside 192.168.50.0/24, reachable via
    # the connected-subnet route that the static address brings with it.
    # never-default tells NetworkManager not to install a default route from
    # this connection even if the AP offers one via DHCP.
    #
    # autoconnect yes + high priority is correct ON THE PI. The `autoconnect
    # no` rule in BUILD_GUIDE applies only to the Nanos, where it stops two
    # boards racing to own the AP. A Pi is always a client and must come back
    # on its own after a master failover or a reboot.
    sudo nmcli connection add type wifi ifname wlan0 con-name "$SSID" \
        ssid "$SSID" autoconnect yes \
        connection.autoconnect-priority "$AUTOCONNECT_PRIORITY" \
        connection.autoconnect-retries 0 \
        ipv4.method manual ipv4.addresses "$PI_ADDR/24" \
        ipv4.never-default yes ipv6.method disabled \
        wifi-sec.key-mgmt wpa-psk wifi-sec.psk "$PSK" || {
            bad "could not create the profile"; return 1; }

    ok "profile created: $PI_ADDR/24, autoconnect yes, priority $AUTOCONNECT_PRIORITY, no default route"

    # Creating the profile and activating it are separate problems. The profile
    # can be written any time; it can only come UP once a Nano is holding the
    # AP. With autoconnect yes, NetworkManager joins on its own the moment the
    # SSID appears, so a missing AP now is not a failure — the board is armed.
    if nmcli device wifi list 2>/dev/null | grep -q "$SSID"; then
        sudo nmcli connection up "$SSID" || { bad "could not activate"; return 1; }
        sudo systemctl restart chrony 2>/dev/null
        ok "activated"
    else
        info "$SSID not on the air yet — profile will autoconnect when a Nano raises it"
        info "chrony stays unsynced until then, which is expected"
        return 0
    fi

    # Prove the wired path survived, since this is the stage that historically
    # broke it.
    local dev
    dev=$(ip route show default 2>/dev/null | awk '{print $5; exit}')
    if [ "$dev" = "wlan0" ]; then
        bad "default route went to wlan0 — internet and wired ssh are at risk"
    elif [ -n "$dev" ]; then
        ok "default route still via $dev"
    else
        bad "no default route at all — is ethernet plugged in?"
    fi
    ping -c1 -W3 "$GATEWAY" >/dev/null 2>&1 \
        && ok "master reachable at $GATEWAY over wlan0" \
        || info "master not answering yet at $GATEWAY"
}

# ----------------------------------------------------------------- check ----

stage_check() {
    head_ "Checking"

    [ -f "$SENSOR_DIR/pi_sensor.py" ] \
        && ok "pi_sensor.py present" \
        || bad "pi_sensor.py MISSING from $SENSOR_DIR"

    # The current pi_sensor.py takes the id from the environment and refuses to
    # be edited per board. BUILD_GUIDE Phase C still says to edit SENSOR_ID by
    # hand; that instruction is stale. A hardcoded "pi2" here would be the
    # warning sign, not the goal.
    if grep -q 'PI_SENSOR_ID' "$SENSOR_DIR/pi_sensor.py" 2>/dev/null; then
        ok "pi_sensor.py reads PI_SENSOR_ID from the environment"
    else
        bad "this pi_sensor.py predates PI_SENSOR_ID — you have a stale copy"
    fi

    printf '  · id at launch: %s\n' "${PI_SENSOR_ID:-pi1 (DEFAULT — must be pi2 on this board)}"

    local addr
    addr=$(ip -4 addr show wlan0 2>/dev/null | grep -o '192\.168\.50\.[0-9]*' | head -1)
    if [ "$addr" = "$PI_ADDR" ]; then ok "wlan0 is $addr"
    elif [ -n "$addr" ]; then bad "wlan0 is $addr — expected $PI_ADDR (that is Pi 1's address)"
    else bad "wlan0 has no swarm address"; fi

    nmcli -t -f NAME,AUTOCONNECT,AUTOCONNECT-PRIORITY connection show 2>/dev/null \
        | grep "^$SSID" | sed 's/^/  · /'

    ping -c2 -W3 "$GATEWAY" >/dev/null 2>&1 \
        && ok "master reachable at $GATEWAY" \
        || bad "cannot reach $GATEWAY — is a Nano holding the access point?"

    # The administration path. If wlan0 ever takes the default route, the wired
    # ssh session and the Pi's internet both go with it.
    local dev
    dev=$(ip route show default 2>/dev/null | awk '{print $5; exit}')
    case "$dev" in
        wlan0) bad "default route via wlan0 — this breaks internet and wired ssh" ;;
        "")    bad "no default route" ;;
        *)     ok "default route via $dev (ethernet intact)" ;;
    esac
    systemctl is-active ssh >/dev/null 2>&1 && ok "sshd running" || bad "sshd NOT running"

    date +"  · clock: %Y-%m-%d %H:%M:%S %Z"
    if command -v chronyc >/dev/null 2>&1; then
        chronyc tracking 2>/dev/null | grep -E "Reference ID|Stratum|System time" | sed 's/^/  · /'
    else
        bad "chrony not installed"
    fi

    printf '\nWhen all of the above is green:\n'
    printf '  cd %s && PI_SENSOR_ID=pi2 timeout 500 python3 pi_sensor.py\n\n' "$SENSOR_DIR"
}

case "${1:-}" in
    apt)   stage_apt ;;
    deps)  stage_deps ;;
    time)  stage_time ;;
    host)  stage_host ;;
    cam)   stage_cam ;;
    eth)   stage_eth ;;
    net)   stage_net ;;
    check) stage_check ;;
    prep)  stage_apt && stage_deps && stage_time && stage_host && stage_cam && stage_eth ;;
    *)     sed -n '3,12p' "$0" | sed 's/^# \?//' ;;
esac
