#!/bin/bash
# Pi sensor node setup — run at the Pi's console, in three stages.
#
#   ./pi_setup.sh deps    install python deps      NEEDS INTERNET
#   ./pi_setup.sh net     join NanoNet             KILLS INTERNET
#   ./pi_setup.sh check   verify everything
#
# Order matters: deps first while the Pi still has its normal network, then
# net to move it onto the swarm. Safe to re-run any stage.

SSID="NanoNet"
PSK="nanonet123"
PI_ADDR="192.168.50.11"          # .12 for Pi 2
GATEWAY="192.168.50.1"
SENSOR_DIR="$HOME/camera_node"

ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$1"; }
info() { printf '  · %s\n' "$1"; }
head_() { printf '\n\033[1m%s\033[0m\n' "$1"; }

stage_deps() {
    head_ "Installing dependencies"
    mkdir -p "$SENSOR_DIR" && ok "$SENSOR_DIR ready"

    if ! ping -c1 -W3 8.8.8.8 >/dev/null 2>&1; then
        bad "no internet — connect the Pi to your normal network first"
        info "this stage needs to reach the apt archives"
        return 1
    fi

    sudo apt update -qq
    sudo apt install -y python3-zmq python3-opencv chrony

    python3 - <<'PY'
import sys
for mod in ("cv2", "zmq"):
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
    print("  · no picamera2 — will use /dev/video via OpenCV")
PY

    # chrony is no longer load-bearing: every latency figure is now composed
    # from single-clock terms. It is here so log timestamps across the Pi, the
    # Nano and the laptop line up when reading them side by side.
    #
    # The reference is Nano 2. Two server lines for one board is not a mistake:
    # Nano 2 answers on .55 while it is a worker and on .1 once it claims the
    # AP, and the Pi has no way to know which is true right now. When Nano 1
    # holds .1 instead, it advertises stratum 10 and loses to Nano 2's 8.
    if [ -x "$(dirname "$0")/chrony_setup.sh" ]; then
        "$(dirname "$0")/chrony_setup.sh" pi
    elif [ -f /etc/chrony/chrony.conf ]; then
        if ! grep -q "192.168.50.55" /etc/chrony/chrony.conf; then
            sudo sed -i '\#^server 192.168.50.23#d' /etc/chrony/chrony.conf
            sudo sh -c 'printf "\nserver 192.168.50.55 iburst prefer trust minpoll 0 maxpoll 2\nserver 192.168.50.1 iburst minpoll 0 maxpoll 2\nmakestep 0.01 -1\n" >> /etc/chrony/chrony.conf'
            sudo systemctl restart chrony 2>/dev/null
            ok "chrony pointed at Nano 2"
        else
            ok "chrony already configured"
        fi
    fi
}

stage_net() {
    head_ "Joining $SSID as $PI_ADDR"

    if ! command -v nmcli >/dev/null 2>&1; then
        bad "nmcli not found — this OS predates NetworkManager"
        info "run:  cat /etc/os-release   and report back"
        return 1
    fi

    info "this replaces the Pi's network — internet goes away"
    sudo nmcli connection delete "$SSID" 2>/dev/null
    sudo nmcli connection add type wifi ifname wlan0 con-name "$SSID" \
        ssid "$SSID" autoconnect yes \
        ipv4.method manual ipv4.addresses "$PI_ADDR/24" ipv4.gateway "$GATEWAY" \
        wifi-sec.key-mgmt wpa-psk wifi-sec.psk "$PSK" || {
            bad "could not create the profile"; return 1; }

    sudo nmcli connection up "$SSID" || {
        bad "could not activate — is $SSID up? a Nano must hold the AP first"
        info "check:  nmcli device wifi list | grep $SSID"
        return 1; }

    sudo systemctl restart chrony 2>/dev/null
    ok "profile up"
}

stage_check() {
    head_ "Checking"

    [ -f "$SENSOR_DIR/pi_sensor.py" ] \
        && ok "pi_sensor.py present" \
        || bad "pi_sensor.py MISSING from $SENSOR_DIR"

    grep -E '^SENSOR_ID|^MASTER_IP' "$SENSOR_DIR/pi_sensor.py" 2>/dev/null \
        | sed 's/^/  · /'

    addr=$(ip -4 addr show wlan0 2>/dev/null | grep -o '192\.168\.50\.[0-9]*' | head -1)
    [ -n "$addr" ] && ok "wlan0 is $addr" || bad "wlan0 has no swarm address"

    if ping -c2 -W3 "$GATEWAY" >/dev/null 2>&1; then
        ok "master reachable at $GATEWAY"
    else
        bad "cannot reach $GATEWAY — is a Nano holding the access point?"
    fi

    if ls /dev/video* >/dev/null 2>&1; then
        ok "camera: $(ls /dev/video* | tr '\n' ' ')"
    elif python3 -c "import picamera2" 2>/dev/null; then
        ok "camera: CSI via picamera2"
    else
        bad "no camera found"
    fi

    date +"  · clock: %Y-%m-%d %H:%M:%S"
    command -v chronyc >/dev/null 2>&1 && \
        chronyc tracking 2>/dev/null | grep -E "Reference ID|System time" | sed 's/^/  · /'

    printf '\nWhen all of the above is green:\n'
    printf '  cd %s && timeout 500 python3 pi_sensor.py\n\n' "$SENSOR_DIR"
}

case "$1" in
    deps)  stage_deps ;;
    net)   stage_net ;;
    check) stage_check ;;
    *)     printf 'usage: ./pi_setup.sh {deps|net|check}\n'
           printf '  deps   install python deps   (needs internet)\n'
           printf '  net    join %s          (kills internet)\n' "$SSID"
           printf '  check  verify everything\n' ;;
esac
