"""
swarm_net.py — NETWORK LAYER

Runs on both Jetsons, starts at boot under systemd, and is the only thing in the
system that starts by itself. Three jobs:

  1. FORM THE NETWORK   scan for the SSID and either join it or build it
  2. FAILSAFE           if the access point dies, rebuild it and take over
  3. TELEMETRY          exchange health data with the other Nano, continuously

It never touches the application. It publishes what it knows to a file and lets
mec_node.py read it.

    swarm_net.py  ──writes──►  swarm/neighbors.json   ──►  mec_node.py
                               role + every node's cpu/ram/gpu/ip/inference

    mec_node.py   ──writes──►  swarm/local_perf.json  ──►  swarm_net.py
                               this node's inference time and queue depth

Two plain files rather than a socket between them, so either program can be
restarted, killed, or started in either order without the other noticing.

WHY THIS LAYER IS SEPARATE
--------------------------
The network has to come up on power-on and heal itself whether or not anyone is
logged in — a network that only recovers while someone has an SSH session open is
not fault tolerance. And the application could not do this job even if you wanted
it to: reconfiguring wlan0 from inside an SSH session drops that session on the
spot.

THE RULE
--------
    boot → scan for NanoNet
            ├── found it  →  join as a station
            └── nothing   →  build it: access point + 192.168.50.1

The role is claimed, not assigned. Whoever holds 192.168.50.1 is the master;
there is no election and no separate virtual IP. A board that reboots after a
failure runs the same rule, finds the network its partner built, and joins it —
it cannot resume a role it no longer holds, because the role lives in the network
rather than in the code.

Install:
    sudo systemctl enable --now swarm_net
Watch:
    journalctl -u swarm_net -f
"""

import argparse
import csv
import glob
import json
import logging
import os
import re
import signal
import socket
import subprocess
import threading
import time

# ═══════════════════════════════════════════════════════════════════════════
#  Per-board identity — derived, so this file is identical on both Nanos
# ═══════════════════════════════════════════════════════════════════════════

# This used to be a hand-edited literal, and that made the file un-copyable:
# every `scp swarm_net.py` to both boards silently gave Nano 2 Nano 1's identity.
# It happened twice on 2026-09-22, and the second time both boards booted
# believing they were nano1, took the same 30 s claim delay, and raced — the
# exact split brain the delays exist to prevent, reintroduced by the deploy.
#
# The hostname already distinguishes the boards and nothing copies over it, so
# the identity now comes from there. SWARM_NODE_ID overrides for a board that is
# renamed or for testing.
NODE_ID_BY_HOST = {
    "admindesktop-desktop":  "nano1",
    "admindesktop1-desktop": "nano2",
}

NODE_ID = (os.environ.get("SWARM_NODE_ID")
           or NODE_ID_BY_HOST.get(socket.gethostname(), socket.gethostname()))

# Seconds to keep scanning before giving up and building the network yourself.
#
# The two boards MUST differ here, and the asymmetry is doing real work. If both
# boot together with the same delay, both scan, both find nothing, and both
# claim — two access points on one SSID, clients split between them, no way to
# reconcile. With 5 s here and 20 s on the other board, the slower one sees the
# faster one's network appear during its own scan window and joins it.
#
#   nano1: 5.0      nano2: 20.0
CLAIM_DELAY_SEC = 5.0

# Added once per position in the node ordering, so that two boards powering on
# together do not claim at the same instant.
#
# The bug this fixes: both Nanos boot, both scan, neither sees an access point
# because neither has claimed yet, both wait exactly CLAIM_DELAY_SEC, and both
# claim. Two APs answer to one SSID, the Pis split between them, and both boards
# believe they are master. The scan is not at fault and no amount of rescanning
# helps — the race is symmetric, so the tie-break has to break the symmetry.
#
# nano1 claims at 5 s, nano2 at 13 s. In those 8 s nano1's AP is broadcasting
# and nano2's next scan finds it, so nano2 joins as a station instead of
# claiming. The cost is paid only when a board is genuinely alone: if the other
# is already up, its AP is found on the first scan and the wait never runs.
#
# 8 s because the loser has to see the winner: the AP takes 3-5 s to start
# beaconing and a scan cycle is SCAN_SETTLE_SEC + POLL_INTERVAL_SEC on top.
CLAIM_STAGGER_SEC = 8.0


# Named boards claim on this schedule at cold boot, overriding the derived
# stagger below. Nano 2 is the intended master — every measurement run in the
# campaign has it in that role — so it claims first and Nano 1 waits long enough
# that it cannot possibly win the race.
#
# 25 s of margin is far more than the mechanism needs. It is set that wide on
# purpose: this is the demo configuration, and the cost of Nano 1 waiting a few
# extra seconds when it is genuinely alone is nothing against the cost of the
# wrong board holding 192.168.50.1 in front of an audience.
CLAIM_DELAY_BY_NODE = {
    "nano2": 5.0,
    "nano1": 30.0,
}

# Used when a board that already held a role comes back here — see acquire().
RECLAIM_DELAY_SEC = 5.0


def claim_delay_for(node_id):
    """Cold-boot claim delay for this board.

    The explicit table wins where it names the board. Anything else falls back
    to the derived stagger: the trailing number in the id is the board's
    position, so nano3 -> 2 stagger periods. Deterministic either way, needing
    no leader election and no shared state — which matters because this runs
    before any network exists.
    """
    if node_id in CLAIM_DELAY_BY_NODE:
        return CLAIM_DELAY_BY_NODE[node_id]
    match = re.search(r"(\d+)$", node_id or "")
    rank = max(0, int(match.group(1)) - 1) if match else 0
    return CLAIM_DELAY_SEC + CLAIM_STAGGER_SEC * rank

# ═══════════════════════════════════════════════════════════════════════════
#  Configuration — identical on both boards
# ═══════════════════════════════════════════════════════════════════════════

SSID = "NanoNet"
AP_PROFILE = "NanoNet-ap"           # NetworkManager profile: AP + 192.168.50.1
CLIENT_PROFILE = "NanoNet-client"   # NetworkManager profile: station, static IP
WIFI_IFACE = "wlan0"

MASTER_ADDRESS = "192.168.50.1"     # The AP's own address. Holding it = master.
BROADCAST_ADDRESS = "192.168.50.255"
TELEMETRY_PORT = 5500               # UDP, Nano ↔ Nano

TELEMETRY_INTERVAL_SEC = 0.5        # How often health is sampled and sent
PEER_TIMEOUT_SEC = 3.0              # No telemetry for this long → peer is gone
# Resolved from __file__, not the working directory, so this agrees with
# mec_node.py no matter where either was started from. systemd runs this one from
# its own WorkingDirectory; you run mec_node.py from wherever your shell is. A
# relative path would put the handoff files in two different places and each side
# would wait forever for the other.
STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runtime")
NEIGHBORS_FILE = os.path.join(STATE_DIR, "neighbors.json")
LOCAL_PERF_FILE = os.path.join(STATE_DIR, "local_perf.json")
LOCAL_PERF_MAX_AGE_SEC = 5.0        # Older than this → the MEC layer is not running

POLL_INTERVAL_SEC = 1.0

# How long to let a requested scan run before reading the result. A NetworkManager
# wifi scan takes roughly 1-3 s, and the read has to follow the scan rather than
# race it — see WifiBackend.scan for why the two cannot be one command here.
# Two seconds still leaves room for two full scan cycles inside CLAIM_DELAY_SEC.
SCAN_SETTLE_SEC = 2.0
ASSOCIATION_FAIL_THRESHOLD = 3      # Consecutive failed polls before declaring loss
# Seconds an access point may sit with no stations before it gives up the claim.
# 0 disables it, which is the default and deliberate.
#
# The intent was to recover from claiming while another AP already existed: the
# one nobody joined should stand down. In practice it does the opposite of what
# is wanted. A board that comes up before the Pis are powered has no stations
# through no fault of its own, and at 60 s it tears down the only network on the
# air — so the Pi boots into nothing, and the board that could have served it is
# busy rescanning. That is the "it stops being an access point" failure.
#
# It is also now redundant. The duplicate-claim case it guarded against is what
# CLAIM_STAGGER_SEC prevents: two boards can no longer claim at the same instant,
# so the second one sees the first's AP and joins it. Guarding a race that cannot
# happen, at the cost of dropping a healthy network, is a bad trade.
#
# Set it to 60 to restore the old behaviour if a split brain is ever observed
# despite the stagger.
EMPTY_NETWORK_GRACE_SEC = 0.0

STATS_INTERVAL_SEC = 15.0

MASTER = "master"
WORKER = "worker"

logging.basicConfig(
    level=logging.DEBUG if os.environ.get("MEC_VERBOSE") else logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-9s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("swarm-net")


# ═══════════════════════════════════════════════════════════════════════════
#  Telemetry readers
# ═══════════════════════════════════════════════════════════════════════════

_prev_cpu_sample = None


def read_cpu_percent():
    """CPU utilisation since the last call, or None if unavailable.

    /proc/stat rather than psutil, which is not installed by default on L4T.
    Utilisation is a rate, so it needs two samples: the first call returns None
    and every call after covers the interval since the previous one. Called on
    the telemetry tick, so it is a rolling half-second window.
    """
    global _prev_cpu_sample
    try:
        with open("/proc/stat", "r") as f:
            fields = f.readline().split()
        values = [int(x) for x in fields[1:]]
        idle = values[3] + (values[4] if len(values) > 4 else 0)   # idle + iowait
        total = sum(values)
    except (IOError, ValueError, IndexError):
        return None

    previous, _prev_cpu_sample = _prev_cpu_sample, (idle, total)
    if previous is None:
        return None
    idle_delta, total_delta = idle - previous[0], total - previous[1]
    if total_delta <= 0:
        return None
    return round(100.0 * (1.0 - idle_delta / total_delta), 1)


def read_ram_percent():
    """Memory in use as a percentage, or None if unavailable.

    MemAvailable rather than MemFree — page cache is reclaimable, and counting it
    as used would report a healthy node as nearly full.
    """
    try:
        info = {}
        with open("/proc/meminfo", "r") as f:
            for line in f:
                key, _, value = line.partition(":")
                info[key] = int(value.split()[0])
    except (IOError, ValueError, IndexError):
        return None
    total = info.get("MemTotal", 0)
    available = info.get("MemAvailable", info.get("MemFree", 0))
    if total <= 0:
        return None
    return round(100.0 * (1.0 - available / total), 1)


def read_gpu_temp():
    """GPU temperature in °C, or 0.0 if no GPU thermal zone is readable.

    The zone is found by reading each `type` file rather than hardcoding an
    index, which is not stable across L4T releases. 0.0 means "unknown", and the
    scheduler treats that as "apply no thermal penalty" rather than as freezing.
    """
    for path in glob.glob("/sys/devices/virtual/thermal/thermal_zone*"):
        type_path, temp_path = os.path.join(path, "type"), os.path.join(path, "temp")
        if not (os.path.exists(type_path) and os.path.exists(temp_path)):
            continue
        try:
            with open(type_path, "r") as f:
                if "GPU" not in f.read().strip().upper():
                    continue
            with open(temp_path, "r") as f:
                return int(f.read().strip()) / 1000.0
        except (IOError, ValueError):
            continue
    return 0.0


def read_battery_pct():
    """Battery charge as a percentage, or None if there is no battery.

    A bench Jetson on a wall adapter has no power-supply node and gets None,
    which the scheduler correctly reads as "no penalty". On a UAV with a smart
    battery exposed through sysfs this picks it up with no other change.
    """
    for path in glob.glob("/sys/class/power_supply/*/capacity"):
        try:
            with open(path, "r") as f:
                pct = float(f.read().strip())
            if 0.0 <= pct <= 100.0:
                return pct
        except (IOError, ValueError):
            continue
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  nmcli
# ═══════════════════════════════════════════════════════════════════════════

class NetworkBackend:
    """Every OS interaction in this file goes through here.

    nmcli for everything rather than a mix of nmcli and iw, because
    NetworkManager owns these interfaces and fighting it with raw iw commands
    produces states it silently reverts. The one exception is peer_count, which
    needs an association count nmcli does not expose.
    """

    def __init__(self, iface=WIFI_IFACE, timeout_sec=20.0):
        self.iface = iface
        self.timeout = timeout_sec

    def _run(self, *args):
        try:
            proc = subprocess.run(["nmcli"] + list(args),
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=self.timeout)
            return proc.returncode == 0, proc.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False, ""

    def scan(self):
        """SSIDs currently visible. Requests a scan, waits for it, then reads.

        Three commands where one would do, because the one does not work here.
        `nmcli device wifi list --rescan yes` is the obvious form, but `--rescan`
        arrived in NetworkManager 1.12 and the Jetsons run 1.10: nmcli prints
        "Unknown parameter" to stderr, lists from its cache anyway, and exits
        zero. The scan reports success on data that can be tens of seconds stale
        — long enough to miss an access point that came up during our own claim
        delay, which is exactly the case the boot-race tie-break depends on
        catching. The signature in the journal is "scan started" and "network
        found" logged in the same second, which no real scan achieves.

        `device wifi rescan` exists in both versions. Its result is deliberately
        ignored: NetworkManager refuses a rescan issued too soon after the last
        one, and that refusal is not a failure — it means the cache we are about
        to read was refreshed moments ago by the scan it is complaining about.
        """
        self._run("device", "wifi", "rescan", "ifname", self.iface)
        time.sleep(SCAN_SETTLE_SEC)
        ok, out = self._run("-t", "-f", "SSID", "device", "wifi", "list")
        return {l.strip() for l in out.splitlines() if l.strip()} if ok else set()

    def activate(self, profile):
        ok, _ = self._run("connection", "up", profile)
        return ok

    def deactivate(self, profile):
        self._run("connection", "down", profile)
        return True

    def associated(self):
        """True if the interface has a working layer-2 association.

        This is the failure signal. When the access point dies the interface
        disassociates within a second or two — an unambiguous OS-level event that
        needs no application heartbeat and arrives well before a request would
        time out.
        """
        ok, out = self._run("-t", "-f", "DEVICE,STATE", "device", "status")
        if not ok:
            return False
        for line in out.splitlines():
            parts = line.split(":")
            if len(parts) >= 2 and parts[0] == self.iface:
                return parts[1] == "connected"
        return False

    def addresses(self):
        """IPv4 addresses on this interface. Holding MASTER_ADDRESS means master."""
        ok, out = self._run("-t", "-f", "IP4.ADDRESS", "device", "show", self.iface)
        if not ok:
            return set()
        found = set()
        for line in out.splitlines():
            if ":" in line:                       # "IP4.ADDRESS[1]:192.168.50.1/24"
                value = line.split(":", 1)[1].strip()
                if value:
                    found.add(value.split("/")[0])
        return found

    def own_ip(self):
        for addr in self.addresses():
            return addr
        return None

    def peer_count(self):
        """Stations associated with our access point.

        Counts layer-2 associations rather than asking the application, because
        the application is started by hand and may not be running yet. An empty
        network and an unstarted program are different problems, and only the
        first warrants tearing the access point down.
        """
        try:
            proc = subprocess.run(["iw", "dev", self.iface, "station", "dump"],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=self.timeout)
            if proc.returncode == 0:
                return proc.stdout.count("Station ")
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        return 0


# ═══════════════════════════════════════════════════════════════════════════
#  Telemetry exchange
# ═══════════════════════════════════════════════════════════════════════════

class TelemetryExchange:
    """Broadcasts this node's health and collects the other Nano's.

    UDP broadcast rather than a connection to a known peer, because who is the
    master changes and a broadcast needs no peer list to maintain. It also works
    identically whichever role this node currently holds.

    Best-effort by design: a lost datagram costs one sample out of two per
    second, and the next one carries the same news. Anything that needed
    guaranteed delivery would not belong at this layer.
    """

    def __init__(self, node_id, backend):
        self.node_id = node_id
        self.backend = backend
        self._lock = threading.Lock()
        self._peers = {}                # node_id -> last telemetry dict
        self._own = {}
        self._stop = threading.Event()
        self._rx = None
        self._tx = None

    # -- local readings ------------------------------------------------

    def _read_local_perf(self):
        """Pick up inference stats the MEC layer left for us.

        Inference time is application knowledge, not network knowledge, so
        mec_node.py writes it to a file and this layer folds it into the
        broadcast. Also tells us whether the MEC layer is running at all — the
        master uses that to decide a peer is actually available for work, rather
        than merely present on the network.
        """
        try:
            with open(LOCAL_PERF_FILE, "r") as f:
                perf = json.load(f)
        except (IOError, ValueError):
            return {}
        if time.time() - perf.get("updated", 0) > LOCAL_PERF_MAX_AGE_SEC:
            return {}                   # Stale: mec_node.py is not running
        return {
            "infer_ms": perf.get("infer_ms"),
            "queue": perf.get("queue", 0),
            "mec_active": True,
            "mec_role": perf.get("role"),
        }

    def sample(self, role):
        """Take one reading of this node's health."""
        payload = {
            "node_id": self.node_id,
            "role": role,
            "ip": self.backend.own_ip(),
            "cpu_pct": read_cpu_percent(),
            "ram_pct": read_ram_percent(),
            "gpu_temp_c": read_gpu_temp(),
            "battery_pct": read_battery_pct(),
            "infer_ms": None,
            "queue": 0,
            "mec_active": False,
            "mec_role": None,
            "sent_at": time.time(),
        }
        payload.update(self._read_local_perf())
        with self._lock:
            self._own = payload
        return payload

    # -- sockets -------------------------------------------------------

    def start(self, role_fn):
        self._rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._rx.settimeout(0.5)
        self._rx.bind(("0.0.0.0", TELEMETRY_PORT))

        self._tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._tx.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        threading.Thread(target=self._receive_loop, daemon=True, name="tele-rx").start()
        threading.Thread(target=self._send_loop, args=(role_fn,),
                         daemon=True, name="tele-tx").start()
        log.info("Telemetry exchange up on UDP %d", TELEMETRY_PORT)

    def _send_loop(self, role_fn):
        while not self._stop.is_set():
            try:
                payload = self.sample(role_fn())
                self._tx.sendto(json.dumps(payload).encode(),
                                (BROADCAST_ADDRESS, TELEMETRY_PORT))
            except OSError as e:
                # Normal during a failover: the interface is mid-reconfiguration
                # and has no route yet. The next tick will get through.
                log.debug("Telemetry send skipped: %s", e)
            except Exception as e:
                log.debug("Telemetry send failed: %s", e)
            self._stop.wait(TELEMETRY_INTERVAL_SEC)

    def _receive_loop(self):
        while not self._stop.is_set():
            try:
                data, _addr = self._rx.recvfrom(8192)
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    break
                time.sleep(0.2)
                continue
            try:
                msg = json.loads(data.decode())
            except (ValueError, UnicodeDecodeError):
                continue

            peer = msg.get("node_id")
            if not peer or peer == self.node_id:
                continue                # Our own broadcast, coming back to us
            msg["last_seen"] = time.time()
            with self._lock:
                if peer not in self._peers:
                    log.info("Peer '%s' appeared at %s", peer, msg.get("ip"))
                self._peers[peer] = msg

    # -- reading -------------------------------------------------------

    def table(self):
        """The neighbour table: this node plus every peer heard from recently."""
        now = time.time()
        with self._lock:
            own = dict(self._own)
            peers = {k: dict(v) for k, v in self._peers.items()}

        table = {}
        if own:
            own["last_seen"] = now
            own["online"] = True
            table[self.node_id] = own
        for node_id, entry in peers.items():
            entry["online"] = (now - entry.get("last_seen", 0)) <= PEER_TIMEOUT_SEC
            table[node_id] = entry
        return table

    def expire(self):
        """Log peers that have gone quiet. Entries are kept, marked offline."""
        now = time.time()
        with self._lock:
            for node_id, entry in self._peers.items():
                stale = (now - entry.get("last_seen", 0)) > PEER_TIMEOUT_SEC
                if stale and not entry.get("_warned"):
                    log.warning("Peer '%s' went quiet", node_id)
                    entry["_warned"] = True
                elif not stale and entry.get("_warned"):
                    log.info("Peer '%s' is back", node_id)
                    entry["_warned"] = False

    def stop(self):
        self._stop.set()
        for sock in (self._rx, self._tx):
            try:
                if sock:
                    sock.close()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════
#  State file
# ═══════════════════════════════════════════════════════════════════════════

def publish_state(role, node_id, table):
    """Write neighbors.json for the MEC layer.

    Written to a temporary file and renamed, because rename is atomic on POSIX:
    mec_node.py polls this file and must never catch it half-written.
    """
    os.makedirs(STATE_DIR, exist_ok=True)
    state = {
        "role": role,
        "node_id": node_id,
        "master_address": MASTER_ADDRESS,
        "updated": time.time(),
        "nodes": table,
    }
    tmp = NEIGHBORS_FILE + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, NEIGHBORS_FILE)
    except (IOError, OSError) as e:
        log.error("Could not publish %s: %s", NEIGHBORS_FILE, e)


class RoleEventLog:
    """Appends every network transition to swarm/role_events_NNN.csv.

    The audit trail for a failover drill. Read it alongside the ground station's
    log rather than on its own: the two Nanos have no shared clock — neither
    board has a battery-backed RTC and the network is isolated — so timestamps
    from different nodes cannot be compared directly.
    """

    def __init__(self, node_id):
        os.makedirs(STATE_DIR, exist_ok=True)
        serial = len(glob.glob(f"{STATE_DIR}/role_events_*.csv")) + 1
        self.path = f"{STATE_DIR}/role_events_{serial:03d}.csv"
        self._file = open(self.path, "w", newline="")
        self._csv = csv.writer(self._file)
        self._csv.writerow(["Wall Clock", "Epoch", "Elapsed (s)", "Node ID", "Event", "Detail"])
        self._file.flush()
        self._node_id = node_id
        self._t0 = time.time()
        log.info("Role events → %s", self.path)

    def write(self, event, detail=""):
        now = time.time()
        self._csv.writerow([time.strftime("%H:%M:%S", time.localtime(now)),
                            round(now, 3), round(now - self._t0, 2),
                            self._node_id, event, detail])
        self._file.flush()

    def close(self):
        try:
            self._file.close()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
#  Role state machine
# ═══════════════════════════════════════════════════════════════════════════

class RoleManager:
    """Claims or joins the network, and reports when the role can no longer be held."""

    def __init__(self, node_id, claim_delay, backend, on_event=None):
        self.node_id = node_id
        self.claim_delay = claim_delay
        self.backend = backend
        self._on_event = on_event
        self.role = None

    def _event(self, event, detail=""):
        log.info("[role] %s%s", event, f" — {detail}" if detail else "")
        if self._on_event:
            try:
                self._on_event(event, detail)
            except Exception:
                pass            # Never let instrumentation break the state machine

    def acquire(self, stop_event):
        """Scan, then join the existing network or claim it. Blocks."""
        # Before the scan, not after. A node that has just yielded the master
        # role still has the access point profile up, so a scan taken now finds
        # its own SSID: it "discovers" the network, calls _join(), and _join()'s
        # first act is to tear down the very access point it is trying to join.
        # It then cannot associate with a network that no longer exists, returns
        # here, and repeats — a self-sustaining loop with a period of about 22 s.
        # Dropping the profile first means the scan sees only what other boards
        # are actually broadcasting.
        #
        # A no-op on the normal boot path, where the profile was never up.
        self.backend.deactivate(AP_PROFILE)

        # The long cold-boot delay exists to lose a race against the other board.
        # Coming back here after holding a role is not that situation: the
        # network demonstrably existed a moment ago and has just gone, so there
        # is no race to lose and nothing to gain by waiting out the full stagger.
        # Failover would otherwise take as long as the boot delay — thirty
        # seconds of dead air on Nano 1 every time the master drops.
        delay = self.claim_delay if self.role is None else RECLAIM_DELAY_SEC

        self._event("scan started",
                    f"looking for '{SSID}', will claim after {delay:.0f}s")

        deadline = time.time() + delay
        while time.time() < deadline:
            if stop_event.is_set():
                return None
            if SSID in self.backend.scan():
                self._event("network found", f"'{SSID}' is up — joining as a station")
                return self._join()
            remaining = deadline - time.time()
            if remaining > 0:
                time.sleep(min(POLL_INTERVAL_SEC, remaining))

        self._event("no network found",
                    f"nothing after {delay:.0f}s — claiming the access point")
        return self._claim()

    def _join(self):
        self.backend.deactivate(AP_PROFILE)
        if not self.backend.activate(CLIENT_PROFILE):
            log.error("Could not activate '%s' — will rescan", CLIENT_PROFILE)
            return None
        self.role = WORKER
        self._event("role worker", "joined the network as a station")
        return WORKER

    def _claim(self):
        self.backend.deactivate(CLIENT_PROFILE)
        if not self.backend.activate(AP_PROFILE):
            log.error("Could not activate '%s' — will rescan", AP_PROFILE)
            return None
        self.role = MASTER
        self._event("role master", f"access point up, holding {MASTER_ADDRESS}")
        return MASTER

    def hold(self, role, stop_event, tick=None):
        """Block until *role* can no longer be held. Returns the reason.

        For a station, the role ends when association is lost — the access point
        died and there is a network to rebuild.

        For the access point holder, it ends only if nobody ever joins. A master
        alone on a network that should have a sensor and a worker on it has
        almost certainly claimed in error while the real network exists
        elsewhere; yielding and rescanning resolves that without intervention.

        Note the deliberate omission: the master does NOT step down because a
        worker went quiet. That is a worker problem, and tearing down a working
        access point over it would take the sensors offline too.
        """
        misses = 0
        claimed_at = time.time()
        seen_anyone = False

        while not stop_event.is_set():
            if tick:
                tick()

            if role == WORKER:
                if self.backend.associated():
                    misses = 0
                else:
                    misses += 1
                    if misses >= ASSOCIATION_FAIL_THRESHOLD:
                        self._event("association lost",
                                    f"no link for {misses}s — the access point is gone")
                        return "association lost"
            else:
                if self.backend.peer_count() > 0:
                    seen_anyone = True
                elif (EMPTY_NETWORK_GRACE_SEC > 0 and not seen_anyone
                      and time.time() - claimed_at > EMPTY_NETWORK_GRACE_SEC):
                    self._event("yielding access point",
                                f"nobody joined in {EMPTY_NETWORK_GRACE_SEC:.0f}s — "
                                "suspecting a bad claim, rescanning")
                    return "empty network"

            stop_event.wait(POLL_INTERVAL_SEC)
        return "stopped"


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="UAV swarm network layer")
    parser.add_argument("--no-telemetry", action="store_true",
                        help="network role only, skip the telemetry exchange")
    args = parser.parse_args()

    stop = threading.Event()

    def handle_signal(signum, _frame):
        log.info("Signal %d received — stopping", signum)
        stop.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    os.makedirs(STATE_DIR, exist_ok=True)
    events = RoleEventLog(NODE_ID)
    backend = NetworkBackend()
    claim_delay = claim_delay_for(NODE_ID)
    manager = RoleManager(NODE_ID, claim_delay, backend, on_event=events.write)

    telemetry = None
    if not args.no_telemetry:
        telemetry = TelemetryExchange(NODE_ID, backend)
        telemetry.start(role_fn=lambda: manager.role or "unknown")

    log.info("Network layer starting — node=%s ssid=%s cold-boot claim delay=%.0fs "
             "(re-claim after a lost role: %.0fs)",
             NODE_ID, SSID, claim_delay, RECLAIM_DELAY_SEC)

    last_report = [time.time()]

    def tick():
        """Runs once a second while a role is held: publish state, log health."""
        table = telemetry.table() if telemetry else {}
        publish_state(manager.role, NODE_ID, table)
        if telemetry:
            telemetry.expire()

        if time.time() - last_report[0] >= STATS_INTERVAL_SEC:
            last_report[0] = time.time()
            rows = []
            for node_id, entry in sorted(table.items()):
                bits = [node_id if entry.get("online") else f"{node_id} (down)"]
                if entry.get("gpu_temp_c"):
                    bits.append("%.0fC" % entry["gpu_temp_c"])
                if entry.get("cpu_pct") is not None:
                    bits.append("cpu %.0f%%" % entry["cpu_pct"])
                if entry.get("ram_pct") is not None:
                    bits.append("ram %.0f%%" % entry["ram_pct"])
                if entry.get("infer_ms") is not None:
                    bits.append("infer %.0fms" % entry["infer_ms"])
                if not entry.get("mec_active"):
                    bits.append("(mec idle)")
                rows.append(" ".join(bits))
            log.info("%s | %s", manager.role or "no role", " · ".join(rows) or "no nodes")

    try:
        while not stop.is_set():
            role = manager.acquire(stop)
            if role is None:
                # Told to stop, or nmcli failed. Pause so a persistent problem
                # does not become a hot loop, then reconsider from scratch.
                if not stop.is_set():
                    stop.wait(2.0)
                continue

            if role == MASTER:
                log.info("This node built the network and holds %s", MASTER_ADDRESS)
            else:
                log.info("This node joined the network as a station")
            publish_state(role, NODE_ID, telemetry.table() if telemetry else {})

            reason = manager.hold(role, stop, tick=tick)
            log.info("Network role '%s' ended — %s", role, reason)
            if reason != "stopped":
                events.write("rescanning", f"after: {reason}")

    finally:
        if telemetry:
            telemetry.stop()
        events.close()
        # Deliberately NOT tearing the network down. A service restart, or
        # someone stopping this to look at something, must not disconnect the
        # sensors — and if this node holds the access point, dropping it would
        # take every other device in the swarm offline with it.
        log.info("Network layer stopped — leaving the interface as it is")


if __name__ == "__main__":
    main()
