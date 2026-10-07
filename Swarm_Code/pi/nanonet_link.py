"""
nanonet_link.py

NanoNet link supervisor for the Raspberry Pi sensor nodes.

Runs as a root systemd service, independently of pi_sensor.py, and does exactly
one job: keep wlan0 associated to NanoNet and keep 192.168.50.1 reachable, for
as long as the board is powered.  It never gives up and it has no terminal
state — if the access point is not on the air it keeps scanning until it is.

Why this exists as a separate process
─────────────────────────────────────
NetworkManager already autoconnects, so on paper nothing here is needed.  In
practice two things defeat it on this rig:

  1. **NM stops trying.**  `connection.autoconnect-retries` defaults to 4.  The
     Nanos now stagger their AP claim (nano2 at 5 s, nano1 at ~30 s) so that
     they cannot both become the access point at boot.  A Pi powered on at the
     same time therefore spends its first half-minute looking for an SSID that
     does not exist yet, exhausts all four attempts, and then sits idle
     *forever* — NM will not retry again without an external event.  The board
     looks healthy, the profile says `autoconnect yes`, and no frame ever
     leaves it.  `autoconnect-retries 0` (infinite) fixes the profile, and this
     daemon fixes the boards whose profile was written before that was known.

  2. **Associated is not the same as connected.**  When the master fails over,
     the Pi can stay associated to a radio that is no longer forwarding
     anything.  NM sees a connected device and does nothing.  Only an actual
     reachability probe notices, which is why this checks the master address
     and not just the interface state.

Both of those are invisible from inside pi_sensor.py, and both happen while it
is running — which is why this is a separate unit with its own lifetime rather
than a thread in the sensor.  It also means the link is already up before the
sensor starts, and stays up when the sensor is stopped for a code change.

What it deliberately does not do
────────────────────────────────
It does not touch the profile's IP address.  Pi 1 is 192.168.50.11 and Pi 2 is
192.168.50.12, that distinction lives in the saved profile, and a supervisor
that rewrote it would silently give two boards the same address.  It repairs
only the settings that are the same on every Pi.

It does not restart chrony on recovery by default.  chrony re-establishes its
own polls when the network returns, and a forced restart would step the clock
mid-run and corrupt the very latency numbers the rig exists to measure.  Set
PI_LINK_RESTART_CHRONY=1 if you want it anyway.
"""

import logging
import os
import subprocess
import sys
import time

logging.basicConfig(
    level=logging.DEBUG if os.environ.get("MEC_VERBOSE") else logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-9s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("link")


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

IFACE = os.environ.get("PI_WIFI_IFACE", "wlan0")
SSID = os.environ.get("PI_SSID", "NanoNet")
PSK = os.environ.get("PI_PSK", "nanonet123")

# The access point's gateway address — whichever Nano currently holds the AP
# answers here.  Reachability of this address is the only definition of "the
# link works" that matters to the sensor.
MASTER_IP = os.environ.get("PI_MASTER_IP", "192.168.50.1")

# How often the health check runs.  Cheap: one `nmcli device status` and, when
# associated, one ping.  Fast enough that the gap between the AP appearing and
# the Pi joining it is bounded by this rather than by NM's own timers.
CHECK_INTERVAL_SEC = float(os.environ.get("PI_LINK_INTERVAL", "2.0"))

# Consecutive failed pings tolerated while the interface still reports itself
# associated, before the link is declared dead and a reassociation is forced.
#
# This is deliberately not aggressive.  A master failover takes the Nanos a few
# seconds (RECLAIM_DELAY_SEC is 5 s in swarm_net.py) and during that window the
# master is legitimately absent; tearing down a perfectly good association in
# the middle of it would add our own recovery time on top of theirs.  Eight
# strikes at 2 s is 16 s — comfortably longer than any planned failover, and
# far shorter than an outage anyone would sit through.
UNREACHABLE_STRIKES = int(os.environ.get("PI_LINK_STRIKES", "8"))

# An activation that is genuinely in progress is not a failure, so the ladder
# below is not advanced while NM reports one.  But a device stuck in
# "connecting" is a real state on this hardware, so patience is bounded.
CONNECTING_PATIENCE_SEC = 30.0

# NetworkManager itself is restarted only as a last resort, and never twice in
# quick succession — it drops every connection on the board including the wired
# path used to administer it.
NM_RESTART_COOLDOWN_SEC = 300.0

# Highest priority of any saved profile, so NanoNet always wins the radio.
AUTOCONNECT_PRIORITY = "100"

# One heartbeat line per this interval while healthy, so that a silent log
# means the daemon died rather than that nothing is wrong.
HEARTBEAT_SEC = 300.0

RESTART_CHRONY = os.environ.get("PI_LINK_RESTART_CHRONY", "0") == "1"


# ═══════════════════════════════════════════════════════════════════════════
#  Command helpers
# ═══════════════════════════════════════════════════════════════════════════

def _run(args, timeout):
    """Run a command and return (returncode, stdout).  Never raises.

    Every call is bounded.  A hung nmcli would freeze the supervisor in exactly
    the situation it exists to handle, so a timeout is treated as a failure of
    that step rather than allowed to propagate.
    """
    try:
        p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=timeout)
        return p.returncode, p.stdout.decode("utf-8", "replace").strip()
    except subprocess.TimeoutExpired:
        log.warning("timed out after %.0fs: %s", timeout, " ".join(args))
        return 124, ""
    except OSError as e:
        log.warning("could not run %s: %s", args[0], e)
        return 127, ""


def _nm_device_state():
    """Return (state, connection_name) for IFACE, e.g. ("connected", "NanoNet").

    ("unknown", "") if the device is not listed at all — which happens when the
    driver has crashed or the interface was renamed, and is itself a fault.
    """
    rc, out = _run(["nmcli", "-t", "-f", "DEVICE,STATE,CONNECTION",
                    "device", "status"], timeout=8)
    if rc != 0:
        return "unknown", ""
    for line in out.splitlines():
        # nmcli -t escapes literal colons as "\:", and a connection name may
        # contain one.  Splitting from the left for the two fixed fields and
        # keeping the remainder intact is correct for any name.
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == IFACE:
            return parts[1], parts[2]
    return "unknown", ""


def _has_ipv4():
    """True when IFACE carries an IPv4 address.

    Associated without an address is a real state — it is what a failed DHCP or
    a half-applied profile looks like — and the sensor cannot send from it.
    """
    rc, out = _run(["ip", "-4", "-o", "addr", "show", "dev", IFACE], timeout=5)
    return rc == 0 and "inet " in out


def _master_reachable():
    """One ICMP probe at the master, with a one second ceiling."""
    rc, _ = _run(["ping", "-c", "1", "-W", "1", "-n", "-q", MASTER_IP], timeout=4)
    return rc == 0


def _ssid_on_air():
    """True when NanoNet appears in the scan list.

    Used only for logging, to distinguish "no access point exists yet" from
    "the access point is there and we cannot join it" — two faults that look
    identical from the sensor's side and need completely different attention.
    """
    rc, out = _run(["nmcli", "-t", "-f", "SSID", "device", "wifi", "list",
                    "ifname", IFACE], timeout=15)
    return rc == 0 and any(s.strip() == SSID for s in out.splitlines())


# ═══════════════════════════════════════════════════════════════════════════
#  Recovery actions
# ═══════════════════════════════════════════════════════════════════════════

def _rescan():
    """Force a fresh scan and let the results land.

    The rescan return code is ignored on purpose: nmcli refuses a scan that
    follows too closely on the last one and reports that refusal as an error,
    which is not a fault — it means recent results already exist.  The sleep is
    what matters.  Without it the list that follows is served from the cache
    and an access point that came up two seconds ago is invisible, which is the
    same stale-cache trap that cost us a fix on the Nano side.
    """
    _run(["nmcli", "device", "wifi", "rescan", "ifname", IFACE], timeout=25)
    time.sleep(2.0)


def _profile_exists():
    rc, out = _run(["nmcli", "-t", "-f", "NAME", "connection", "show"], timeout=8)
    return rc == 0 and any(n.strip() == SSID for n in out.splitlines())


def _repair_profile():
    """Make the saved profile survivable, without touching its IP address.

    Called once at startup.  Every setting here is identical on both Pis and
    every one of them has been a failure on this rig:

      autoconnect-retries 0   infinite retries.  The default of 4 is the whole
                              reason a Pi that boots alongside the Nanos can
                              end up permanently disconnected.
      autoconnect-priority    NanoNet beats any other saved network for the radio.
      never-default / gateway NanoNet has no route to anywhere.  Pi 1's original
                              profile set ipv4.gateway, which installs a default
                              route over wlan0 and takes out both internet and
                              the wired ssh path used to administer the board.
                              Clearing it is not cosmetic.
      powersave 2             disabled.  Power save parks the radio between
                              beacons and shows up as periodic multi-hundred-ms
                              latency spikes and dropped associations.

    ipv4.addresses is untouched: it is the one setting that legitimately
    differs between the boards (.11 and .12), and a supervisor that normalised
    it would hand both Pis the same address.
    """
    if not _profile_exists():
        log.warning("no saved '%s' profile — creating one (DHCP, no default route)", SSID)
        # No static address is invented here.  If the profile is missing
        # entirely something is wrong with the board's setup, and a DHCP
        # fallback at least restores a usable link and ssh rather than
        # guessing which Pi this is and possibly colliding with the other.
        _run(["nmcli", "connection", "add", "type", "wifi", "ifname", IFACE,
              "con-name", SSID, "ssid", SSID,
              "autoconnect", "yes",
              "connection.autoconnect-priority", AUTOCONNECT_PRIORITY,
              "connection.autoconnect-retries", "0",
              "ipv4.method", "auto", "ipv4.never-default", "yes",
              "ipv6.method", "disabled",
              "802-11-wireless.powersave", "2",
              "wifi-sec.key-mgmt", "wpa-psk", "wifi-sec.psk", PSK], timeout=20)
        return

    rc, out = _run(["nmcli", "connection", "modify", SSID,
                    "connection.autoconnect", "yes",
                    "connection.autoconnect-priority", AUTOCONNECT_PRIORITY,
                    "connection.autoconnect-retries", "0",
                    "ipv4.never-default", "yes",
                    "ipv4.gateway", "",
                    "ipv6.method", "disabled",
                    "802-11-wireless.powersave", "2"], timeout=20)
    if rc == 0:
        log.info("profile '%s' hardened — infinite retries, priority %s, "
                 "no default route, power save off", SSID, AUTOCONNECT_PRIORITY)
    else:
        log.warning("could not harden profile '%s': %s", SSID, out)

    # The profile's powersave setting only applies on the next activation, so
    # the running interface is told directly as well.  Failure is not fatal —
    # some drivers do not expose it.
    _run(["iw", "dev", IFACE, "set", "power_save", "off"], timeout=5)


def _connection_up():
    """Ask NetworkManager to bring NanoNet up.  Returns True on success."""
    rc, out = _run(["nmcli", "connection", "up", SSID, "ifname", IFACE], timeout=45)
    if rc != 0:
        log.debug("connection up failed: %s", out.replace("\n", " ")[:200])
    return rc == 0


def _bounce_device():
    """Disconnect and reconnect the interface itself.

    One level above re-activating the profile: this clears supplicant state
    that a profile activation reuses, which is what recovers an association to
    an access point that has gone away without telling anyone.
    """
    _run(["nmcli", "device", "disconnect", IFACE], timeout=20)
    time.sleep(1.0)
    _run(["nmcli", "device", "connect", IFACE], timeout=30)


def _bounce_radio():
    """Turn the Wi-Fi radio off and on.

    Recovers a wedged driver without restarting NetworkManager, so the wired
    administration path stays up.
    """
    _run(["nmcli", "radio", "wifi", "off"], timeout=15)
    time.sleep(2.0)
    _run(["nmcli", "radio", "wifi", "on"], timeout=15)
    time.sleep(3.0)


def _restart_nm():
    """Last resort.  Drops every connection on the board, wired included."""
    log.warning("restarting NetworkManager — every connection on this board "
                "drops briefly, including the wired ssh session")
    _run(["systemctl", "restart", "NetworkManager"], timeout=60)
    time.sleep(5.0)


# ═══════════════════════════════════════════════════════════════════════════
#  Supervisor
# ═══════════════════════════════════════════════════════════════════════════

class LinkSupervisor:
    """Keeps the NanoNet link up, escalating through progressively blunter fixes.

    The ladder exists because the cheap fix handles almost every case and the
    expensive ones are disruptive.  Starting at the top every time would mean a
    NetworkManager restart for a fault that one `connection up` would have
    cleared; starting at the bottom and never escalating would mean retrying a
    fix that has already been shown not to work.  The attempt counter resets on
    every recovery, so a link that flaps stays on the cheap rungs.
    """

    def __init__(self):
        self._attempt = 0
        self._waiting = 0
        self._strikes = 0
        self._healthy = None          # None = not yet determined, so the first
                                      # result always logs
        self._last_nm_restart = 0.0
        self._last_heartbeat = 0.0
        self._connecting_since = None
        self._down_since = None

    # -- health ------------------------------------------------------------

    def _assess(self):
        """Return one of: "up", "connecting", "down".

        "down" means the sensor cannot deliver a frame right now, for any
        reason — not associated, no address, or associated to something that
        does not answer.  The distinction between those causes is logged but
        does not change the response, because the response is the same ladder.
        """
        state, conn = _nm_device_state()

        if state.startswith("connect") and state != "connected":
            # "connecting", "connecting (prepare)", "connecting (getting IP)" …
            return "connecting"

        if state != "connected" or conn != SSID:
            log.debug("device state=%s connection=%s", state, conn)
            return "down"

        if not _has_ipv4():
            log.debug("associated to %s but no IPv4 address on %s", SSID, IFACE)
            return "down"

        if _master_reachable():
            self._strikes = 0
            return "up"

        # Associated, addressed, and the master does not answer.  This is the
        # failover window or a zombie access point, and the two are
        # indistinguishable from here — so give the swarm time to sort itself
        # out before intervening.
        self._strikes += 1
        if self._strikes < UNREACHABLE_STRIKES:
            log.debug("master %s unreachable (%d/%d) — waiting, the swarm may "
                      "be mid-failover", MASTER_IP, self._strikes, UNREACHABLE_STRIKES)
            return "connecting"

        log.warning("associated to %s but %s unreachable for %d checks — "
                    "treating the association as dead",
                    SSID, MASTER_IP, self._strikes)
        self._strikes = 0
        return "down"

    # -- recovery ----------------------------------------------------------

    def _recover(self):
        """One rung of the ladder per call, so the health check runs in between."""
        self._attempt += 1
        n = self._attempt

        if n == 1:
            log.info("link down — reactivating %s", SSID)
            _connection_up()

        elif n in (2, 3, 5, 6, 7):
            log.info("link down (attempt %d) — rescanning, then reactivating", n)
            _rescan()
            if not _connection_up() and n == 3:
                # Worth knowing which of the two faults this is, once, rather
                # than on every pass: an absent access point is the Nanos' job
                # and nothing here will fix it; a present one that will not
                # accept us is this board's problem.
                if _ssid_on_air():
                    log.warning("%s IS on the air but will not accept us — "
                                "check the passphrase and this board's radio", SSID)
                else:
                    log.warning("%s is not on the air — no Nano is holding the "
                                "access point yet.  Still scanning.", SSID)

        elif n == 4:
            log.info("link down (attempt %d) — bouncing %s", n, IFACE)
            _bounce_device()
            _connection_up()

        else:
            # Everything below this point is disruptive — cycling the radio
            # drops the association, restarting NetworkManager drops every
            # connection on the board including the wired ssh path — so none of
            # it happens until it is established that the fault is HERE.
            #
            # That distinction matters enormously on this rig.  The Nanos
            # stagger their AP claim to avoid a split brain, so a Pi powered on
            # with them spends the better part of a minute with no NanoNet on
            # the air, through no fault of its own.  A ladder that escalated on
            # elapsed time alone would reach "restart NetworkManager" during
            # every single cold boot, punishing the board for the Nanos being
            # slow and adding a NM restart to the demo's critical path.
            #
            # An absent SSID is therefore not an escalation at all.  Keep
            # scanning, forever if necessary, and say so.
            if not _ssid_on_air():
                # Hold the ladder where it is rather than letting it advance.
                #
                # Waiting for an access point that does not exist is not a
                # failed repair attempt, and counting it as one has a nasty
                # consequence: after a few minutes of waiting the counter is
                # deep into the blunt rungs, so the very moment the AP finally
                # appears the response is to restart NetworkManager — instead
                # of the single `connection up` that would have worked. The
                # board would then take the longest possible path to joining a
                # network that was ready for it.
                #
                # Frozen here, the ladder resumes from the cheap rung when the
                # AP shows up, which is exactly where it should be.
                self._attempt = n - 1
                self._waiting += 1
                if self._waiting % 15 == 1:
                    log.warning("still waiting for %s — no Nano is holding the "
                                "access point.  Nothing to fix on this board; "
                                "scanning.", SSID)
                _rescan()
                _connection_up()
                return

            if self._waiting:
                # The access point has just come on the air after a wait.
                #
                # Every failure counted in the ladder up to this point is
                # explained by the AP not existing, so none of it is evidence
                # about this board and the ladder's position is meaningless.
                # Throw it away and start again at the cheapest rung — which,
                # for a network that has only just appeared, is overwhelmingly
                # likely to be the one that works.
                log.info("%s is on the air after %d scans — reconnecting",
                         SSID, self._waiting)
                self._waiting = 0
                self._attempt = 1
                _connection_up()
                return

            if n in (8, 9):
                log.warning("%s IS on the air and we still cannot join "
                            "(attempt %d) — cycling the radio", SSID, n)
                _bounce_radio()
                _connection_up()
                return

            # The access point is up, the radio has been cycled, and this board
            # still will not join.  NetworkManager itself is the last suspect,
            # restarted at most once every NM_RESTART_COOLDOWN_SEC.
            now = time.time()
            if now - self._last_nm_restart >= NM_RESTART_COOLDOWN_SEC:
                self._last_nm_restart = now
                _restart_nm()
                _connection_up()
            else:
                _rescan()
                _connection_up()

    # -- main loop ---------------------------------------------------------

    def run(self):
        log.info("NanoNet link supervisor — iface=%s ssid=%s master=%s "
                 "interval=%.1fs", IFACE, SSID, MASTER_IP, CHECK_INTERVAL_SEC)
        _repair_profile()

        while True:
            status = self._assess()

            if status == "up":
                self._connecting_since = None
                if self._healthy is not True:
                    if self._down_since is not None:
                        log.info("link UP — %s reachable after %.0fs down",
                                 MASTER_IP, time.time() - self._down_since)
                        if RESTART_CHRONY:
                            _run(["systemctl", "restart", "chrony"], timeout=30)
                    else:
                        log.info("link UP — %s reachable via %s", MASTER_IP, SSID)
                    self._healthy = True
                    self._attempt = 0
                    self._down_since = None
                    self._last_heartbeat = time.time()

                elif time.time() - self._last_heartbeat >= HEARTBEAT_SEC:
                    self._last_heartbeat = time.time()
                    log.info("link healthy — %s reachable", MASTER_IP)

            elif status == "connecting":
                # An activation in progress is not a fault, so the ladder is
                # not advanced — but it is not allowed to last forever either.
                if self._connecting_since is None:
                    self._connecting_since = time.time()
                elif time.time() - self._connecting_since > CONNECTING_PATIENCE_SEC:
                    log.warning("%s stuck connecting for %.0fs — escalating",
                                IFACE, time.time() - self._connecting_since)
                    self._connecting_since = None
                    self._healthy = False
                    if self._down_since is None:
                        self._down_since = time.time()
                    self._recover()

            else:
                self._connecting_since = None
                if self._healthy is not False:
                    self._healthy = False
                    self._down_since = time.time()
                    self._attempt = 0
                self._recover()

            time.sleep(CHECK_INTERVAL_SEC)


if __name__ == "__main__":
    if os.geteuid() != 0:
        log.error("must run as root — nmcli needs it to activate a connection")
        sys.exit(1)
    try:
        LinkSupervisor().run()
    except KeyboardInterrupt:
        log.info("stopped")
