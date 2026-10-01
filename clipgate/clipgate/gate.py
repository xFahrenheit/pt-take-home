"""Turn metrics into PASS / FLAG / REJECT with reasons.

FLAG means "a human or a later stage looks at it", not "drop it". With 4 days
and a fixed count, I'd rather keep borderline clips in a review queue than
silently shrink the candidate pool.
"""
from dataclasses import dataclass

from .measure import ClipMetrics


@dataclass
class Thresholds:
    min_dur: float = 25.0
    max_dur: float = 35.0
    # Wideband speech carries real content to 7-8 kHz. Telephone stops at ~3.4-4 kHz.
    reject_bw_hz: float = 4500.0
    flag_bw_hz: float = 7000.0
    # declared SR is a claim; if content uses under half the band, the claim is wrong
    flag_bw_ratio: float = 0.5
    reject_clip: float = 1e-3
    flag_clip: float = 1e-4
    reject_silence: float = 0.5
    flag_silence: float = 0.3
    reject_snr: float = 15.0
    flag_snr: float = 25.0
    reject_rms_dbfs: float = -40.0


REJECT, FLAG, PASS = "REJECT", "FLAG", "PASS"


def judge(m: ClipMetrics, t: Thresholds = Thresholds()):
    reasons = []  # (severity, code, detail)

    def add(sev, code, detail):
        reasons.append((sev, code, detail))

    if not (t.min_dur <= m.duration_sec <= t.max_dur):
        add(REJECT, "duration", f"{m.duration_sec}s outside [{t.min_dur}, {t.max_dur}]")

    if m.effective_bw_hz < t.reject_bw_hz:
        add(REJECT, "narrowband", f"content stops at {m.effective_bw_hz:.0f} Hz (telephone band)")
    elif m.effective_bw_hz < t.flag_bw_hz:
        add(FLAG, "narrowband", f"content stops at {m.effective_bw_hz:.0f} Hz")

    if m.bw_ratio < t.flag_bw_ratio:
        add(FLAG, "upsampled", f"declared {m.declared_sr} Hz, content uses {m.bw_ratio:.0%} of band")

    if m.clip_fraction > t.reject_clip:
        add(REJECT, "clipping", f"{m.clip_fraction:.4%} samples at full scale")
    elif m.clip_fraction > t.flag_clip:
        add(FLAG, "clipping", f"{m.clip_fraction:.4%} samples at full scale")

    if m.silence_fraction > t.reject_silence:
        add(REJECT, "silence", f"{m.silence_fraction:.0%} of frames silent")
    elif m.silence_fraction > t.flag_silence:
        add(FLAG, "silence", f"{m.silence_fraction:.0%} of frames silent")

    if m.snr_db_est < t.reject_snr:
        add(REJECT, "noisy", f"est. SNR {m.snr_db_est:.1f} dB")
    elif m.snr_db_est < t.flag_snr:
        add(FLAG, "noisy", f"est. SNR {m.snr_db_est:.1f} dB")

    if m.rms_dbfs < t.reject_rms_dbfs:
        add(REJECT, "quiet", f"RMS {m.rms_dbfs:.1f} dBFS")

    sevs = {r[0] for r in reasons}
    verdict = REJECT if REJECT in sevs else FLAG if FLAG in sevs else PASS
    return verdict, [f"{s}:{c}:{d}" for s, c, d in reasons]
