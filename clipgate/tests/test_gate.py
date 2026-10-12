import numpy as np
import pytest
import soundfile as sf
from scipy.signal import resample_poly

from clipgate import synth
from clipgate.gate import FLAG, PASS, REJECT, judge
from clipgate.measure import measure

SR = 44100


@pytest.fixture(scope="module")
def clean():
    return synth.speechlike(sr=SR)[0]


def run(tmp_path, name, y, sr=SR):
    p = tmp_path / f"{name}.wav"
    sf.write(p, y, sr)
    m = measure(p)
    return m, *judge(m)


def codes(reasons):
    return {r.split(":")[1] for r in reasons}


def test_clean_passes(tmp_path, clean):
    m, v, r = run(tmp_path, "clean", clean)
    assert v == PASS, r
    assert m.bw_ratio > 0.9


def test_digital_phone_upsample_rejected(tmp_path, clean):
    m, v, r = run(tmp_path, "phone", synth.telephone(clean, SR))
    assert v == REJECT
    assert {"narrowband", "upsampled"} <= codes(r)
    assert 3000 < m.effective_bw_hz < 4500


def test_tape_phone_with_hiss_rejected(tmp_path, clean):
    # the case a fixed level threshold misses: hiss fills the top of the band
    m, v, r = run(tmp_path, "tape", synth.telephone(clean, SR, hiss_dbfs=-55))
    assert v == REJECT
    assert "narrowband" in codes(r)


def test_heavy_tape_hiss_phone_rejected(tmp_path, clean):
    # hiss ~13 dB under the speech shrinks the cliff below 15 dB; the flat hiss above it confirms the edge
    m, v, r = run(tmp_path, "heavy_tape", synth.telephone(clean, SR, hiss_dbfs=-40))
    assert v == REJECT
    assert "narrowband" in codes(r)


def test_codec_lowpass_passes(tmp_path, clean):
    m, v, r = run(tmp_path, "mp3ish", synth.lowpassed(clean, SR, 16000))
    assert v == PASS, r
    assert 15000 < m.effective_bw_hz < 18000  # edge resolution is ~win_hz


def test_true_16k_wideband_passes(tmp_path, clean):
    y = resample_poly(clean, 16000, SR)
    m, v, r = run(tmp_path, "wb16", y, sr=16000)
    assert v == PASS, r


def test_wideband_upsampled_flagged_not_rejected(tmp_path, clean):
    # 16k content stored at 44.1k: usable audio, wrong metadata
    y = resample_poly(resample_poly(clean, 16000, SR), SR, 16000)
    m, v, r = run(tmp_path, "up", y)
    assert v == FLAG
    assert codes(r) == {"upsampled"}


def test_clipping_rejected(tmp_path, clean):
    _, v, r = run(tmp_path, "clip", synth.clipped(clean))
    assert v == REJECT and "clipping" in codes(r)


def test_noise_rejected(tmp_path, clean):
    _, v, r = run(tmp_path, "noisy", synth.noisy(clean, -25))
    assert v == REJECT and "noisy" in codes(r)


def test_silence_rejected(tmp_path, clean):
    _, v, r = run(tmp_path, "silent", synth.mostly_silent(clean, SR))
    assert v == REJECT and "silence" in codes(r)


def test_wrong_duration_rejected(tmp_path, clean):
    _, v, r = run(tmp_path, "short", clean[: 10 * SR])
    assert v == REJECT and "duration" in codes(r)


def test_soft_voice_not_read_as_narrowband(tmp_path):
    # soft speech drops steeply above ~1 kHz; searching from 1 kHz called this telephone audio
    soft, _ = synth.speechlike(sr=SR, tilt=2.5, breath=0.0)
    m, v, r = run(tmp_path, "soft", soft)
    assert "narrowband" not in codes(r)
    assert m.effective_bw_hz > 7000


def test_quiet_clean_recording_flagged_not_rejected(tmp_path, clean):
    # 30 dB down is fixable with gain; it must not read as silent or be rejected
    m, v, r = run(tmp_path, "quiet", clean * 10 ** (-30 / 20))
    assert v == FLAG and codes(r) == {"quiet"}
    assert m.silence_fraction < 0.3


def test_stereo_is_mixed_down(tmp_path, clean):
    m, v, _ = run(tmp_path, "stereo", np.stack([clean, clean], axis=1))
    assert m.channels == 2 and v == PASS
