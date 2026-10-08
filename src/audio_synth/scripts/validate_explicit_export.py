"""Audit local trained exports against their pinned publisher originals.

Requires onnx, numpy and onnxruntime. No downloads or test speech substitutes.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, default=Path("build/pretrained-inspect/vctk"))
parser.add_argument("--assets", type=Path, default=Path("model_assets/explicit_neural"))
parser.add_argument("--output", type=Path, default=Path("artifacts/explicit_neural/export_validation.json"))
args = parser.parse_args()
options = ort.SessionOptions()
options.intra_op_num_threads = 4
options.inter_op_num_threads = 1

def session(path):
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])

def constants(path):
    model = onnx.load(path)
    return {node.output[0]: node.attribute[0].t.SerializeToString()
            for node in model.graph.node if node.op_type == "Constant"}

audit = {}
for exported, original in (("acoustic_generator.onnx", "fastspeech2_vctk.onnx"),
                           ("prosody_predictor.onnx", "fastspeech2_vctk.onnx"),
                           ("vocoder_hifigan.onnx", "hifigan_vctk.onnx")):
    before, after = constants(args.source / original), constants(args.assets / exported)
    retained = before.keys() & after.keys()
    assert retained and all(before[key] == after[key] for key in retained)
    audit[exported] = {"retained_constants": len(retained),
                       "retained_tensor_bytes": sum(len(before[key]) for key in retained),
                       "retained_values_byte_identical": True}

diagnostics = json.loads((args.output.parent / "neutral.diagnostics.json").read_text())
ids = np.asarray([diagnostics["input_ids"]], dtype=np.int64)
sid = np.array([0], np.int64)
predictor = session(args.assets / "prosody_predictor.onnx")
durations, pitch, energy = predictor.run(None, {"input_ids": ids, "sid": sid})
feed = {"input_ids": ids, "sid": sid, "durations": durations,
        "f0": np.repeat(pitch[0], durations[0])[None, :],
        "energy": np.repeat(energy[0], durations[0])[None, :]}
source = session(args.source / "fastspeech2_vctk.onnx")
original = source.run(None, {"text": ids[0], "spk_id": sid})[0]
acoustic = session(args.assets / "acoustic_generator.onnx")
mel = acoustic.run(None, feed)[0]
error = float(np.max(np.abs(mel[0].T - original)))
assert error < 1e-4, error
audit["neutral_mel_max_abs_error"] = error
audit["exact_acoustic_inputs"] = [item.name for item in acoustic.get_inputs()]
assert audit["exact_acoustic_inputs"] == ["input_ids", "durations", "f0", "energy", "sid"]
changes = {}
for name, changed in (("f0", feed["f0"] * 1.15), ("energy", feed["energy"] * .7),
                      ("sid", np.array([1], np.int64))):
    altered = acoustic.run(None, {**feed, name: changed})[0]
    difference = float(np.max(np.abs(altered - mel)))
    assert difference > 1e-3
    changes[name] = difference
zero = durations.copy(); zero[0, 0] = 0
frames = int(zero.sum())
zero_mel = acoustic.run(None, {**feed, "durations": zero,
    "f0": np.repeat(pitch[0], zero[0])[None, :], "energy": np.repeat(energy[0], zero[0])[None, :]})[0]
assert zero_mel.shape == (1, 80, frames)
audit["control_mel_max_abs_changes"] = changes
audit["zero_duration_frame_count"] = frames
original_vocoder = session(args.source / "hifigan_vctk.onnx")
native = original_vocoder.run(None, {original_vocoder.get_inputs()[0].name: mel[0].T.copy()})[0]
vocoder = session(args.assets / "vocoder_hifigan.onnx")
audio = vocoder.run(None, {"mel": mel})[0]
vocoder_error = float(np.max(np.abs(audio - native.T)))
assert vocoder_error < 1e-6 and audio.shape == (1, int(durations.sum()) * 300)
audit["vocoder_max_abs_error"] = vocoder_error
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit, indent=2))
