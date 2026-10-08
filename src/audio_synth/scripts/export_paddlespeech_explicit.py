#!/usr/bin/env python3
"""Export the pinned PaddleSpeech VCTK trained graphs to our explicit interface.

Weights are retained verbatim. The trained 1x1 pitch/energy embeddings commute
with duration expansion, so repeated token contours reproduce upstream output;
varying frame contours can condition the decoder at individual frame positions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T, helper as H, numpy_helper as N


def make_model(source, extra, inputs, outputs, replace=(), name="explicit_vctk"):
    producers = {output: node for node in source.graph.node for output in node.output}
    for removed in replace:
        producers.pop(removed)
    for node in extra:
        for output in node.output: producers[output] = node
    input_names = {v.name for v in inputs}
    ordered, visited = [], set()

    def visit(value):
        if not value or value in input_names: return
        node = producers[value]
        if node.output[0] in visited: return
        for item in node.input: visit(item)
        visited.add(node.output[0]); ordered.append(node)

    for output in outputs: visit(output.name)
    graph = H.make_graph(ordered, name, inputs, outputs)
    model = H.make_model(graph, opset_imports=source.opset_import, ir_version=source.ir_version)
    H.set_model_props(model, {"sample_rate": "24000", "hop_length": "300", "speaker_count": "109",
                             "source": "PaddleSpeech fastspeech2_vctk_onnx_1.1.0",
                             "control_basis": "frame-level learned 1x1 embeddings; continuous log F0"})
    onnx.checker.check_model(model)
    return model


def export(source_dir: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    model = onnx.load(source_dir / "fastspeech2_vctk.onnx")
    pm, ps = np.load(source_dir / "pitch_stats.npy").reshape(-1)
    em, es = np.load(source_dir / "energy_stats.npy").reshape(-1)
    # Stable internal names are pinned to this archive's SHA-256 by the fetcher.
    assert any(n.output[0] == "linear_53.tmp_1" for n in model.graph.node)
    constants = []
    def const(name, value, dtype=np.int64):
        constants.append(H.make_node("Constant", [], [name], value=N.from_array(np.asarray(value, dtype=dtype))))
        return name
    for name, value in [("explicit_axis0", [0]), ("explicit_axis1", [1]), ("explicit_axis2", [2]),
                        ("explicit_axis3", [3]), ("explicit_axes13", [1, 3]),
                        ("explicit_zero", 0), ("explicit_one", 1), ("explicit_cumsum_axis", 1)]:
        const(name, value)
    for name, value in [("explicit_pitch_mean", pm), ("explicit_pitch_std", ps),
                        ("explicit_energy_mean", em), ("explicit_energy_std", es),
                        ("explicit_float_zero", 0.), ("explicit_f0_floor", 1.)]:
        const(name, value, np.float32)
    def node(op, inputs, output, **kwargs): return H.make_node(op, inputs, [output], **kwargs)
    prefix = [node("Squeeze", ["input_ids", "explicit_axis0"], "text"), node("Identity", ["sid"], "spk_id")]
    acoustic_inputs = [H.make_tensor_value_info(name, kind, shape) for name, kind, shape in
                       [("input_ids", T.INT64, [1, "tokens"]), ("durations", T.INT64, [1, "tokens"]),
                        ("f0", T.FLOAT, [1, "frames"]), ("energy", T.FLOAT, [1, "frames"]), ("sid", T.INT64, [1])]]
    extra = constants + prefix + [
        node("CumSum", ["durations", "explicit_cumsum_axis"], "explicit_ends"),
        node("ReduceSum", ["durations", "explicit_axis1"], "explicit_count", keepdims=0),
        node("Squeeze", ["explicit_count", "explicit_axis0"], "explicit_count_scalar"),
        node("Range", ["explicit_zero", "explicit_count_scalar", "explicit_one"], "explicit_frames"),
        node("Squeeze", ["explicit_ends", "explicit_axis0"], "explicit_ends_vector"),
        node("Unsqueeze", ["explicit_frames", "explicit_axis1"], "explicit_frame_column"),
        node("GreaterOrEqual", ["explicit_frame_column", "explicit_ends_vector"], "explicit_after_token"),
        node("Cast", ["explicit_after_token"], "explicit_after_int", to=T.INT64),
        node("ReduceSum", ["explicit_after_int", "explicit_axis1"], "explicit_token_indices", keepdims=0),
        node("Gather", ["linear_53.tmp_1", "explicit_token_indices"], "explicit_expanded_hidden", axis=1),
        node("Max", ["f0", "explicit_f0_floor"], "explicit_safe_f0"),
        node("Log", ["explicit_safe_f0"], "explicit_log_f0"),
        node("Sub", ["explicit_log_f0", "explicit_pitch_mean"], "explicit_centered_f0"),
        node("Div", ["explicit_centered_f0", "explicit_pitch_std"], "explicit_normalized_f0"),
        node("Greater", ["f0", "explicit_float_zero"], "explicit_voiced"),
        node("Where", ["explicit_voiced", "explicit_normalized_f0", "explicit_float_zero"], "explicit_pitch"),
        node("Mul", ["energy", "explicit_energy_mean"], "explicit_raw_energy"),
        node("Sub", ["explicit_raw_energy", "explicit_energy_mean"], "explicit_centered_energy"),
        node("Div", ["explicit_centered_energy", "explicit_energy_std"], "explicit_energy"),
        node("Unsqueeze", ["explicit_pitch", "explicit_axes13"], "explicit_pitch_conv_input"),
        node("Unsqueeze", ["explicit_energy", "explicit_axes13"], "explicit_energy_conv_input"),
        node("Conv", ["explicit_pitch_conv_input", "conv1d_15.w_0", "conv1d_15.b_0"], "explicit_pitch_embedding"),
        node("Conv", ["explicit_energy_conv_input", "conv1d_18.w_0", "conv1d_18.b_0"], "explicit_energy_embedding"),
        node("Squeeze", ["explicit_pitch_embedding", "explicit_axis3"], "explicit_pitch_channels"),
        node("Squeeze", ["explicit_energy_embedding", "explicit_axis3"], "explicit_energy_channels"),
        node("Transpose", ["explicit_pitch_channels"], "explicit_pitch_frames", perm=[0, 2, 1]),
        node("Transpose", ["explicit_energy_channels"], "explicit_energy_frames", perm=[0, 2, 1]),
        node("Add", ["explicit_expanded_hidden", "explicit_energy_frames"], "explicit_hidden_energy"),
        node("Add", ["explicit_hidden_energy", "explicit_pitch_frames"], "p2o.MatMul.57"),
        node("Transpose", [model.graph.output[0].name], "explicit_mel_channels", perm=[1, 0]),
        node("Unsqueeze", ["explicit_mel_channels", "explicit_axis0"], "mel"),
    ]
    acoustic = make_model(model, extra, acoustic_inputs, [H.make_tensor_value_info("mel", T.FLOAT, [1, 80, "frames"])],
                          replace=["p2o.MatMul.57"])
    onnx.save(acoustic, output / "acoustic_generator.onnx")
    predictor_nodes = constants + prefix + [
        node("Cast", ["where_10.tmp_0"], "durations", to=T.INT64),
        node("Squeeze", ["where_8.tmp_0", "explicit_axis2"], "predict_pitch"),
        node("Mul", ["predict_pitch", "explicit_pitch_std"], "predict_scaled_pitch"),
        node("Add", ["predict_scaled_pitch", "explicit_pitch_mean"], "predict_log_pitch"),
        node("Exp", ["predict_log_pitch"], "token_f0_hz"),
        node("Squeeze", ["where_9.tmp_0", "explicit_axis2"], "predict_energy"),
        node("Mul", ["predict_energy", "explicit_energy_std"], "predict_scaled_energy"),
        node("Add", ["predict_scaled_energy", "explicit_energy_mean"], "predict_raw_energy"),
        node("Div", ["predict_raw_energy", "explicit_energy_mean"], "predict_relative_energy"),
        node("Max", ["predict_relative_energy", "explicit_float_zero"], "token_energy"),
    ]
    predictor = make_model(model, predictor_nodes, [acoustic_inputs[0], acoustic_inputs[-1]],
                           [H.make_tensor_value_info("durations", T.INT64, [1, "tokens"]),
                            H.make_tensor_value_info("token_f0_hz", T.FLOAT, [1, "tokens"]),
                            H.make_tensor_value_info("token_energy", T.FLOAT, [1, "tokens"])], name="trained_vctk_prosody")
    onnx.save(predictor, output / "prosody_predictor.onnx")
    vocoder = onnx.load(source_dir / "hifigan_vctk.onnx")
    original_input, original_output = vocoder.graph.input[0].name, vocoder.graph.output[0].name
    vocoder_nodes = [
        node("Squeeze", ["mel"], "vocoder_channels", axes=[0]),
        node("Transpose", ["vocoder_channels"], original_input, perm=[1, 0]),
        node("Transpose", [original_output], "audio", perm=[1, 0]),
        node("Constant", [], "explicit_axis0", value=N.from_array(np.array([0], np.int64))),
    ]
    adapted = make_model(vocoder, vocoder_nodes, [H.make_tensor_value_info("mel", T.FLOAT, [1, 80, "frames"])],
                         [H.make_tensor_value_info("audio", T.FLOAT, [1, "samples"])], name="trained_vctk_hifigan")
    H.set_model_props(adapted, {"sample_rate": "24000", "hop_length": "300",
                              "source": "PaddleSpeech hifigan_vctk_onnx_1.1.0",
                              "control_basis": "layout-only adaptation of trained vocoder"})
    onnx.save(adapted, output / "vocoder_hifigan.onnx")
    vocabulary = (source_dir / "phone_id_map.txt").read_text(encoding="utf-8")
    (output / "tokens.tsv").write_text("# frontend=arpabet\n" + "\n".join("\t".join(line.split()) for line in vocabulary.splitlines()) + "\n", encoding="utf-8")
    speaker_map = (source_dir / "speaker_id_map.txt").read_text(encoding="utf-8")
    speakers = len(speaker_map.splitlines())
    for filename in ("acoustic_generator.onnx", "prosody_predictor.onnx", "vocoder_hifigan.onnx"):
        exported = onnx.load(output / filename)
        props = {p.key: p.value for p in exported.metadata_props}; props["speaker_count"] = str(speakers)
        H.set_model_props(exported, props); onnx.save(exported, output / filename)
    (output / "speaker_id_map.txt").write_text(speaker_map, encoding="utf-8")
    (output / "pipeline.properties").write_text(f"sample_rate=24000\nhop_length=300\nnum_speakers={speakers}\nprosody_predictor=required\n", encoding="utf-8")
    print(f"Exported trained VCTK explicit pipeline: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("model_assets/explicit_neural"))
    args = parser.parse_args()
    export(args.source, args.output)
