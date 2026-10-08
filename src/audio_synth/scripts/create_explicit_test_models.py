#!/usr/bin/env python3
"""Create tiny, untrained ONNX contract fixtures. These do not synthesize speech."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T, helper as H, numpy_helper as N


def acoustic(output: Path, mode: str = "valid") -> None:
    inputs = [H.make_tensor_value_info(name, kind, shape) for name, kind, shape in [
        ("input_ids", T.INT64, [1, "tokens"]), ("durations", T.INT64, [1, "tokens"]),
        ("f0", T.FLOAT, [1, "frames"]), ("energy", T.FLOAT, [1, "frames"]), ("sid", T.INT64, [1])]]
    constants = [N.from_array(np.array(value, dtype=dtype), name) for name, value, dtype in [
        ("f0_scale", .001, np.float32), ("energy_scale", .01, np.float32),
        ("token_scale", .00001, np.float32), ("duration_scale", .000001, np.float32),
        ("sid_scale", .001, np.float32), ("axis1", [1], np.int64),
        ("bins", np.arange(80, dtype=np.float32).reshape(1, 80, 1) * .01, np.float32)]]
    nodes = [H.make_node("Mul", ["f0", "f0_scale"], ["f0_term"]),
             H.make_node("Mul", ["energy", "energy_scale"], ["energy_term"]),
             H.make_node("Add", ["f0_term", "energy_term"], ["frame_terms"])]
    previous = "frame_terms"
    for name, scale in [("input_ids", "token_scale"), ("durations", "duration_scale"), ("sid", "sid_scale")]:
        nodes += [H.make_node("Cast", [name], [name + "_float"], to=T.FLOAT),
                  H.make_node("ReduceSum", [name + "_float"], [name + "_sum"], keepdims=0),
                  H.make_node("Mul", [name + "_sum", scale], [name + "_term"]),
                  H.make_node("Add", [previous, name + "_term"], [name + "_combined"])]
        previous = name + "_combined"
    nodes += [H.make_node("Unsqueeze", [previous, "axis1"], ["channel"]),
              H.make_node("Add", ["channel", "bins"], ["mel_float"])]
    if mode == "integer":
        nodes.append(H.make_node("Cast", ["mel_float"], ["mel"], to=T.INT64))
    elif mode == "nan":
        constants.append(N.from_array(np.array(float("nan"), np.float32), "nan"))
        nodes.append(H.make_node("Mul", ["mel_float", "nan"], ["mel"]))
    elif mode == "wrong_frames":
        constants += [N.from_array(np.array([x], np.int64), xname) for xname, x in [("start", 0), ("end", 1), ("axis", 2)]]
        nodes.append(H.make_node("Slice", ["mel_float", "start", "end", "axis"], ["mel"]))
    else:
        nodes.append(H.make_node("Identity", ["mel_float"], ["mel"]))
    graph = H.make_graph(nodes, "UNTRAINED_contract_fixture", inputs,
                         [H.make_tensor_value_info("mel", T.INT64 if mode == "integer" else T.FLOAT, [1, 80, "frames"])], constants)
    model = H.make_model(graph, opset_imports=[H.make_opsetid("", 17)], ir_version=9)
    if mode == "wrong_name":
        model.graph.input[0].name = "wrong_ids"
        for node in model.graph.node:
            for i, name in enumerate(node.input):
                if name == "input_ids": node.input[i] = "wrong_ids"
    onnx.checker.check_model(model)
    onnx.save(model, output)


def vocoder(output: Path, hop: int = 256) -> None:
    # Channel 0, repeated by hop length. Step-shaped test data, not a trained vocoder.
    constants = [N.from_array(np.array(value, np.int64), name) for name, value in [
        ("index", [0]), ("axis", [2]), ("repeats", [1, 1, hop]), ("reshape", [1, -1])]]
    nodes = [H.make_node("Gather", ["mel", "index"], ["channel"], axis=1),
             H.make_node("Tanh", ["channel"], ["bounded"]),
             H.make_node("Unsqueeze", ["bounded", "axis"], ["expanded"]),
             H.make_node("Reshape", ["expanded", "reshape"], ["frames"]),
             H.make_node("Unsqueeze", ["frames", "axis"], ["frame_columns"]),
             H.make_node("Tile", ["frame_columns", "repeats"], ["repeated"]),
             H.make_node("Reshape", ["repeated", "reshape"], ["audio"])]
    graph = H.make_graph(nodes, "UNTRAINED_vocoder_fixture",
                         [H.make_tensor_value_info("mel", T.FLOAT, [1, 80, "frames"])],
                         [H.make_tensor_value_info("audio", T.FLOAT, [1, "samples"])], constants)
    model = H.make_model(graph, opset_imports=[H.make_opsetid("", 17)], ir_version=9)
    onnx.checker.check_model(model)
    onnx.save(model, output)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "model_assets/explicit_fixture")
    parser.add_argument("--negative-fixtures", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    acoustic(args.output / "acoustic_generator.onnx")
    vocoder(args.output / "vocoder_hifigan.onnx")
    for name in ("cmudict.dict", "tokens.tsv"):
        shutil.copyfile(root / "tests/fixtures" / name, args.output / name)
    if args.negative_fixtures:
        for mode in ("integer", "nan", "wrong_frames", "wrong_name"):
            acoustic(args.output / (mode + ".onnx"), mode)
        vocoder(args.output / "wrong_hop.onnx", hop=128)
    print(f"UNTRAINED test fixtures: {args.output}")


if __name__ == "__main__":
    main()
