"""Export the locally trained heatmap model and characterize static INT8 drift.

Calibration uses training images only. Export success is not accuracy or hardware qualification.
"""

import hashlib
import json
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from fetch_benchmark import ROOT
from image_perception import prepared
from learned_perception import HeatmapNet, decode, image_tensor, split_training


class TrainingCalibration(CalibrationDataReader):
    def __init__(self, values):
        self.values = iter(values)

    def get_next(self):
        value = next(self.values, None)
        return None if value is None else {"image": value}


def main():
    torch.set_num_threads(4)
    out = ROOT / "output/learned"
    metadata = json.loads((out / "training.json").read_text())
    selection = json.loads((ROOT / "output/datasets/selection.json").read_text())
    train, dev = split_training(selection["groups"]["speedplus/synthetic/train"])
    speed = ROOT / "output/datasets/speedplus/speedplusv2"
    camera = json.loads((speed / "camera.json").read_text())

    def tensors(samples, permitted):
        result = []
        for sample in samples:
            path = speed / "synthetic/images" / sample["filename"]
            if hashlib.sha256(path.read_bytes()).hexdigest() not in permitted:
                raise ValueError("Export/calibration provenance mismatch")
            image, _ = prepared(path, camera)
            result.append(image_tensor(image)[None].numpy())
        return result

    calibration = tensors(train[:32], set(metadata["training_hashes"]))
    development = tensors(dev[:16], set(metadata["development_hashes"]))
    model = HeatmapNet()
    model.load_state_dict(
        torch.load(out / "heatmap.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    float_path, int_path = out / "heatmap.onnx", out / "heatmap-int8.onnx"
    torch.onnx.export(
        model,
        torch.zeros(1, 1, 192, 256),
        str(float_path),
        input_names=["image"],
        output_names=["heatmaps"],
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(str(float_path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    reference = ort.InferenceSession(
        str(float_path), options, providers=["CPUExecutionProvider"]
    )
    maximum = 0.0
    for value in development:
        with torch.inference_mode():
            expected = model(torch.from_numpy(value)).numpy()
        actual = reference.run(None, {"image": value})[0]
        maximum = max(maximum, float(np.abs(expected - actual).max()))
    if maximum > 1e-4:
        raise ValueError(
            f"ONNX FP32 parity failed: max absolute logits difference {maximum}"
        )
    prepared_path = out / "heatmap-prepared.onnx"
    quant_pre_process(str(float_path), str(prepared_path))
    quantize_static(
        str(prepared_path),
        str(int_path),
        TrainingCalibration(calibration),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    onnx.checker.check_model(str(int_path))
    quantized = ort.InferenceSession(
        str(int_path), options, providers=["CPUExecutionProvider"]
    )
    displacement, timing = [], []
    # Warm-up, then characterize development heatmaps; audit images never calibrate INT8.
    quantized.run(None, {"image": development[0]})
    for value in development:
        floating = reference.run(None, {"image": value})[0][0]
        start = time.perf_counter_ns()
        integer = quantized.run(None, {"image": value})[0][0]
        timing.append((time.perf_counter_ns() - start) / 1000)
        first, _ = decode(floating, (480, 640))
        second, _ = decode(integer, (480, 640))
        displacement.extend(np.linalg.norm(first - second, axis=1).tolist())
    report = {
        "onnxruntime": ort.__version__,
        "onnx": onnx.__version__,
        "provider": "CPUExecutionProvider",
        "threads": 4,
        "calibration_training_images": len(calibration),
        "development_images": len(development),
        "fp32_max_absolute_logit_difference": maximum,
        "fp32_bytes": float_path.stat().st_size,
        "int8_bytes": int_path.stat().st_size,
        "int8_development_keypoint_drift_rms_px": float(
            np.sqrt(np.mean(np.square(displacement)))
        ),
        "int8_development_keypoint_drift_worst_px": max(displacement),
        "int8_host_inference_p95_us": float(np.percentile(timing, 95)),
        "int8_sha256": hashlib.sha256(int_path.read_bytes()).hexdigest(),
        "hardware_verified": False,
        "deployment_approved": False,
        "limitations": "Heatmap drift only; task accuracy and confidence calibration require separate evaluation. Export/INT8 runtime exclude image IO, rectification and PnP.",
    }
    (out / "export.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
