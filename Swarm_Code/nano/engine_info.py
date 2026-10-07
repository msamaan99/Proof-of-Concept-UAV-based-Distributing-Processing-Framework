#!/usr/bin/env python3
"""Report what a TensorRT engine actually contains, so two can be compared.

Run on both Nanos and diff the output:

    python3 engine_info.py

A checksum tells you two engines differ; it cannot tell you why, and "why" is
the whole question when one board runs the same model at half the speed. This
opens the engine and reports the things that change speed:

  precision     the input/output binding dtypes, and a per-layer precision
                histogram where TensorRT exposes one. FP32 against FP16 is
                roughly 2x on a Nano's GPU, and it is a build-time flag, not a
                property of the board.
  shapes        an engine built for a larger input costs more per frame, and the
                shape is fixed at build time.
  TRT version   engines are serialized against a specific TensorRT build. Two
                boards on different JetPack releases cannot produce comparable
                engines even from identical weights and flags.
  memory        device memory the engine reserves, which reflects workspace and
                tactic choices made during the build.

Python 3.6 compatible — the Jetsons run 3.6.9 on Ubuntu 18.04.
"""

import hashlib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
YOLO_ROOT = os.path.dirname(HERE)
DEFAULT = os.path.join(YOLO_ROOT, "yolov5n.engine")


def file_facts(path):
    h = hashlib.md5()
    size = 0
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            h.update(chunk)
    return size, h.hexdigest(), os.path.getmtime(path)


def dtype_name(dt):
    return str(dt).replace("DataType.", "")


def describe(path):
    import tensorrt as trt

    print("\n\033[1mFile\033[0m")
    size, md5, mtime = file_facts(path)
    print("  %-22s %s" % ("path", path))
    print("  %-22s %d bytes (%.2f MB)" % ("size", size, size / 1e6))
    print("  %-22s %s" % ("md5", md5))
    print("  %-22s %s" % ("built", time.strftime("%Y-%m-%d %H:%M:%S",
                                                 time.localtime(mtime))))

    print("\n\033[1mTensorRT\033[0m")
    print("  %-22s %s" % ("runtime version", trt.__version__))

    logger = trt.Logger(trt.Logger.ERROR)      # quiet: we want our own output
    with open(path, "rb") as fh, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(fh.read())

    if engine is None:
        print("\n  \033[31mDeserialize failed.\033[0m An engine only loads on the "
              "TensorRT version\n  it was built with — this alone would explain "
              "two boards behaving differently.")
        return

    print("\n\033[1mBindings\033[0m")
    halves = 0
    floats = 0

    # TensorRT 8.5+ replaced the binding API with named tensors. Try the new
    # names first and fall back, so this runs on whatever JetPack each board has.
    if hasattr(engine, "num_io_tensors"):
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(name)
            dt = engine.get_tensor_dtype(name)
            mode = engine.get_tensor_mode(name)
            is_in = str(mode).endswith("INPUT")
            print("  %-6s %-22s %-22s %s"
                  % ("in" if is_in else "out", name, tuple(shape), dtype_name(dt)))
            if dtype_name(dt) == "HALF":
                halves += 1
            elif dtype_name(dt) == "FLOAT":
                floats += 1
    else:
        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            shape = engine.get_binding_shape(i)
            dt = engine.get_binding_dtype(i)
            is_in = engine.binding_is_input(i)
            print("  %-6s %-22s %-22s %s"
                  % ("in" if is_in else "out", name, tuple(shape), dtype_name(dt)))
            if dtype_name(dt) == "HALF":
                halves += 1
            elif dtype_name(dt) == "FLOAT":
                floats += 1

    print("\n\033[1mEngine\033[0m")
    for attr, label in (("num_layers", "layers"),
                        ("device_memory_size", "device memory"),
                        ("max_batch_size", "max batch"),
                        ("num_optimization_profiles", "opt profiles")):
        if hasattr(engine, attr):
            v = getattr(engine, attr)
            if label == "device memory":
                print("  %-22s %d bytes (%.2f MB)" % (label, v, v / 1e6))
            else:
                print("  %-22s %s" % (label, v))
    if hasattr(engine, "has_implicit_batch_dimension"):
        print("  %-22s %s" % ("implicit batch", engine.has_implicit_batch_dimension))

    # Per-layer precision is the definitive answer, and only some TRT versions
    # expose it. Counted rather than dumped: a yolov5n engine is ~100 layers and
    # the histogram is what distinguishes an FP16 build from an FP32 one.
    counts = {}
    try:
        import json
        insp = engine.create_engine_inspector()
        raw = insp.get_engine_information(trt.LayerInformationFormat.JSON)
        info = json.loads(raw)
        layers = info.get("Layers", info if isinstance(info, list) else [])
        for layer in layers:
            if isinstance(layer, dict):
                p = layer.get("Precision") or layer.get("precision") or "?"
                counts[p] = counts.get(p, 0) + 1
    except Exception:
        pass

    print("\n\033[1mPrecision\033[0m")
    if counts:
        for p in sorted(counts, key=lambda k: -counts[k]):
            print("  %-22s %d layers" % (p, counts[p]))
    else:
        print("  %-22s %s" % ("per-layer", "not exposed by TensorRT %s" % trt.__version__))
    print("  %-22s %d half, %d float" % ("bindings", halves, floats))

    if floats and not halves:
        print("\n  \033[33m! Every binding is FP32. If the other board's engine is FP16,\n"
              "    that is your 2x — rebuild this one with --half.\033[0m")
    print("")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    if not os.path.exists(path):
        raise SystemExit("No engine at %s" % path)
    print("\n" + "=" * 70)
    print("  %s" % os.uname()[1])
    print("=" * 70)
    try:
        describe(path)
    except ImportError:
        raise SystemExit("tensorrt is not importable here — run this on a Nano.")


if __name__ == "__main__":
    main()
