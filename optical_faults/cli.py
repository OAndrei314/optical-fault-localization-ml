"""`python -m optical_faults.cli generate|train|plot-examples ...`"""
from __future__ import annotations

import argparse
import os

import joblib
import numpy as np

from . import FAULT_TYPES
from .dataset import generate_dataset, load_dataset, save_dataset
from .model import train_and_evaluate
from .simulate import simulate_trace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="optical-faults")
    sub = parser.add_subparsers(dest="command", required=True)

    gen_p = sub.add_parser("generate", help="generate a synthetic labeled dataset")
    gen_p.add_argument("--n", type=int, default=1200)
    gen_p.add_argument("--seed", type=int, default=0)
    gen_p.add_argument("--out", required=True, help="output .npz path")

    train_p = sub.add_parser("train", help="train + evaluate the classifier and regressor")
    train_p.add_argument("--data", required=True, help="path to dataset .npz from `generate`")
    train_p.add_argument("--out", required=True, help="output directory for saved models")
    train_p.add_argument("--seed", type=int, default=0)

    plot_p = sub.add_parser("plot-examples", help="save one example trace per fault type")
    plot_p.add_argument("--seed", type=int, default=1)
    plot_p.add_argument("--out", required=True, help="output directory for PNGs")

    args = parser.parse_args(argv)

    if args.command == "generate":
        X, y_type, y_position = generate_dataset(args.n, seed=args.seed)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        save_dataset(args.out, X, y_type, y_position)
        print(f"wrote {args.n} samples -> {args.out}")
        return 0

    if args.command == "train":
        X, y_type, y_position = load_dataset(args.data)
        result = train_and_evaluate(X, y_type, y_position, seed=args.seed)
        os.makedirs(args.out, exist_ok=True)
        joblib.dump(result.classifier, os.path.join(args.out, "classifier.joblib"))
        joblib.dump(result.regressor, os.path.join(args.out, "regressor.joblib"))
        print(f"fault-type accuracy: {result.accuracy:.3f}")
        print(f"localization MAE: {result.localization_mae_km:.3f} km")
        print(result.report)
        return 0

    if args.command == "plot-examples":
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(args.out, exist_ok=True)
        rng = np.random.default_rng(args.seed)
        for fault_type in FAULT_TYPES:
            sample = simulate_trace(fault_type, rng=rng)
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.plot(sample.distance_km, sample.power_db, linewidth=1)
            if sample.fault_position_km is not None:
                ax.axvline(sample.fault_position_km, color="red", linestyle="--", alpha=0.6)
            ax.set_title(fault_type)
            ax.set_xlabel("distance (km)")
            ax.set_ylabel("power (dB)")
            fig.tight_layout()
            fig.savefig(os.path.join(args.out, f"{fault_type}.png"), dpi=120)
            plt.close(fig)
        print(f"wrote {len(FAULT_TYPES)} example plots -> {args.out}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
