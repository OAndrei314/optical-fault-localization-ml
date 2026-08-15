"""`python -m optical_faults.cli generate|train|plot-examples ...`"""
from __future__ import annotations

import argparse
import os

import joblib
import numpy as np

from . import FAULT_TYPES
from .dataset import generate_dataset, load_dataset, save_dataset
from .model import render_markdown_report, train_and_evaluate
from .multi_event import render_multi_event_report, run_multi_event_detection
from .multi_fault import render_multi_fault_report, run_multi_fault_stress
from .simulate import simulate_trace
from .stress import render_stress_report, run_domain_shift_stress


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
    train_p.add_argument("--report", help="optional markdown report path")

    plot_p = sub.add_parser("plot-examples", help="save one example trace per fault type")
    plot_p.add_argument("--seed", type=int, default=1)
    plot_p.add_argument("--out", required=True, help="output directory for PNGs")

    stress_p = sub.add_parser("stress", help="train once and evaluate shifted synthetic domains")
    stress_p.add_argument("--train-n", type=int, default=800)
    stress_p.add_argument("--test-n", type=int, default=300)
    stress_p.add_argument("--seed", type=int, default=0)
    stress_p.add_argument("--estimators", type=int, default=80)
    stress_p.add_argument("--report", required=True, help="output markdown report path")

    multi_p = sub.add_parser(
        "multi-fault-stress",
        help="train on single-fault traces, evaluate on traces with a second, unrelated fault",
    )
    multi_p.add_argument("--train-n", type=int, default=800)
    multi_p.add_argument("--eval-n", type=int, default=300)
    multi_p.add_argument("--seed", type=int, default=0)
    multi_p.add_argument("--estimators", type=int, default=80)
    multi_p.add_argument("--min-separation-km", type=float, default=5.0)
    multi_p.add_argument("--report", required=True, help="output markdown report path")

    event_p = sub.add_parser(
        "multi-event-detect",
        help="changepoint-scan-and-classify pipeline: detect and classify both events on a "
        "two-fault trace, instead of only measuring degradation on the labeled one",
    )
    event_p.add_argument("--train-n", type=int, default=800)
    event_p.add_argument("--n-traces", type=int, default=300)
    event_p.add_argument("--seed", type=int, default=0)
    event_p.add_argument("--estimators", type=int, default=100)
    event_p.add_argument("--min-separation-km", type=float, default=5.0)
    event_p.add_argument("--match-tolerance-km", type=float, default=3.0)
    event_p.add_argument("--report", required=True, help="output markdown report path")

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
        if args.report:
            os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
            with open(args.report, "w", encoding="utf-8") as handle:
                handle.write(render_markdown_report(result, sample_count=len(X)))
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

    if args.command == "stress":
        results = run_domain_shift_stress(
            train_n=args.train_n,
            test_n=args.test_n,
            seed=args.seed,
            n_estimators=args.estimators,
        )
        report = render_stress_report(results)
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as handle:
            handle.write(report)
        print(report)
        return 0

    if args.command == "multi-fault-stress":
        result = run_multi_fault_stress(
            train_n=args.train_n,
            eval_n=args.eval_n,
            seed=args.seed,
            n_estimators=args.estimators,
            min_separation_km=args.min_separation_km,
        )
        report = render_multi_fault_report(result)
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as handle:
            handle.write(report)
        print(report)
        return 0

    if args.command == "multi-event-detect":
        result = run_multi_event_detection(
            train_n=args.train_n,
            n_traces=args.n_traces,
            seed=args.seed,
            n_estimators=args.estimators,
            min_separation_km=args.min_separation_km,
            match_tolerance_km=args.match_tolerance_km,
        )
        report = render_multi_event_report(result)
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as handle:
            handle.write(report)
        print(report)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
