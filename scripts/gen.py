"""Generative training of the multi universal approximator (one qubit per pixel).

    python scripts/gen.py clic --pixels 8 --depth 8 ...
    python scripts/gen.py mnist --depth 16 ...

Writes results/<dataset>/gen_<name>.json: configuration, per-step batch MMD, held-out MMD
and correlation distance per epoch, final parameters and samples generated after training
next to the same number of test samples. Defaults per study are in PRESETS (--preset).
"""
import argparse

import _common  # noqa: F401  (sets sys.path)
import numpy as np

from mua import data as datasets  # noqa: E402
from mua.scalers import Expectations, Linear  # noqa: E402
from mua.train import gen  # noqa: E402
from mua.utils import save_json, to_list  # noqa: E402

PRESETS = {
    # CLIC 8 px: as notebooks/clic.ipynb (depth 8, 50 epochs, batch 2000, lr 5e-3)
    "clic_q8_d8": dict(dataset="clic", pixels=8, depth=8, epochs=50, batch=2000, lr=5e-3, w_scale=1.0),
    # CLIC 25 px, depth 22: w ~ U[0,1) gives a flat MMD (angles ~ 5 rad over 22 noise inputs)
    "clic_q25_d22": dict(dataset="clic", pixels=25, depth=22, epochs=25, batch=1000, lr=2e-3, w_scale=0.1),
    # full MNIST (784 qubits)
    "mnist_d16": dict(dataset="mnist", depth=16, epochs=40, batch=500, lr=5e-3, w_scale=0.1),
    "mnist_d32": dict(dataset="mnist", depth=32, epochs=60, batch=500, lr=3e-3, w_scale=0.05),
    # long runs, cosine decay (GPU: ~1 s / epoch at depth 32 on a P100), stop cleanly after max_hours
    "mnist_d32_long": dict(dataset="mnist", depth=32, epochs=1800, batch=500, lr=3e-3,
                           w_scale=0.05, schedule="cosine", max_hours=60),
    "mnist_d64_long": dict(dataset="mnist", depth=64, epochs=1800, batch=500, lr=2e-3,
                           w_scale=0.025, schedule="cosine", max_hours=20),
}


def load(args):
    """Encoded train / test sets, decoded test set, scaler, labels and dataset info."""
    if args.dataset == "clic":
        train, test = datasets.load_clic(args.pixels, "train"), datasets.load_clic(args.pixels, "test")
        scaler, labels = Expectations().fit(train), None
        info = {"dataset": f"CLIC ECAL, {args.pixels} px", "source": str(datasets.CLIC_DIR),
                "shape": [args.pixels]}
    else:
        (train, _), (test, labels) = datasets.load_mnist("train"), datasets.load_mnist("test")
        scaler = Linear(0, 1)
        info = {"dataset": "MNIST, full 28x28, all digits, unconditional",
                "source": str(datasets.MNIST_FILE), "shape": [28, 28]}
    return scaler.encode(train), scaler.encode(test), test, scaler, labels, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["clic", "mnist"], nargs="?")
    ap.add_argument("--preset", choices=list(PRESETS))
    ap.add_argument("--pixels", type=int)
    ap.add_argument("--depth", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--w-scale", dest="w_scale", type=float)
    ap.add_argument("--bandwidth-factors", type=float, nargs="+", default=[0.25, 0.5, 1, 2, 4])
    ap.add_argument("--eval-size", type=int, default=1000)
    ap.add_argument("--num-samples", type=int)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--schedule", choices=["constant", "cosine"])
    ap.add_argument("--max-hours", dest="max_hours", type=float)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--name")
    args = ap.parse_args()
    if args.preset:  # preset values unless given explicitly
        for k, v in PRESETS[args.preset].items():
            if getattr(args, k, None) is None:
                setattr(args, k, v)
        args.name = args.name or args.preset
    args.schedule = args.schedule or "constant"
    args.num_samples = args.num_samples or (5000 if args.dataset == "clic" else 1000)
    tag = f"q{args.pixels}_d{args.depth}" if args.dataset == "clic" else f"d{args.depth}"
    args.name = args.name or f"{args.dataset}_{tag}"

    enc_train, enc_test, test, scaler, labels, info = load(args)
    print(f"{args.name}: train {enc_train.shape}, test {enc_test.shape}", flush=True)

    path = _common.RESULTS / args.dataset / f"gen_{args.name}.json"

    def save(params, h, complete):
        rng = np.random.default_rng(args.seed + 1)
        enc_fake = gen.generate(params, args.num_samples, rng)
        idx = rng.choice(len(test), args.num_samples, replace=False)  # comparison test samples
        num_params = int(sum(np.prod(v.shape) for v in params.values()))
        digits = 4 if args.dataset == "mnist" else 7
        out = {
            "name": args.name, **info, "num_qubits": enc_train.shape[1], "depth": args.depth,
            "noise": "z ~ U[0,1)^depth, shared by all qubits", "num_params": num_params,
            "num_epochs": h["epoch"][-1], "planned_epochs": args.epochs, "complete": complete,
            "batch": args.batch, "lr": args.lr, "lr_schedule": args.schedule,
            "optimizer": "adam",
            "init": f"phi, alpha ~ U[0,1), w ~ {args.w_scale} U[0,1)", "w_scale": args.w_scale,
            "loss": "MMD^2 (biased), Gaussian kernels exp(-d^2/(2h^2)) averaged over h, encoded space",
            "bandwidth_factors": args.bandwidth_factors, "bandwidths": h["bandwidths"],
            "seed": args.seed, "scaler": scaler.state(),
            "note": ("epoch = one pass over the training set, one Adam step per batch; step_loss = "
                     "batch MMD of every step; mmd, corr = held-out test set (eval_size samples, "
                     "fixed noise) after every epoch, epoch 0 = initialisation; encoded space. "
                     "lr_schedule cosine: decay from lr to lr/100 over planned_epochs."),
            "eval_size": args.eval_size,
            "history": {"epoch": h["epoch"], "mmd": h["mmd"], "corr": h["corr"], "time": h["time"],
                        "step_loss": to_list(h["step_loss"], 8)},
            "final_params": {k: to_list(v) for k, v in params.items()},
            "samples": {"fake": to_list(scaler.decode(enc_fake), digits),
                        "fake_encoded": to_list(enc_fake, digits),
                        "real": to_list(test[idx], digits)},
            "provenance": _common.provenance(),
        }
        if labels is not None:
            out["samples"]["real_labels"] = np.asarray(labels)[idx].tolist()
        save_json(out, path)

    params, h = gen.train(enc_train, args.depth, args.epochs, args.batch, args.lr,
                          bandwidth_factors=args.bandwidth_factors, w_scale=args.w_scale,
                          test_data=enc_test, eval_size=args.eval_size, seed=args.seed,
                          progress_bar=False, schedule=args.schedule,
                          max_time=args.max_hours * 3600 if args.max_hours else None,
                          callback=lambda p, hh: save(p, hh, complete=False),
                          callback_every=args.checkpoint_every)
    save(params, h, complete=True)


if __name__ == "__main__":
    main()
