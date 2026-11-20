"""
Leak-free, subject-grouped nested cross-validation for the EEG-ADHD features.

Design
  * Outer loop: StratifiedGroupKFold(5) over SUBJECTS, repeated with several seeds.
    Every subject is tested exactly once per repeat, never seen in training.
  * Inner loop (hyper-parameter tuning): GroupKFold(3) over the training subjects only.
    The outer test fold is never used for any choice.
  * Scaler lives inside a Pipeline, so it is re-fit on each training split.
  * Primary metrics are SUBJECT-level: per-subject mean of out-of-fold scores
    -> AUC and balanced accuracy over 121 independent subjects, with a 95%
    bootstrap CI (resampling subjects). Epoch-level metrics are reported for reference.
  * Feature sets, models and grids are fixed in advance; nothing is chosen
    by looking at test results.
  * A label-permutation test (labels shuffled per SUBJECT) gives the chance distribution.
  * A deliberately leaky epoch-level split is run once to show how much it inflates scores.

Usage:  python3 clean_pipeline/evaluate.py [--repeats 3] [--perms 100]
Output: clean_pipeline/output/{results.csv, subject_predictions.csv, permutation.csv, leaky_contrast.csv}
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import (GridSearchCV, GroupKFold, StratifiedGroupKFold,
                                     StratifiedKFold)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")

CH = ["Fz", "Cz", "Pz", "C3", "T3", "C4", "T4", "Fp1", "Fp2", "F3",
      "F4", "F7", "F8", "P3", "P4", "T5", "T6", "O1", "O2"]


def cols(prefixes):
    return [f"{p}_{c}" for p in prefixes for c in CH]


# Feature sets mirror the original experiments (with duplicates removed).
FEATURE_SETS = {
    "full (5 bands + TBR + HFD)": cols(["delta", "theta", "alpha", "beta", "gamma", "tbr", "hfd"]),
    "theta/beta ratio": cols(["tbr"]),
    "alpha power": cols(["alpha"]),
    "slow-wave power (1-8 Hz)": cols(["slow"]),
    "old 'slow' = 12.5 Hz+ power": cols(["fast12p5"]),
}


N_JOBS = 2       # set from --n-jobs; keeps CPU/RAM load moderate on a laptop
N_TREES = 200


def models():
    return {
        "RandomForest": (
            RandomForestClassifier(n_estimators=N_TREES, max_features="sqrt",
                                   class_weight="balanced", random_state=42, n_jobs=1),
            {"clf__max_depth": [5, 10, None], "clf__min_samples_leaf": [2, 20]}),
        "SVM-RBF": (
            SVC(kernel="rbf", gamma="scale", class_weight="balanced", cache_size=1000),
            {"clf__C": [0.1, 1, 10]}),
        "KNN": (
            KNeighborsClassifier(),
            {"clf__n_neighbors": [5, 15, 45, 135], "clf__weights": ["uniform", "distance"]}),
    }


def score_of(est, X):
    """Continuous score with 0.5 (proba) or 0 (decision) as the natural threshold."""
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1], 0.5
    return est.decision_function(X), 0.0


def run_fold(X, y, groups, tr, te, clf, grid, n_jobs_inner):
    pipe = Pipeline([("scale", StandardScaler()), ("clf", clf)])
    gs = GridSearchCV(pipe, grid, cv=GroupKFold(3), scoring="roc_auc", n_jobs=n_jobs_inner)
    gs.fit(X[tr], y[tr], groups=groups[tr])
    s, thr = score_of(gs.best_estimator_, X[te])
    return te, s, thr, gs.best_params_


def subject_metrics(df, thr):
    subj = df.groupby("subject").agg(label=("label", "first"), score=("score", "mean"))
    auc = roc_auc_score(subj.label, subj.score)
    bacc = balanced_accuracy_score(subj.label, (subj.score > thr).astype(int))
    return subj, auc, bacc


def bootstrap_ci(subj, thr, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y, s = subj.label.values, subj.score.values
    aucs, baccs = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        aucs.append(roc_auc_score(y[i], s[i]))
        baccs.append(balanced_accuracy_score(y[i], (s[i] > thr).astype(int)))
    return np.percentile(aucs, [2.5, 97.5]), np.percentile(baccs, [2.5, 97.5])


def nested_cv(feats, fcols, name, clf, grid, seed):
    X = feats[fcols].values
    y = feats.label.values
    g = feats.subject.values
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    folds = Parallel(n_jobs=N_JOBS)(
        delayed(run_fold)(X, y, g, tr, te, clf, grid, 1) for tr, te in outer.split(X, y, g))
    score = np.empty(len(y))
    params = []
    for te, s, thr, p in folds:
        score[te] = s
        params.append(p)
    # sanity: no subject in both train and test of any fold
    for tr, te in outer.split(X, y, g):
        assert not set(g[tr]) & set(g[te])
    return score, thr, params


def permutation_test(feats, fcols, n_perm, seed=0):
    """Shuffle labels at the SUBJECT level and rerun a fixed (untuned) RF grouped CV."""
    X = feats[fcols].values
    g = feats.subject.values
    subj_lab = feats.groupby("subject").label.first()

    def one(i):
        rng = np.random.default_rng(seed + i)
        perm = pd.Series(rng.permutation(subj_lab.values), index=subj_lab.index)
        y = feats.subject.map(perm).values if i > 0 else feats.label.values
        score = np.empty(len(y))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, g):
            rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=20,
                                        class_weight="balanced", random_state=0, n_jobs=1)
            rf.fit(X[tr], y[tr])
            score[te] = rf.predict_proba(X[te])[:, 1]
        d = pd.DataFrame({"subject": g, "label": y, "score": score})
        return subject_metrics(d, 0.5)[1]

    aucs = Parallel(n_jobs=N_JOBS)(delayed(one)(i) for i in range(n_perm + 1))
    return aucs[0], np.array(aucs[1:])


def leaky_contrast(feats, fcols):
    """What the old ADHD-DA.ipynb did: random epoch-level split (subjects on both sides)."""
    X, y, g = feats[fcols].values, feats.label.values, feats.subject.values
    out = {}
    for name, cv in [("LEAKY epoch-level StratifiedKFold", StratifiedKFold(5, shuffle=True, random_state=0)),
                     ("correct subject-grouped StratifiedGroupKFold", StratifiedGroupKFold(5, shuffle=True, random_state=0))]:
        pred = np.empty(len(y))
        for tr, te in cv.split(X, y, g):
            pipe = Pipeline([("s", StandardScaler()), ("c", SVC(C=1, class_weight="balanced"))])
            pipe.fit(X[tr], y[tr])
            pred[te] = pipe.predict(X[te])
        out[name] = balanced_accuracy_score(y, pred)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--perms", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--quick", action="store_true",
                    help="non-overlapping epochs only (every 2nd 50%%-overlap epoch) and 100 trees")
    args = ap.parse_args()
    global N_JOBS, N_TREES
    N_JOBS = args.n_jobs

    feats = pd.read_csv(os.path.join(OUT, "features.csv"))
    if args.quick:
        # Adjacent 50%-overlap epochs share half their samples; keeping every other one
        # gives non-overlapping 2 s segments with ~no loss of information, at half the cost.
        feats = feats[feats.epoch % 2 == 0].reset_index(drop=True)
        N_TREES = 100
    num = feats.columns.difference(["subject", "label", "epoch"])
    feats[num] = feats[num].astype(np.float32)
    print(f"{feats.subject.nunique()} subjects, {len(feats)} epochs", flush=True)

    rows, preds = [], []
    for fs_name, fcols in FEATURE_SETS.items():
        for m_name, (clf, grid) in models().items():
            t = time.time()
            reps = []
            for r in range(args.repeats):
                score, thr, params = nested_cv(feats, fcols, m_name, clf, grid, seed=r)
                d = feats[["subject", "label"]].assign(score=score)
                subj, auc, bacc = subject_metrics(d, thr)
                ep_auc = roc_auc_score(d.label, d.score)
                ep_bacc = balanced_accuracy_score(d.label, (d.score > thr).astype(int))
                reps.append((subj, auc, bacc, ep_auc, ep_bacc, params))
                preds.append(subj.assign(feature_set=fs_name, model=m_name, repeat=r).reset_index())
            # CI on repeat-averaged subject scores
            subj_mean = pd.concat([x[0] for x in reps]).groupby(level=0).agg(
                label=("label", "first"), score=("score", "mean"))
            ci_auc, ci_bacc = bootstrap_ci(subj_mean, thr)
            row = {
                "feature_set": fs_name, "model": m_name, "n_features": len(fcols),
                "subject_AUC": np.mean([x[1] for x in reps]),
                "subject_AUC_CI95": f"[{ci_auc[0]:.3f}, {ci_auc[1]:.3f}]",
                "subject_BalAcc": np.mean([x[2] for x in reps]),
                "subject_BalAcc_CI95": f"[{ci_bacc[0]:.3f}, {ci_bacc[1]:.3f}]",
                "subject_BalAcc_sd_over_repeats": np.std([x[2] for x in reps]),
                "epoch_AUC": np.mean([x[3] for x in reps]),
                "epoch_BalAcc": np.mean([x[4] for x in reps]),
                "chosen_params": str(sorted({str(p) for x in reps for p in x[5]})),
            }
            rows.append(row)
            print(f"{fs_name:32s} {m_name:12s} subjAUC={row['subject_AUC']:.3f} {row['subject_AUC_CI95']} "
                  f"subjBalAcc={row['subject_BalAcc']:.3f} epochBalAcc={row['epoch_BalAcc']:.3f} "
                  f"({time.time() - t:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUT, "results.csv"), index=False)
    pd.concat(preds).to_csv(os.path.join(OUT, "subject_predictions.csv"), index=False)

    print("\nLeaky vs correct split (SVM, full features, balanced accuracy):", flush=True)
    lk = leaky_contrast(feats, FEATURE_SETS["full (5 bands + TBR + HFD)"])
    for k, v in lk.items():
        print(f"  {k:45s} {v:.3f}")
    pd.Series(lk).to_csv(os.path.join(OUT, "leaky_contrast.csv"))

    if args.perms:
        print(f"\nPermutation test ({args.perms} subject-level label shuffles, RF, full features)...", flush=True)
        obs, null = permutation_test(feats, FEATURE_SETS["full (5 bands + TBR + HFD)"], args.perms)
        p = (1 + np.sum(null >= obs)) / (1 + len(null))
        print(f"  observed subject AUC={obs:.3f}; null mean={null.mean():.3f}, "
              f"95th pct={np.percentile(null, 95):.3f}; p={p:.4f}")
        pd.DataFrame({"null_auc": null}).assign(observed=obs, p_value=p).to_csv(
            os.path.join(OUT, "permutation.csv"), index=False)


if __name__ == "__main__":
    main()
