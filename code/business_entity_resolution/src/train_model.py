"""
train_model.py — train the matching classifier and tune the F0.5 threshold.

Trains gradient-boosted trees (XGBoost, LightGBM) + a logistic baseline on the
featurized candidate pairs, then evaluates them the way the challenge does:
**macro F0.5 per Source-1 entity**, sweeping the decision threshold on the
realistic-base-rate validation set. Blocking-missed true matches are counted as
forced false negatives (recall is against the FULL ground-truth set, not just the
candidates), so the reported F0.5 is honest end-to-end.

Usage:
    python .../train_model.py --train data/interim/train_feats.parquet \
        --val data/interim/val_feats.parquet --gt dataset/train/train_ground_truth.tsv \
        --outdir models
"""
from __future__ import annotations
import argparse, os, sys, time, json
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FEATURE_NAMES  # noqa: E402

PROC = "data/processed"


def load_xy(path):
    t = pq.read_table(path)
    y = t.column("label").to_numpy()
    X = np.column_stack([t.column(n).to_numpy() for n in FEATURE_NAMES]).astype(np.float32)
    return X, y, t


def f05_matrix(s1_idx, scores, labels, total_true, n_groups, thresholds):
    """Return F_all[thr_i, group] = per-group F0.5 at each threshold, plus the
    per-group precision/recall matrices (for reporting)."""
    lab_mask = labels == 1
    singleton = total_true == 0
    F_all = np.empty((len(thresholds), n_groups), dtype=np.float64)
    P_all = np.empty_like(F_all); R_all = np.empty_like(F_all)
    for i, thr in enumerate(thresholds):
        pm = scores >= thr
        pred = np.bincount(s1_idx[pm], minlength=n_groups).astype(np.float64)
        tp = np.bincount(s1_idx[pm & lab_mask], minlength=n_groups).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            P = np.where(pred > 0, tp / pred, 0.0)
            R = np.where(total_true > 0, tp / total_true, 0.0)
            denom = 0.25 * P + R
            F = np.where(denom > 0, 1.25 * P * R / denom, 0.0)
        F = np.where(singleton, np.where(pred == 0, 1.0, 0.0), F)
        F_all[i] = F; P_all[i] = P; R_all[i] = R
    return F_all, P_all, R_all


def evaluate(name, scores, val_tbl, gt_full, s1_country_map, thresholds):
    s1_ids = val_tbl.column("s1").to_pylist()
    labels = val_tbl.column("label").to_numpy()
    # group ids
    uniq = {}
    s1_idx = np.empty(len(s1_ids), dtype=np.int64)
    for i, s in enumerate(s1_ids):
        gi = uniq.get(s)
        if gi is None:
            gi = len(uniq); uniq[s] = gi
        s1_idx[i] = gi
    n_groups = len(uniq)
    total_true = np.zeros(n_groups, dtype=np.float64)
    group_country = np.zeros(n_groups, dtype=np.int64)
    CC = {"US": 0, "India": 1, "France": 2}
    for s, gi in uniq.items():
        total_true[gi] = len(gt_full.get(s, ()))
        group_country[gi] = CC.get(s1_country_map.get(s, "?"), 3)
    F_all, P_all, R_all = f05_matrix(s1_idx, scores, labels, total_true,
                                     n_groups, thresholds)
    inv = {0: "US", 1: "India", 2: "France", 3: "?"}
    # --- global single threshold ---
    macro_by_thr = F_all.mean(axis=1)
    gi = int(np.argmax(macro_by_thr))
    g_thr = float(thresholds[gi]); g_macro = float(macro_by_thr[gi])
    # P/R at global best
    gP = P_all[gi][total_true > 0].mean(); gR = R_all[gi][total_true > 0].mean()
    # --- per-country thresholds ---
    per_country_thr = {}
    F_best_per_group = np.empty(n_groups)
    for cc in np.unique(group_country):
        m = group_country == cc
        ci = int(np.argmax(F_all[:, m].mean(axis=1)))
        per_country_thr[inv[int(cc)]] = (float(thresholds[ci]), float(F_all[ci, m].mean()))
        F_best_per_group[m] = F_all[ci, m]
    pc_macro = float(F_best_per_group.mean())
    pcs = "  ".join(f"{k}:thr={v[0]:.3f}/F={v[1]:.4f}" for k, v in per_country_thr.items())
    print(f"[{name}] GLOBAL F0.5={g_macro:.4f} @thr={g_thr:.3f} (P={gP:.3f} R={gR:.3f}) | "
          f"PER-COUNTRY F0.5={pc_macro:.4f}", flush=True)
    print(f"        per-country: {pcs}", flush=True)
    return max(g_macro, pc_macro), g_thr, per_country_thr, n_groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--outdir", default="models")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

    Xtr, ytr, _ = load_xy(args.train)
    log(f"train X={Xtr.shape} pos={ytr.mean():.3%}")
    Xva, yva, val_tbl = load_xy(args.val)
    log(f"val   X={Xva.shape} pos={yva.mean():.3%}")

    # full GT match sets for val S1
    gt_full = {}
    with open(args.gt, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            gt_full[s1] = set(rest.split(",")) if rest.strip() else set()
    # S1 -> country
    s1c = {}
    tc = pq.read_table(f"{PROC}/train_source1.parquet", columns=["entity_id", "country"])
    for e, c in zip(tc.column("entity_id").to_pylist(), tc.column("country").to_pylist()):
        s1c[e] = c
    log("loaded GT + country map")

    # precision-heavy F0.5 favours high thresholds, so sample the top end finely
    thresholds = np.unique(np.concatenate([
        np.linspace(0.05, 0.95, 37),
        np.linspace(0.95, 0.999, 40),
    ]))
    results = {}
    pos_w = (ytr == 0).sum() / max(1, (ytr == 1).sum())

    # ---- XGBoost ----
    import xgboost as xgb
    dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=FEATURE_NAMES)
    params = dict(max_depth=8, eta=0.2, subsample=0.8, colsample_bytree=0.8,
                  objective="binary:logistic", eval_metric="aucpr",
                  scale_pos_weight=pos_w, tree_method="hist", nthread=0)
    t = time.time()
    bst = xgb.train(params, dtr, num_boost_round=300)
    log(f"XGBoost trained in {time.time()-t:.0f}s")
    sc = bst.predict(xgb.DMatrix(Xva, feature_names=FEATURE_NAMES))
    macro, thr, curve, ng = evaluate("XGBoost", sc, val_tbl, gt_full, s1c, thresholds)
    results["xgboost"] = (macro, thr)
    bst.save_model(f"{args.outdir}/xgb.json")
    imp = bst.get_score(importance_type="gain")
    top = sorted(imp.items(), key=lambda x: -x[1])[:12]
    print("  top features (gain):", ", ".join(f"{k}={v:.0f}" for k, v in top), flush=True)

    # ---- LightGBM ----
    import lightgbm as lgb
    dtr = lgb.Dataset(Xtr, label=ytr, feature_name=list(FEATURE_NAMES))
    lp = dict(objective="binary", metric="average_precision", num_leaves=127,
              learning_rate=0.1, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, scale_pos_weight=pos_w, num_threads=0, verbose=-1)
    t = time.time()
    gbm = lgb.train(lp, dtr, num_boost_round=300)
    log(f"LightGBM trained in {time.time()-t:.0f}s")
    sc = gbm.predict(Xva)
    macro, thr, curve, ng = evaluate("LightGBM", sc, val_tbl, gt_full, s1c, thresholds)
    results["lightgbm"] = (macro, thr)
    gbm.save_model(f"{args.outdir}/lgbm.txt")

    # ---- Logistic baseline ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sca = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=200, C=1.0, class_weight="balanced")
    lr.fit(sca.transform(Xtr), ytr)
    sc = lr.predict_proba(sca.transform(Xva))[:, 1]
    macro, thr, curve, ng = evaluate("Logistic", sc, val_tbl, gt_full, s1c, thresholds)
    results["logistic"] = (macro, thr)

    print("\n===== MODEL BAKE-OFF (macro F0.5, val) =====", flush=True)
    for k, (m, th) in sorted(results.items(), key=lambda x: -x[1][0]):
        print(f"  {k:10s} F0.5={m:.4f}  thr={th:.3f}", flush=True)
    best = max(results.items(), key=lambda x: x[1][0])
    print(f"\nBEST: {best[0]} (F0.5={best[1][0]:.4f})", flush=True)
    json.dump({"results": results, "n_val_groups": ng, "best": best[0]},
              open(f"{args.outdir}/bakeoff.json", "w"), indent=2)


if __name__ == "__main__":
    main()
