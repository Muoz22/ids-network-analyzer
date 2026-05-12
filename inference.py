# ================================================================
# inference.py — AI Agents IDS (ONNX version) v3
# ================================================================

import pickle, json, os
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
    precision_recall_curve, roc_curve, auc)
from datetime import datetime

_MODEL_DIR = "models/"


def load_models(model_dir: str = "models/"):
    global _MODEL_DIR
    _MODEL_DIR = model_dir
    import onnxruntime as ort

    models = {}

    meta_path = os.path.join(model_dir, "metadata_v3.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        models["meta"]        = meta
        models["features"]    = meta["feat_cols"]
        models["class_names"] = meta["class_names"]
        models["n_features"]  = meta["n_features"]
        print(f"✅ metadata_v3.json loaded")
        print(f"   features: {meta['feat_cols']}")
        print(f"   classes : {meta['class_names']}")
    else:
        with open(os.path.join(
                model_dir, "selected_features.json")) as f:
            feats = json.load(f)
        seen = set()
        models["features"] = [
            x for x in feats
            if not (x in seen or seen.add(x))]
        with open(os.path.join(
                model_dir, "class_names.json")) as f:
            models["class_names"] = json.load(f)
        models["n_features"] = len(models["features"])

    sess = ort.InferenceSession(
        os.path.join(model_dir, "model.onnx"))
    models["session"]     = sess
    models["input_name"]  = sess.get_inputs()[0].name
    models["output_name"] = sess.get_outputs()[0].name
    print(f"✅ ONNX loaded: "
          f"input={sess.get_inputs()[0].shape}")

    with open(os.path.join(model_dir, "scaler.pkl"),
              "rb") as f:
        models["scaler"] = pickle.load(f)
    print(f"✅ Scaler: "
          f"{models['scaler'].n_features_in_} features")

    hist_path = os.path.join(
        model_dir, "training_history.json")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            models["training_history"] = json.load(f)
        print(f"✅ training_history.json — "
              f"{len(models['training_history']['loss'])}"
              f" epochs")

    return models


def auto_exclude(df, label_col, benign_label):
    df = df.replace(
        [float('inf'), float('-inf')], float('nan'))
    obj_cols = [c for c in df.select_dtypes(
        'object').columns if c != label_col]
    if obj_cols:
        df = df.drop(columns=obj_cols)

    excl    = [label_col]
    removed = {}

    if label_col in df.columns:
        df["_lb"] = (
            df[label_col] != benign_label).astype(int)
        excl.append("_lb")

    num_df = df.select_dtypes(
        include=[float, int, "int64", "float64"])

    if "_lb" in df.columns:
        for col in num_df.columns:
            if col in excl: continue
            uv = set(df[col].dropna().unique())
            if uv.issubset({0, 1, 0.0, 1.0}):
                corr = abs(df[col].corr(df["_lb"]))
                if corr > 0.95:
                    excl.append(col)
                    removed[col] = f"leakage({corr:.3f})"

    for col in num_df.columns:
        if col in excl: continue
        if df[col].median() > 1e9:
            excl.append(col)
            removed[col] = "timestamp"

    for col in num_df.columns:
        if col in excl: continue
        sv = df[col].dropna().sort_values(
            ).reset_index(drop=True)
        if len(sv) > 100:
            diffs = sv.diff().dropna()
            if (len(diffs) > 0 and
                    (diffs >= 0).mean() > 0.99 and
                    diffs.std() < 0.01 and
                    diffs.mean() > 0):
                excl.append(col)
                removed[col] = "sequential_id"

    for col in num_df.columns:
        if col in excl: continue
        if df[col].std() == 0:
            excl.append(col)
            removed[col] = "zero_var"

    for col in num_df.columns:
        if col in excl: continue
        vc = df[col].value_counts(normalize=True)
        if len(vc) > 0 and vc.iloc[0] > 0.999:
            excl.append(col)
            removed[col] = "near_const"

    if "_lb" in df.columns:
        df = df.drop(columns=["_lb"])

    avail = [c for c in df.columns if c not in excl]
    df[avail] = df[avail].fillna(df[avail].median())
    return df, avail, removed


def align_features(df, avail, features, n_model):
    result  = np.zeros(
        (len(df), n_model), dtype=np.float32)
    matched = []
    missing = []
    for i, feat in enumerate(features[:n_model]):
        if feat in avail and feat in df.columns:
            result[:, i] = df[feat].values.astype(
                np.float32)
            matched.append(feat)
        else:
            missing.append(feat)
    return result, matched, missing


def run_inference(df, models, label_col,
                  benign_label, ft_unk_thr=0.60):
    t0 = datetime.now()

    df_clean, avail, removed = auto_exclude(
        df, label_col, benign_label)

    n_model = models["n_features"]
    X_raw, matched, missing = align_features(
        df_clean, avail, models["features"], n_model)

    n_sc = models["scaler"].n_features_in_
    if X_raw.shape[1] < n_sc:
        Xp = np.zeros(
            (len(X_raw), n_sc), dtype=np.float32)
        Xp[:, :X_raw.shape[1]] = X_raw
        X_sc = models["scaler"].transform(Xp)
    elif X_raw.shape[1] > n_sc:
        X_sc = models["scaler"].transform(
            X_raw[:, :n_sc])
    else:
        X_sc = models["scaler"].transform(X_raw)

    X_final = X_sc[:, :n_model].astype(np.float32)

    batch_size = 4096
    all_probs  = []
    for i in range(0, len(X_final), batch_size):
        batch = X_final[i:i+batch_size]
        probs = models["session"].run(
            [models["output_name"]],
            {models["input_name"]: batch})[0]
        all_probs.append(probs)

    y_probs = np.vstack(all_probs)
    y_pred  = np.argmax(y_probs, axis=1)
    y_conf  = y_probs.max(axis=1)
    y_unk   = y_conf < ft_unk_thr

    y_pred_names = [
        models["class_names"][i]
        if i < len(models["class_names"]) else "?"
        for i in y_pred]

    total   = len(y_pred_names)
    benign  = sum(1 for p in y_pred_names
                  if p == benign_label)
    unknown = int(y_unk.sum())
    attacks = total - benign - unknown

    atk_counts = Counter(
        p for p, u in zip(y_pred_names, y_unk)
        if p != benign_label and not u)

    metrics = {}
    if label_col in df.columns:
        y_true = df[label_col].values[:total]
        try:
            metrics["accuracy"]    = accuracy_score(
                y_true, y_pred_names)
            metrics["weighted_f1"] = f1_score(
                y_true, y_pred_names,
                average="weighted", zero_division=0)
            metrics["macro_f1"]    = f1_score(
                y_true, y_pred_names,
                average="macro", zero_division=0)
            metrics["report"]      = classification_report(
                y_true, y_pred_names,
                labels=sorted(
                    set(y_true)|set(y_pred_names)),
                zero_division=0)
            metrics["cm_true"]     = y_true
            metrics["y_true"]      = y_true
        except Exception as e:
            metrics["error"] = str(e)

    elapsed = (datetime.now() - t0).seconds

    return {
        "y_pred"       : y_pred_names,
        "y_probs"      : y_probs,
        "y_conf"       : y_conf,
        "y_unknown"    : y_unk,
        "n_samples"    : total,
        "n_benign"     : benign,
        "n_attacks"    : attacks,
        "n_unknown"    : unknown,
        "atk_counts"   : atk_counts,
        "metrics"      : metrics,
        "removed_cols" : removed,
        "matched_feats": matched,
        "missing_feats": missing,
        "elapsed_sec"  : elapsed,
        "X_final"      : X_final,
        "ft_unk_thr"   : ft_unk_thr,
        # ← feat_names من النموذج الأصلي
        "feat_names"   : models["features"],
    }


def _compute_shap_values(X, y_probs, feat_names):
    n_feat   = X.shape[1]
    max_prob = y_probs.max(axis=1)

    shap_vals = np.zeros(n_feat)
    shap_sign = np.zeros(n_feat)

    for i in range(n_feat):
        feat_col = X[:, i]
        if feat_col.std() > 1e-8:
            corr = np.corrcoef(
                feat_col, max_prob)[0, 1]
            if np.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0

        mean_f = feat_col.mean()
        dev    = np.abs(feat_col - mean_f)
        impact = np.mean(dev * np.abs(
            max_prob - max_prob.mean()))

        shap_vals[i] = impact
        shap_sign[i] = np.sign(corr)

    total = shap_vals.sum()
    if total > 0:
        shap_vals = shap_vals / total

    return shap_vals, shap_sign


def make_plots(results, benign_label,
               out_dir="plots/",
               models_ref=None):
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    y_pred = results["y_pred"]
    y_conf = results["y_conf"]
    y_unk  = results["y_unknown"]
    total  = results["n_samples"]
    benign = results["n_benign"]
    atks   = results["n_attacks"]
    unk    = results["n_unknown"]
    thr    = results.get("ft_unk_thr", 0.60)
    m      = results["metrics"]

    plt.rcParams.update({
        "figure.dpi": 120, "font.size": 11,
        "figure.facecolor": "white"})

    # ── 1. Decision Pie ───────────────────────────────────
    try:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.pie(
            [max(s,1) for s in [benign, atks, unk]],
            labels=["Benign","Attack","Unknown"],
            colors=["#2ecc71","#e74c3c","#f39c12"],
            autopct="%1.1f%%", startangle=140,
            explode=(0.02,0.05,0.05), shadow=True)
        ax.set_title(
            f"Decision Distribution\nTotal: {total:,}",
            fontweight="bold")
        p = os.path.join(out_dir, "pie.png")
        fig.savefig(p, bbox_inches="tight")
        plt.close()
        paths.append(("Decision Distribution", p))
    except Exception:
        pass

    # ── 2. Attack Bar ─────────────────────────────────────
    try:
        if results["atk_counts"]:
            fig, ax = plt.subplots(figsize=(10, 6))
            top  = results["atk_counts"].most_common(15)
            nms  = [x[0] for x in top]
            vals = [x[1] for x in top]
            cols = plt.cm.Reds_r(
                [0.3+0.5*(i/max(len(nms),1))
                 for i in range(len(nms))])
            ax.barh(nms, vals, color=cols, alpha=0.85)
            for i, v in enumerate(vals):
                ax.text(v+max(vals)*0.01, i,
                        f"{v:,}", va="center",
                        fontsize=9)
            ax.set_xlabel("Count")
            ax.set_title("Top Attack Types",
                         fontweight="bold")
            ax.set_xlim(0, max(vals)*1.15)
            plt.tight_layout()
            p = os.path.join(out_dir, "attacks.png")
            fig.savefig(p, bbox_inches="tight")
            plt.close()
            paths.append(("Attack Types", p))
    except Exception:
        pass

    # ── 3. Confidence ─────────────────────────────────────
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        n = min(3000, len(y_conf))
        c = ["#2ecc71" if p==benign_label
             else "#e74c3c" for p in y_pred[:n]]
        axes[0].scatter(range(n), y_conf[:n],
                        c=c, alpha=0.4, s=6)
        axes[0].axhline(thr, color="black", ls="--",
                        lw=1.5,
                        label=f"Unknown thr={thr}")
        axes[0].set_xlabel("Sample Index")
        axes[0].set_ylabel("Confidence")
        axes[0].set_title("Confidence Timeline",
                          fontweight="bold")
        axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].hist(y_conf[~y_unk], bins=50,
                     color="#3498db", alpha=0.7,
                     label="Known", density=True)
        axes[1].hist(y_conf[y_unk], bins=20,
                     color="#e74c3c", alpha=0.7,
                     label="Unknown", density=True)
        axes[1].axvline(thr, color="black",
                        ls="--", lw=1.5)
        axes[1].set_xlabel("Confidence")
        axes[1].set_title("Confidence Distribution",
                          fontweight="bold")
        axes[1].legend(); axes[1].grid(alpha=0.3)
        plt.suptitle("Confidence Analysis",
                     fontweight="bold")
        plt.tight_layout()
        p = os.path.join(out_dir, "confidence.png")
        fig.savefig(p, bbox_inches="tight")
        plt.close()
        paths.append(("Confidence", p))
    except Exception:
        pass

    # ── 4. Confusion Matrix ───────────────────────────────
    if "cm_true" in m:
        try:
            y_true  = m["cm_true"]
            present = sorted(
                set(y_true)|set(y_pred))[:15]
            cm = confusion_matrix(
                y_true, y_pred, labels=present)
            sz = max(7, len(present))
            fig, ax = plt.subplots(
                figsize=(sz+1, sz))
            sns.heatmap(
                cm, annot=True, fmt="d",
                cmap="Blues",
                xticklabels=present,
                yticklabels=present,
                annot_kws={"size":8}, ax=ax)
            ax.set_title("Confusion Matrix",
                         fontweight="bold")
            ax.set_ylabel("True")
            ax.set_xlabel("Predicted")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            p = os.path.join(out_dir, "cm.png")
            fig.savefig(p, bbox_inches="tight")
            plt.close()
            paths.append(("Confusion Matrix", p))
        except Exception:
            pass

    # ── Training Curves ───────────────────────────────────
    try:
        hist = None
        if models_ref and \
                "training_history" in models_ref:
            hist = models_ref["training_history"]
        else:
            hist_path = os.path.join(
                _MODEL_DIR, "training_history.json")
            if os.path.exists(hist_path):
                with open(hist_path) as f:
                    hist = json.load(f)

        if hist and len(hist.get("loss", [])) > 0:
            epochs = range(1, len(hist["loss"])+1)
            fig, axes = plt.subplots(
                1, 2, figsize=(14, 5))

            axes[0].plot(
                epochs, hist["loss"],
                color="#3498db", lw=2,
                label="Train Loss")
            axes[0].plot(
                epochs, hist["val_loss"],
                color="#e74c3c", lw=2, ls="--",
                label="Val Loss")
            best_ep = int(np.argmin(
                hist["val_loss"])) + 1
            best_vl = min(hist["val_loss"])
            axes[0].axvline(
                best_ep, color="gray",
                ls=":", lw=1.5,
                label=f"Best={best_ep}")
            axes[0].annotate(
                f"min={best_vl:.4f}",
                xy=(best_ep, best_vl),
                xytext=(best_ep+1, best_vl+0.005),
                fontsize=8, color="#e74c3c",
                arrowprops=dict(
                    arrowstyle="->",
                    color="#e74c3c"))
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].set_title(
                "Agent 4 — Training Loss",
                fontweight="bold")
            axes[0].legend()
            axes[0].grid(alpha=0.3)

            if hist.get("accuracy"):
                axes[1].plot(
                    epochs, hist["accuracy"],
                    color="#2ecc71", lw=2,
                    label="Train Accuracy")
                axes[1].plot(
                    epochs, hist["val_accuracy"],
                    color="#f39c12", lw=2, ls="--",
                    label="Val Accuracy")
                best_ep_a = int(np.argmax(
                    hist["val_accuracy"])) + 1
                best_va = max(hist["val_accuracy"])
                axes[1].axvline(
                    best_ep_a, color="gray",
                    ls=":", lw=1.5,
                    label=f"Best={best_ep_a}")
                axes[1].annotate(
                    f"max={best_va:.4f}",
                    xy=(best_ep_a, best_va),
                    xytext=(best_ep_a+1,
                            best_va-0.02),
                    fontsize=8, color="#f39c12",
                    arrowprops=dict(
                        arrowstyle="->",
                        color="#f39c12"))
                axes[1].set_xlabel("Epoch")
                axes[1].set_ylabel("Accuracy")
                axes[1].set_title(
                    "Agent 4 — Training Accuracy",
                    fontweight="bold")
                axes[1].legend()
                axes[1].grid(alpha=0.3)

            plt.suptitle(
                f"Agent 4 — Training Curves  "
                f"({len(hist['loss'])} epochs)",
                fontweight="bold")
            plt.tight_layout()
            p = os.path.join(
                out_dir, "training_curves.png")
            fig.savefig(p, bbox_inches="tight")
            plt.close()
            paths.append(("Training Curves", p))
    except Exception:
        pass

    # ── 5. Per-Class F1 ───────────────────────────────────
    if "cm_true" in m:
        try:
            y_true = m["cm_true"]
            labels = sorted(set(y_true)|set(y_pred))
            f1s = f1_score(
                y_true, y_pred,
                labels=labels, average=None,
                zero_division=0)
            fig, ax = plt.subplots(figsize=(12, 6))
            colors_f = ["#2ecc71" if f>=0.8
                        else "#f39c12" if f>=0.5
                        else "#e74c3c" for f in f1s]
            bars = ax.barh(labels, f1s,
                           color=colors_f, alpha=0.85)
            ax.axvline(0.8, color="gray",
                       ls="--", lw=1.5)
            for bar, val in zip(bars, f1s):
                ax.text(
                    val+0.01,
                    bar.get_y()+bar.get_height()/2,
                    f"{val:.2f}", va="center",
                    fontsize=9)
            ax.set_xlabel("F1 Score")
            ax.set_title("Per-Class F1 Score",
                         fontweight="bold")
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            p = os.path.join(out_dir, "f1.png")
            fig.savefig(p, bbox_inches="tight")
            plt.close()
            paths.append(("Per-Class F1", p))
        except Exception:
            pass

    # ── 6. Severity Distribution ──────────────────────────
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        high_sev = ["ddos","dos","ransomware",
                    "backdoor","mitm","DDoS","DoS",
                    "Ransomware","Backdoor","MITM"]
        n_high = sum(1 for p in y_pred
                     if any(h in p for h in high_sev)
                     and p != benign_label)
        n_med  = atks - n_high
        ax.pie(
            [max(benign,1), max(n_high,1),
             max(n_med,1)],
            labels=["None","High","Medium"],
            colors=["#2ecc71","#e74c3c","#f39c12"],
            autopct="%1.1f%%", startangle=140,
            shadow=True)
        ax.set_title("Severity Distribution",
                     fontweight="bold")
        plt.tight_layout()
        p = os.path.join(out_dir, "severity.png")
        fig.savefig(p, bbox_inches="tight")
        plt.close()
        paths.append(("Severity Distribution", p))
    except Exception:
        pass

    # ── 7. Class Distribution ─────────────────────────────
    try:
        class_counts = Counter(y_pred)
        top_classes  = [x[0] for x in
                        class_counts.most_common(10)]
        top_counts   = [class_counts[c]
                        for c in top_classes]
        fig, ax = plt.subplots(figsize=(10, 5))
        colors_c = ["#2ecc71" if c==benign_label
                    else "#e74c3c"
                    for c in top_classes]
        ax.bar(range(len(top_classes)),
               top_counts, color=colors_c,
               alpha=0.85)
        ax.set_xticks(range(len(top_classes)))
        ax.set_xticklabels(
            [c[:15] for c in top_classes],
            rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Count")
        ax.set_title(
            "Prediction Distribution by Class",
            fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        p = os.path.join(out_dir, "class_dist.png")
        fig.savefig(p, bbox_inches="tight")
        plt.close()
        paths.append(("Class Distribution", p))
    except Exception:
        pass

    # ── 8. PR Curve ───────────────────────────────────────
    if "y_true" in m:
        try:
            y_true   = m["y_true"]
            y_bin    = np.array(
                [0 if y==benign_label else 1
                 for y in y_true])
            y_scores = 1 - y_conf
            prec, rec, _ = precision_recall_curve(
                y_bin, y_scores)
            pr_auc = auc(rec, prec)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(rec, prec, color="#e74c3c",
                    lw=2,
                    label=f"PR (AUC={pr_auc:.3f})")
            ax.fill_between(rec, prec,
                            alpha=0.1,
                            color="#e74c3c")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(
                "Precision-Recall Curve",
                fontweight="bold")
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            p = os.path.join(out_dir, "pr_curve.png")
            fig.savefig(p, bbox_inches="tight")
            plt.close()
            paths.append(("PR Curve", p))
        except Exception:
            pass

    # ── 9. Summary Dashboard ──────────────────────────────
    try:
        fig = plt.figure(figsize=(18, 10))
        gs  = gridspec.GridSpec(2, 3, figure=fig,
                                hspace=0.4, wspace=0.35)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.pie(
            [max(s,1) for s in [benign, atks, unk]],
            labels=["Benign","Attack","Unknown"],
            colors=["#2ecc71","#e74c3c","#f39c12"],
            autopct="%1.1f%%", startangle=140)
        ax1.set_title("Decisions", fontweight="bold")
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(y_conf, bins=50,
                 color="#9b59b6", alpha=0.7,
                 density=True)
        ax2.axvline(thr, color="red", ls="--", lw=1.5)
        ax2.set_title("Confidence Dist",
                      fontweight="bold")
        ax2.set_xlabel("Confidence")
        ax2.grid(alpha=0.3)
        ax3 = fig.add_subplot(gs[0, 2])
        if results["atk_counts"]:
            top5 = results["atk_counts"].most_common(5)
            nms5 = [x[0][:12] for x in top5]
            vls5 = [x[1] for x in top5]
            ax3.barh(nms5, vls5,
                     color="#e74c3c", alpha=0.85)
            ax3.set_title("Top 5 Attacks",
                          fontweight="bold")
            ax3.grid(axis="x", alpha=0.3)
        if "cm_true" in m:
            ax4  = fig.add_subplot(gs[1, :])
            y_tr = m["cm_true"]
            lbs  = sorted(set(y_tr)|set(y_pred))
            f1s  = f1_score(y_tr, y_pred,
                            labels=lbs, average=None,
                            zero_division=0)
            cols_f = ["#2ecc71" if f>=0.8
                      else "#f39c12" if f>=0.5
                      else "#e74c3c" for f in f1s]
            ax4.bar(range(len(lbs)), f1s,
                    color=cols_f, alpha=0.85)
            ax4.set_xticks(range(len(lbs)))
            ax4.set_xticklabels(
                [l[:12] for l in lbs],
                rotation=45, ha="right", fontsize=9)
            ax4.axhline(0.8, color="gray",
                        ls="--", lw=1.5)
            ax4.set_ylabel("F1 Score")
            ax4.set_title("Per-Class F1",
                          fontweight="bold")
            ax4.grid(axis="y", alpha=0.3)
        acc_str = (
            f"Acc={m['accuracy']*100:.1f}%  "
            f"W-F1={m['weighted_f1']*100:.1f}%  "
            f"Macro-F1={m['macro_f1']*100:.1f}%"
            if "accuracy" in m else "")
        plt.suptitle(
            f"Summary Dashboard  {acc_str}",
            fontweight="bold", fontsize=13)
        p = os.path.join(out_dir, "dashboard.png")
        fig.savefig(p, bbox_inches="tight")
        plt.close()
        paths.append(("Summary Dashboard", p))
    except Exception:
        pass

    # ── 10. Drift Analysis ────────────────────────────────
    try:
        from scipy.stats import ks_2samp
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ref = np.random.RandomState(42).normal(
            0.8, 0.1, len(y_conf))
        axes[0].hist(y_conf, bins=40,
                     color="#3498db", alpha=0.7,
                     label="Current", density=True)
        axes[0].hist(ref, bins=40,
                     color="#e74c3c", alpha=0.5,
                     label="Reference (Normal)",
                     density=True)
        axes[0].set_xlabel("Confidence Score")
        axes[0].set_ylabel("Density")
        axes[0].set_title(
            "Drift Analysis — Confidence Distribution",
            fontweight="bold")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        stat, p_val = ks_2samp(y_conf, ref)
        drifted     = p_val < 0.05
        drift_color = "#e74c3c" if drifted \
            else "#2ecc71"
        drift_text  = "DRIFT DETECTED ⚠️" if drifted \
            else "Stable ✅"
        axes[1].bar(
            ["KS Statistic", "P-Value"],
            [stat, p_val],
            color=[drift_color, drift_color],
            alpha=0.8)
        axes[1].axhline(0.05, color="gray",
                        ls="--", lw=1.5,
                        label="Threshold=0.05")
        axes[1].set_title(
            f"KS Test — {drift_text}",
            fontweight="bold", color=drift_color)
        axes[1].legend()
        axes[1].grid(axis="y", alpha=0.3)
        axes[1].text(
            0.5, 0.5,
            f"KS={stat:.4f}\np={p_val:.4f}\n\n"
            f"{drift_text}",
            transform=axes[1].transAxes,
            ha="center", va="center",
            fontsize=14, fontweight="bold",
            color=drift_color)
        plt.suptitle("Agent 5 — Drift Analysis",
                     fontweight="bold")
        plt.tight_layout()
        p = os.path.join(out_dir, "drift.png")
        fig.savefig(p, bbox_inches="tight")
        plt.close()
        paths.append(("Drift Analysis", p))
    except Exception:
        pass

    return paths


# ================================================================
# Custom Model Training (RandomForest — CPU friendly)
# ================================================================

def train_custom_model(df, label_col, benign_label,
                       max_rows=10000):
    from sklearn.preprocessing import (
        RobustScaler, LabelEncoder)
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    results = {
        "success"     : False,
        "message"     : "",
        "model"       : None,
        "scaler"      : None,
        "le"          : None,
        "features"    : None,
        "classes"     : None,
        "metrics"     : {},
        "n_samples"   : 0,
        "n_classes"   : 0,
        "removed_cols": {},
    }

    try:
        if len(df) > max_rows:
            df = df.sample(max_rows, random_state=42)
        results["n_samples"] = len(df)

        df_clean, avail, removed = auto_exclude(
            df, label_col, benign_label)
        results["removed_cols"] = removed

        if len(avail) == 0:
            results["message"] = \
                "❌ لا توجد features صالحة بعد الفلترة"
            return results

        le = LabelEncoder()
        y  = le.fit_transform(df[label_col].values)
        results["classes"]   = list(le.classes_)
        results["n_classes"] = len(le.classes_)

        X      = df_clean[avail].values.astype(
            np.float32)
        scaler = RobustScaler()
        X_sc   = scaler.fit_transform(X)
        results["features"] = avail

        counts    = Counter(y)
        valid_idx = np.array(
            [i for i, yi in enumerate(y)
             if counts[yi] >= 2])

        if len(valid_idx) < 50:
            results["message"] = \
                "❌ البيانات غير كافية"
            return results

        X_sc_f = X_sc[valid_idx]
        y_f    = y[valid_idx]

        counts_f = Counter(y_f)
        stratify = y_f if min(
            counts_f.values()) >= 2 else None

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_sc_f, y_f,
            test_size=0.2,
            random_state=42,
            stratify=stratify)

        model = RandomForestClassifier(
            n_estimators = 100,
            max_depth    = 15,
            random_state = 42,
            n_jobs       = -1,
            class_weight = "balanced",
        )
        model.fit(X_tr, y_tr)

        y_pred       = model.predict(X_te)
        y_pred_names = le.inverse_transform(y_pred)
        y_true_names = le.inverse_transform(y_te)

        acc = accuracy_score(y_true_names, y_pred_names)
        wf1 = f1_score(y_true_names, y_pred_names,
                       average="weighted",
                       zero_division=0)
        mf1 = f1_score(y_true_names, y_pred_names,
                       average="macro",
                       zero_division=0)
        report = classification_report(
            y_true_names, y_pred_names,
            zero_division=0)

        results.update({
            "success": True,
            "message": f"✅ تم التدريب! "
                       f"Accuracy={acc*100:.2f}%",
            "model"  : model,
            "scaler" : scaler,
            "le"     : le,
            "metrics": {
                "accuracy"   : acc,
                "weighted_f1": wf1,
                "macro_f1"   : mf1,
                "report"     : report,
            },
        })

    except Exception as e:
        results["message"] = f"❌ خطأ: {str(e)}"

    return results


# ================================================================
# Inference with Custom Model (RandomForest)
# ================================================================

def run_inference_custom(df, custom, label_col,
                         benign_label, ft_unk_thr=0.60):
    t0 = datetime.now()

    df_clean, avail, removed = auto_exclude(
        df, label_col, benign_label)

    train_feats = custom["features"]
    X = np.zeros(
        (len(df_clean), len(train_feats)),
        dtype=np.float32)
    matched = []
    missing = []
    for i, feat in enumerate(train_feats):
        if feat in avail and feat in df_clean.columns:
            X[:, i] = df_clean[feat].values.astype(
                np.float32)
            matched.append(feat)
        else:
            missing.append(feat)

    X_sc     = custom["scaler"].transform(X)
    y_pred_e = custom["model"].predict(X_sc)
    y_probs  = custom["model"].predict_proba(X_sc)
    y_conf   = y_probs.max(axis=1)
    y_unk    = y_conf < ft_unk_thr

    le           = custom["le"]
    y_pred_names = list(le.inverse_transform(y_pred_e))

    total   = len(y_pred_names)
    benign  = sum(1 for p in y_pred_names
                  if p == benign_label)
    unknown = int(y_unk.sum())
    attacks = total - benign - unknown

    atk_counts = Counter(
        p for p, u in zip(y_pred_names, y_unk)
        if p != benign_label and not u)

    metrics = {}
    if label_col in df.columns:
        y_true = df[label_col].values[:total]
        try:
            metrics["accuracy"]    = accuracy_score(
                y_true, y_pred_names)
            metrics["weighted_f1"] = f1_score(
                y_true, y_pred_names,
                average="weighted", zero_division=0)
            metrics["macro_f1"]    = f1_score(
                y_true, y_pred_names,
                average="macro", zero_division=0)
            metrics["report"]      = classification_report(
                y_true, y_pred_names,
                zero_division=0)
            metrics["cm_true"]     = y_true
            metrics["y_true"]      = y_true
        except Exception as e:
            metrics["error"] = str(e)

    elapsed = (datetime.now() - t0).seconds

    return {
        "y_pred"       : y_pred_names,
        "y_probs"      : y_probs,
        "y_conf"       : y_conf,
        "y_unknown"    : y_unk,
        "n_samples"    : total,
        "n_benign"     : benign,
        "n_attacks"    : attacks,
        "n_unknown"    : unknown,
        "atk_counts"   : atk_counts,
        "metrics"      : metrics,
        "removed_cols" : removed,
        "matched_feats": matched,
        "missing_feats": missing,
        "elapsed_sec"  : elapsed,
        "X_final"      : X_sc,
        "ft_unk_thr"   : ft_unk_thr,
        # ← feat_names من النموذج المخصص
        "feat_names"   : train_feats,
    }


# ================================================================
# Explainability Plots — من البيانات الفعلية
# ================================================================

def make_explainability_plots(results, models,
                              out_dir="plots/"):
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    y_pred  = results["y_pred"]
    y_probs = results["y_probs"]
    y_conf  = results["y_conf"]
    X       = results["X_final"]
    n_feats = X.shape[1]

    # ── أسماء الـ features — دائماً من النتائج ────────────
    # هذا يضمن التوافق مع أي نموذج (أصلي أو مخصص)
    feat_names = results.get("feat_names", [])

    # تأكد من التوافق مع حجم X
    if len(feat_names) != n_feats:
        # محاولة من الـ models
        if models and "features" in models:
            orig = models["features"]
            if len(orig) == n_feats:
                feat_names = orig
            else:
                feat_names = [
                    f"Feature_{i}"
                    for i in range(n_feats)]
        else:
            feat_names = [
                f"Feature_{i}"
                for i in range(n_feats)]

    feat_names = list(feat_names)[:n_feats]

    # ── حساب SHAP values من الداتاست الفعلية ─────────────
    shap_vals, shap_sign = _compute_shap_values(
        X, y_probs, feat_names)

    idx   = np.argsort(shap_vals)[::-1]
    top_n = min(len(feat_names), n_feats)
    t_f   = [feat_names[i] for i in idx[:top_n]]
    t_v   = shap_vals[idx[:top_n]]
    t_sgn = shap_sign[idx[:top_n]]

    # ── 1. SHAP Summary Bar Plot ──────────────────────────
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors_s = ["#e74c3c" if s >= 0
                    else "#3498db"
                    for s in t_sgn[::-1]]
        bars = ax.barh(
            t_f[::-1], t_v[::-1],
            color=colors_s, alpha=0.85)
        for bar, val in zip(bars, t_v[::-1]):
            ax.text(
                val + max(t_v)*0.01,
                bar.get_y()+bar.get_height()/2,
                f"{val:.4f}",
                va="center", fontsize=9)
        from matplotlib.patches import Patch
        legend_els = [
            Patch(color="#e74c3c",
                  label="Positive impact (+)"),
            Patch(color="#3498db",
                  label="Negative impact (−)")]
        ax.legend(handles=legend_els,
                  loc="lower right")
        ax.set_xlabel(
            "Mean |SHAP Value| "
            "(Impact on Model Output)")
        ax.set_title(
            "SHAP Summary — Feature Impact\n"
            f"(computed from {len(X):,} samples)",
            fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        p = os.path.join(out_dir, "shap_summary.png")
        fig.savefig(p, bbox_inches="tight")
        plt.close()
        paths.append(("SHAP Summary", p))
    except Exception:
        pass

    # ── 2. SHAP Waterfall Plot ────────────────────────────
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        base_val   = float(np.mean(y_conf))
        x_pos      = np.arange(top_n)
        cumulative = base_val
        lefts      = []
        widths     = []
        colors_w   = []

        for val, sgn in zip(t_v[::-1], t_sgn[::-1]):
            signed_val = val * (1 if sgn >= 0 else -1)
            lefts.append(
                cumulative if signed_val >= 0
                else cumulative + signed_val)
            widths.append(abs(signed_val))
            colors_w.append(
                "#e74c3c" if signed_val >= 0
                else "#3498db")
            cumulative += signed_val

        ax.barh(x_pos, widths, left=lefts,
                color=colors_w, alpha=0.85,
                height=0.6)
        ax.axvline(base_val, color="#2c3e50",
                   lw=1.5, ls="--",
                   label=f"E[f(X)]={base_val:.3f}")
        ax.axvline(cumulative, color="#27ae60",
                   lw=1.5, ls=":",
                   label=f"f(x)={cumulative:.3f}")
        ax.set_yticks(x_pos)
        ax.set_yticklabels(t_f[::-1], fontsize=10)

        for i, (w, l, sgn) in enumerate(
                zip(widths, lefts, t_sgn[::-1])):
            val = w * (1 if sgn >= 0 else -1)
            ax.text(
                l + w/2, i,
                f"{val:+.4f}",
                va="center", ha="center",
                fontsize=8, color="white",
                fontweight="bold")

        ax.set_xlabel(
            "SHAP Value (Cumulative Impact)")
        ax.set_title(
            "SHAP Waterfall — Feature Contribution\n"
            f"Base={base_val:.3f} → "
            f"Output={cumulative:.3f}",
            fontweight="bold")
        ax.legend(loc="lower right")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        p = os.path.join(
            out_dir, "shap_waterfall.png")
        fig.savefig(p, bbox_inches="tight")
        plt.close()
        paths.append(("SHAP Waterfall", p))
    except Exception:
        pass

    # ── 3. Confidence per Class ───────────────────────────
    try:
        class_conf = defaultdict(list)
        for pred, conf in zip(y_pred, y_conf):
            class_conf[pred].append(conf)

        cls_names = list(class_conf.keys())[:12]
        cls_means = [np.mean(class_conf[c])
                     for c in cls_names]
        cls_stds  = [np.std(class_conf[c])
                     for c in cls_names]

        fig, ax = plt.subplots(figsize=(12, 5))
        colors_c = ["#2ecc71" if mv >= 0.8
                    else "#f39c12" if mv >= 0.6
                    else "#e74c3c"
                    for mv in cls_means]
        ax.bar(range(len(cls_names)),
               cls_means, color=colors_c,
               alpha=0.85, yerr=cls_stds, capsize=4)
        ax.set_xticks(range(len(cls_names)))
        ax.set_xticklabels(
            [c[:14] for c in cls_names],
            rotation=45, ha="right", fontsize=9)
        ax.axhline(0.8, color="gray",
                   ls="--", lw=1.5,
                   label="Threshold=0.8")
        ax.set_ylabel("Mean Confidence")
        ax.set_title(
            "Mean Confidence per Class",
            fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        p = os.path.join(
            out_dir, "conf_per_class.png")
        fig.savefig(p, bbox_inches="tight")
        plt.close()
        paths.append(("Confidence per Class", p))
    except Exception:
        pass

    # ── 4. Probability Heatmap ────────────────────────────
    try:
        n_show  = min(30, len(y_probs))
        indices = sorted(np.random.choice(
            len(y_probs), n_show, replace=False))
        n_cls       = min(10, y_probs.shape[1])
        prob_subset = y_probs[indices, :n_cls]

        fig, ax = plt.subplots(
            figsize=(12, max(6, n_show*0.3)))
        im = ax.imshow(
            prob_subset, aspect="auto",
            cmap="YlOrRd", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="Probability")
        ax.set_xlabel("Class Index")
        ax.set_ylabel("Sample")
        ax.set_title(
            "Prediction Probability Heatmap",
            fontweight="bold")
        plt.tight_layout()
        p = os.path.join(
            out_dir, "prob_heatmap.png")
        fig.savefig(p, bbox_inches="tight")
        plt.close()
        paths.append(("Probability Heatmap", p))
    except Exception:
        pass

    # ── 5. Confidence Breakdown ───────────────────────────
    try:
        bins   = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
        labels = ["<50%","50-60%","60-70%",
                  "70-80%","80-90%","90-100%"]
        counts = [int(np.sum(
            (y_conf >= bins[i]) &
            (y_conf < bins[i+1])))
            for i in range(len(bins)-1)]

        fig, ax = plt.subplots(figsize=(10, 5))
        colors_b = ["#e74c3c","#e67e22","#f1c40f",
                    "#2ecc71","#27ae60","#1abc9c"]
        bars = ax.bar(labels, counts,
                      color=colors_b, alpha=0.85)
        for bar, val in zip(bars, counts):
            ax.text(
                bar.get_x()+bar.get_width()/2,
                bar.get_height()+max(counts)*0.01,
                f"{val:,}", ha="center", fontsize=9)
        ax.set_xlabel("Confidence Range")
        ax.set_ylabel("Number of Samples")
        ax.set_title(
            "Decision Confidence Breakdown",
            fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        p = os.path.join(
            out_dir, "conf_breakdown.png")
        fig.savefig(p, bbox_inches="tight")
        plt.close()
        paths.append(("Confidence Breakdown", p))
    except Exception:
        pass

    return paths
