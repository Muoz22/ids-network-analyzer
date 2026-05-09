# ================================================================
# inference.py — AI Agents IDS Inference Module
# ================================================================

import pickle, json, re
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score)
from datetime import datetime
import os


def load_models(model_dir: str = "models/"):
    """تحميل النماذج المحفوظة"""
    import tensorflow as tf
    from tensorflow import keras

    models = {}

    # FT-Transformer
    class FeatureTokenizer(tf.keras.layers.Layer):
        def __init__(self, n_feat, dim, **kw):
            super().__init__(**kw)
            self.n_feat = n_feat
            self.dim = dim
        def build(self, _):
            self.W = self.add_weight(
                name="W", shape=(self.n_feat, self.dim),
                initializer="glorot_uniform", trainable=True)
            self.b = self.add_weight(
                name="b", shape=(self.n_feat, self.dim),
                initializer="zeros", trainable=True)
            super().build(_)
        def call(self, x):
            return tf.expand_dims(x, -1) * self.W + self.b
        def get_config(self):
            cfg = super().get_config()
            cfg.update({"n_feat": self.n_feat, "dim": self.dim})
            return cfg

    class CLSToken(tf.keras.layers.Layer):
        def __init__(self, dim, **kw):
            super().__init__(**kw)
            self.dim = dim
        def build(self, _):
            self.cls = self.add_weight(
                name="cls", shape=(1, 1, self.dim),
                initializer="random_normal", trainable=True)
            super().build(_)
        def call(self, x):
            return tf.concat(
                [tf.tile(self.cls, [tf.shape(x)[0], 1, 1]), x],
                axis=1)
        def get_config(self):
            cfg = super().get_config()
            cfg.update({"dim": self.dim})
            return cfg

    model_path = os.path.join(model_dir, "ft_transformer.keras")
    models["model"] = keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={
            "FeatureTokenizer": FeatureTokenizer,
            "CLSToken": CLSToken,
        })

    with open(os.path.join(model_dir, "scaler.pkl"), "rb") as f:
        models["scaler"] = pickle.load(f)

    with open(os.path.join(model_dir, "selected_features.json")) as f:
        feats = json.load(f)
    seen = set()
    models["features"] = [
        x for x in feats if not (x in seen or seen.add(x))]

    with open(os.path.join(model_dir, "class_names.json")) as f:
        models["class_names"] = json.load(f)

    models["n_features"] = models["model"].input_shape[1]
    return models


def auto_exclude(df, label_col, benign_label):
    """اكتشاف الأعمدة الضارة تلقائياً بالمنطق"""
    df = df.replace([float('inf'), float('-inf')], float('nan'))

    obj_cols = [c for c in df.select_dtypes('object').columns
                if c != label_col]
    if obj_cols:
        df = df.drop(columns=obj_cols)

    excl    = [label_col]
    removed = {}

    if label_col in df.columns:
        df["_lb"] = (df[label_col] != benign_label).astype(int)
        excl.append("_lb")

    num_df = df.select_dtypes(include=[float, int, "int64", "float64"])

    # 1. Data Leakage
    if "_lb" in df.columns:
        for col in num_df.columns:
            if col in excl: continue
            uv = set(df[col].dropna().unique())
            if uv.issubset({0, 1, 0.0, 1.0}):
                corr = abs(df[col].corr(df["_lb"]))
                if corr > 0.95:
                    excl.append(col)
                    removed[col] = f"data leakage ({corr:.3f})"

    # 2. Timestamps
    for col in num_df.columns:
        if col in excl: continue
        if df[col].median() > 1e9:
            excl.append(col)
            removed[col] = "timestamp"

    # 3. Sequential IDs
    for col in num_df.columns:
        if col in excl: continue
        sv = df[col].dropna().sort_values().reset_index(drop=True)
        if len(sv) > 100:
            diffs = sv.diff().dropna()
            if (len(diffs) > 0 and
                    (diffs >= 0).mean() > 0.99 and
                    diffs.std() < 0.01 and
                    diffs.mean() > 0):
                excl.append(col)
                removed[col] = "sequential ID"

    # 4. Zero Variance
    for col in num_df.columns:
        if col in excl: continue
        if df[col].std() == 0:
            excl.append(col)
            removed[col] = "zero variance"

    # 5. Near-Constant
    for col in num_df.columns:
        if col in excl: continue
        vc = df[col].value_counts(normalize=True)
        if len(vc) > 0 and vc.iloc[0] > 0.999:
            excl.append(col)
            removed[col] = f"near-constant ({vc.iloc[0]*100:.1f}%)"

    if "_lb" in df.columns:
        df = df.drop(columns=["_lb"])

    avail = [c for c in df.columns if c not in excl]
    df[avail] = df[avail].fillna(df[avail].median())
    return df, avail, removed


def align_features(df, avail, features, n_model):
    """مطابقة الـ features مع النموذج"""
    result  = np.zeros((len(df), n_model), dtype=np.float32)
    matched = []
    missing = []

    for i, feat in enumerate(features[:n_model]):
        if feat in avail and feat in df.columns:
            result[:, i] = df[feat].values.astype(np.float32)
            matched.append(feat)
        else:
            missing.append(feat)

    return result, matched, missing


def run_inference(df, models, label_col, benign_label,
                  ft_unk_thr=0.60):
    """
    تشغيل الـ inference على dataframe.
    يرجع dict بالنتائج الكاملة.
    """
    t0 = datetime.now()

    # تنظيف
    df_clean, avail, removed = auto_exclude(df, label_col, benign_label)

    # features
    n_model  = models["n_model"] if "n_model" in models \
               else models["n_features"]
    X_raw, matched, missing = align_features(
        df_clean, avail, models["features"], n_model)

    # Scaler
    n_scaler = models["scaler"].n_features_in_
    if X_raw.shape[1] != n_scaler:
        X_pad = np.zeros(
            (len(X_raw), n_scaler), dtype=np.float32)
        X_pad[:, :min(X_raw.shape[1], n_scaler)] = \
            X_raw[:, :min(X_raw.shape[1], n_scaler)]
        X_raw = X_pad

    X_scaled = models["scaler"].transform(X_raw)

    # Model padding
    if X_scaled.shape[1] != n_model:
        X_pad2 = np.zeros(
            (len(X_scaled), n_model), dtype=np.float32)
        X_pad2[:, :X_scaled.shape[1]] = X_scaled
        X_scaled = X_pad2

    # تنبؤ
    y_probs = models["model"].predict(
        X_scaled.astype(np.float32),
        batch_size=4096, verbose=0)
    y_pred  = np.argmax(y_probs, axis=1)
    y_conf  = y_probs.max(axis=1)
    y_unk   = y_conf < ft_unk_thr

    y_pred_names = [
        models["class_names"][i]
        if i < len(models["class_names"]) else "?"
        for i in y_pred]

    # إحصاءات
    total   = len(y_pred_names)
    benign  = sum(1 for p in y_pred_names if p == benign_label)
    unknown = int(y_unk.sum())
    attacks = total - benign - unknown

    atk_counts = Counter(
        p for p, u in zip(y_pred_names, y_unk)
        if p != benign_label and not u)

    # دقة لو عندنا labels
    metrics = {}
    if label_col in df.columns:
        y_true = df[label_col].values[:len(y_pred_names)]
        try:
            metrics["accuracy"]    = accuracy_score(y_true, y_pred_names)
            metrics["weighted_f1"] = f1_score(
                y_true, y_pred_names, average="weighted",
                zero_division=0)
            metrics["macro_f1"]    = f1_score(
                y_true, y_pred_names, average="macro",
                zero_division=0)
            metrics["report"]      = classification_report(
                y_true, y_pred_names,
                labels=sorted(set(y_true) | set(y_pred_names)),
                zero_division=0)
            metrics["cm_true"]     = y_true
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
    }


def make_plots(results, benign_label, out_dir="plots/"):
    """إنشاء الرسوم البيانية وحفظها"""
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    y_pred = results["y_pred"]
    y_conf = results["y_conf"]
    y_unk  = results["y_unknown"]
    total  = results["n_samples"]
    benign = results["n_benign"]
    atks   = results["n_attacks"]
    unk    = results["n_unknown"]

    plt.rcParams.update({
        "figure.dpi": 120, "font.size": 11,
        "figure.facecolor": "white"})

    # ── 1. Pie ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    sizes  = [benign, atks, unk]
    labels = ["Benign", "Attack", "Unknown"]
    colors = ["#2ecc71", "#e74c3c", "#f39c12"]
    ax.pie([max(s,1) for s in sizes],
           labels=labels, colors=colors,
           autopct="%1.1f%%", startangle=140,
           explode=(0.02,0.05,0.05), shadow=True)
    ax.set_title(f"Decision Distribution\nTotal: {total:,}",
                 fontweight="bold")
    p = os.path.join(out_dir, "pie.png")
    fig.savefig(p, bbox_inches="tight"); plt.close()
    paths.append(("Decision Distribution", p))

    # ── 2. Attack Bar ──────────────────────────────────────────
    if results["atk_counts"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        top = results["atk_counts"].most_common(15)
        names = [x[0] for x in top]
        vals  = [x[1] for x in top]
        colors_b = plt.cm.Reds_r(
            [0.3 + 0.5*(i/len(names)) for i in range(len(names))])
        ax.barh(names, vals, color=colors_b, alpha=0.85)
        for i, v in enumerate(vals):
            ax.text(v + max(vals)*0.01, i,
                    f"{v:,}", va="center", fontsize=9)
        ax.set_xlabel("Count")
        ax.set_title("Top Attack Types Detected", fontweight="bold")
        ax.set_xlim(0, max(vals)*1.15)
        plt.tight_layout()
        p = os.path.join(out_dir, "attacks.png")
        fig.savefig(p, bbox_inches="tight"); plt.close()
        paths.append(("Attack Types", p))

    # ── 3. Confidence Distribution ─────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    n = min(3000, len(y_conf))
    colors_s = ["#2ecc71" if p == benign_label
                else "#e74c3c" for p in y_pred[:n]]
    ax.scatter(range(n), y_conf[:n],
               c=colors_s, alpha=0.4, s=6)
    ax.axhline(0.60, color="black", ls="--",
               lw=1.5, label="Unknown threshold=0.60")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Confidence")
    ax.set_title("Prediction Confidence", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    p = os.path.join(out_dir, "confidence.png")
    fig.savefig(p, bbox_inches="tight"); plt.close()
    paths.append(("Confidence", p))

    # ── 4. Confusion Matrix (لو عندنا labels) ──────────────────
    m = results["metrics"]
    if "cm_true" in m:
        try:
            y_true = m["cm_true"]
            present = sorted(set(y_true) | set(y_pred))[:15]
            cm = confusion_matrix(y_true, y_pred, labels=present)
            sz = max(7, len(present))
            fig, ax = plt.subplots(figsize=(sz+1, sz))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=present,
                        yticklabels=present,
                        annot_kws={"size": 8}, ax=ax)
            ax.set_title("Confusion Matrix", fontweight="bold")
            ax.set_ylabel("True"); ax.set_xlabel("Predicted")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            p = os.path.join(out_dir, "cm.png")
            fig.savefig(p, bbox_inches="tight"); plt.close()
            paths.append(("Confusion Matrix", p))
        except Exception:
            pass

    return paths
