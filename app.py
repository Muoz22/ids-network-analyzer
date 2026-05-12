# ================================================================
# app.py — AI Agents IDS v3 — Network Security Analyzer
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import os, tempfile, json
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(
    page_title="AI Agents IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
    padding: 2rem; border-radius: 12px;
    margin-bottom: 2rem; text-align: center;
}
.main-header h1 { color:#00d4ff; font-size:2.5rem; margin:0; }
.main-header p  { color:#a0aec0; font-size:1.1rem; margin:0.5rem 0 0; }
.metric-value   { font-size:2rem; font-weight:bold; margin:0; }
.metric-label   { color:#a0aec0; font-size:0.9rem; }
.status-excellent { color:#00d4ff; }
.status-good      { color:#48bb78; }
.status-warning   { color:#ed8936; }
.status-bad       { color:#e74c3c; }
.agent-card {
    border-left: 4px solid;
    padding: 1rem; margin: 0.5rem 0;
    background: rgba(0,0,0,0.05);
    border-radius: 0 8px 8px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🛡️ AI Agents IDS</h1>
    <p>Network Intrusion Detection System — Powered by 6 AI Agents | v3 Universal</p>
    <p style="color:#666;font-size:0.8rem;margin-top:0.5rem;">
    © 2025 Muaz Al-Soufi |
    <a href="https://github.com/Muoz22/ids-network-analyzer"
    style="color:#00d4ff;">GitHub</a></p>
</div>
""", unsafe_allow_html=True)


# ================================================================
# Smart Detection Engine
# ================================================================

def smart_detect(df):
    result = {
        "label_col"   : None,
        "benign_label": None,
        "problem_type": None,
        "n_classes"   : 0,
        "classes"     : [],
        "confidence"  : 0,
        "reason"      : "",
        "all_options" : [],
    }

    benign_vals = [
        "normal","Normal","BENIGN","benign","Benign",
        "BenignTraffic","legitimate","Legitimate",
        "background","Background","safe","Safe",
        "none","None","0",0]

    all_candidates = [
        c for c in df.columns
        if df[c].dtype == object or
        df[c].nunique() < 50]

    options = []
    for col in all_candidates:
        n_unique = df[col].nunique()
        vals     = df[col].unique().tolist()
        found_benign = None
        for b in benign_vals:
            if b in vals:
                found_benign = str(b)
                break
        options.append({
            "col"         : col,
            "n_unique"    : n_unique,
            "found_benign": found_benign,
            "vals_sample" : [str(v) for v in vals[:5]],
        })

    result["all_options"] = options

    multiclass_priority = [
        "category","subcategory","type","label",
        "attack_type","attack_cat","traffic_type",
        "class_label","Label","Type","Category","Class"]

    for col in multiclass_priority:
        if col not in df.columns: continue
        n_unique = df[col].nunique()
        vals     = df[col].unique().tolist()
        found_benign = None
        for b in benign_vals:
            if b in vals:
                found_benign = str(b)
                break
        if n_unique > 2 and found_benign:
            result.update({
                "label_col"   : col,
                "benign_label": found_benign,
                "problem_type": "multiclass",
                "n_classes"   : n_unique,
                "classes"     : [str(v) for v in vals],
                "confidence"  : 95,
                "reason"      : f"عمود '{col}' يحتوي "
                                f"{n_unique} كلاس مع "
                                f"'{found_benign}' كـ Normal",
            })
            return result

    binary_priority = [
        "label","type","attack","target",
        "Label","Attack","is_attack","malicious"]

    for col in binary_priority:
        if col not in df.columns: continue
        vals     = df[col].unique().tolist()
        n_unique = len(vals)
        if n_unique == 2:
            vc         = df[col].value_counts()
            benign_val = str(vc.index[-1])
            result.update({
                "label_col"   : col,
                "benign_label": benign_val,
                "problem_type": "binary",
                "n_classes"   : 2,
                "classes"     : [str(v) for v in vals],
                "confidence"  : 80,
                "reason"      : f"عمود '{col}' binary — "
                                f"Benign={benign_val}",
            })
            return result

    for opt in options:
        if opt["found_benign"] and opt["n_unique"] > 2:
            col = opt["col"]
            result.update({
                "label_col"   : col,
                "benign_label": opt["found_benign"],
                "problem_type": "multiclass",
                "n_classes"   : opt["n_unique"],
                "classes"     : opt["vals_sample"],
                "confidence"  : 70,
                "reason"      : f"عمود '{col}' يحتوي "
                                f"'{opt['found_benign']}'",
            })
            return result

    return result


def apply_detection(df, mode, detection,
                    manual_lc, manual_bl):
    if mode == "🤖 Auto (ذكي)":
        if detection["label_col"]:
            return (detection["label_col"],
                    detection["benign_label"],
                    detection["problem_type"])
        return manual_lc, manual_bl, "unknown"
    elif mode == "🟢 Multi-class (أنواع هجمات)":
        for opt in detection.get("all_options", []):
            if opt["n_unique"] > 2 and opt["found_benign"]:
                return (opt["col"],
                        opt["found_benign"],
                        "multiclass")
        return manual_lc, manual_bl, "multiclass"
    elif mode == "🟡 Binary (هجوم/طبيعي)":
        for opt in detection.get("all_options", []):
            if opt["n_unique"] == 2:
                return (opt["col"],
                        opt["found_benign"] or "0",
                        "binary")
        return manual_lc, manual_bl, "binary"
    return manual_lc, manual_bl, "unknown"


def is_compatible(df, model_features):
    if not model_features:
        return False, 0.0
    available = df.columns.tolist()
    matched   = [f for f in model_features
                 if f in available]
    pct = len(matched) / len(model_features)
    return pct >= 0.7, pct


def auto_train_if_needed(df, use_lc, use_bl,
                         status_placeholder):
    """
    """
    from inference import train_custom_model

    # ── هل النموذج الأصلي متوافق؟ ─────────────────────────
    models, err = load_models_cached(version="v3")
    if not err and models:
        orig_feats  = models.get("features", [])
        compatible, pct = is_compatible(df, orig_feats)
        if compatible:
            # احذف أي نموذج مخصص قديم
            if "_auto_custom_model" in st.session_state:
                del st.session_state["_auto_custom_model"]
                del st.session_state["_auto_custom_meta"]
            return False

    # ── لا يوجد نموذج متوافق → درّب ضمنياً ───────────────
    if not use_lc or use_lc not in df.columns:
        return False

    status_placeholder.text("⚙️ تحليل الداتاست...")

    train_results = train_custom_model(
        df,
        label_col    = use_lc,
        benign_label = use_bl,
        max_rows     = 10000)

    if train_results["success"]:
        st.session_state["_auto_custom_model"] = {
            "model"   : train_results["model"],
            "scaler"  : train_results["scaler"],
            "le"      : train_results["le"],
            "features": train_results["features"],
        }
        st.session_state["_auto_custom_meta"] = {
            "classes"     : train_results["classes"],
            "n_classes"   : train_results["n_classes"],
            "features"    : train_results["features"],
            "metrics"     : train_results["metrics"],
            "label_col"   : use_lc,
            "benign_label": use_bl,
        }
        return True

    return False


@st.cache_resource
def load_models_cached(version="v3"):
    try:
        from inference import load_models
        models = load_models(model_dir="models/")
        return models, None
    except Exception as e:
        return None, str(e)

models_global, models_err = load_models_cached(version="v3")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ إعدادات التحليل")
    default_lc = "type"
    default_bl = "normal"
    if models_global and "meta" in models_global:
        meta = models_global["meta"]
        default_lc = meta.get("label_col", "type")
        default_bl = meta.get("benign_label", "normal")

    label_col    = st.text_input("اسم عمود الـ Label",
                                 value=default_lc)
    benign_label = st.text_input("اسم الكلاس الطبيعي",
                                 value=default_bl)
    threshold    = st.slider("Unknown Threshold",
                             min_value=0.3, max_value=0.9,
                             value=0.6, step=0.05)

    st.markdown("---")
    st.markdown("### 📊 النموذج")

    # الـ sidebar يعرض النموذج الأصلي فقط
    if models_global and "meta" in models_global:
        meta = models_global["meta"]
        st.success("✅ النموذج الأصلي (TON-IoT)")
        st.write(f"**Accuracy:** "
                 f"{meta.get('accuracy',0)*100:.2f}%")
        st.write(f"**Classes:** "
                 f"{meta.get('n_classes',0)}")
        st.write(f"**Features:** "
                 f"{meta.get('n_features',0)}")
        with st.expander("🔍 التفاصيل"):
            st.write("**Class Names:**")
            for c in meta.get("class_names", []):
                st.write(f"  • {c}")
            st.write("**Features:**")
            for f in meta.get("feat_cols", []):
                st.write(f"  • {f}")
    elif models_err:
        st.error(f"❌ {models_err}")

    st.markdown("---")
    st.markdown("### 🔗 روابط")
    st.markdown(
        "[GitHub](https://github.com/Muoz22/ids-network-analyzer)")


# ── 4 Tabs فقط (بدون Train Custom Model) ─────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 تحليل الشبكة",
    "📊 تقرير تفصيلي",
    "📖 كيف يعمل",
    "📋 عن المشروع",
])


# ══════════════════════════════════════════════════════════════
# Tab 1
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## 📁 ارفع ملف CSV للتحليل")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "اختر ملف CSV يحتوي على network traffic data",
            type=["csv"])
    with col2:
        st.markdown("""
        **الداتاسيت المدعومة:**
        - TON-IoT
        - CIC-IoT-2023
        - UNSW-NB15
        - Bot-IoT
        - CICIDS2017
        - أي CSV بـ network features
        """)

    if uploaded_file is not None:
        with st.spinner("جاري قراءة الملف..."):
            try:
                df = pd.read_csv(uploaded_file,
                                 low_memory=False)
                st.success(f"✅ تم تحميل الملف: "
                           f"{df.shape[0]:,} صف × "
                           f"{df.shape[1]} عمود")
            except Exception as e:
                st.error(f"❌ خطأ: {e}")
                st.stop()

        with st.expander("👁️ معاينة البيانات",
                         expanded=False):
            st.dataframe(df.head(10),
                         use_container_width=True)
            st.write(f"**الأعمدة:** "
                     f"{df.columns.tolist()}")

        # ── Smart Detection ────────────────────────────────
        detection = smart_detect(df)

        st.markdown("### 🎯 نوع التحليل")
        dc1, dc2 = st.columns([1, 1])

        with dc1:
            if detection["label_col"]:
                conf_color = (
                    "🟢" if detection["confidence"] >= 90
                    else "🟡" if detection["confidence"] >= 70
                    else "🔴")
                st.success(
                    f"{conf_color} **اكتشاف تلقائي** "
                    f"(ثقة {detection['confidence']}%)\n\n"
                    f"**النوع:** {detection['problem_type']}\n\n"
                    f"**Label:** `{detection['label_col']}`\n\n"
                    f"**Benign:** `{detection['benign_label']}`\n\n"
                    f"**Classes:** {detection['n_classes']}\n\n"
                    f"**السبب:** {detection['reason']}")
            else:
                st.warning(
                    "⚠️ لم يُعثر على عمود label تلقائياً\n\n"
                    "اختر **Manual** وأدخل المعلومات يدوياً")

        with dc2:
            analysis_mode = st.radio(
                "اختر طريقة التحليل:",
                ["🤖 Auto (ذكي)",
                 "🟢 Multi-class (أنواع هجمات)",
                 "🟡 Binary (هجوم/طبيعي)",
                 "✏️ Manual (يدوي)"],
                index=0,
                key="analysis_mode_tab1")

        # ── Manual override ────────────────────────────────
        if analysis_mode == "✏️ Manual (يدوي)":
            mc1, mc2 = st.columns(2)
            with mc1:
                manual_lc = st.selectbox(
                    "اختر عمود الـ Label:",
                    options=df.columns.tolist(),
                    index=0,
                    key="manual_lc_tab1")
            with mc2:
                if manual_lc in df.columns:
                    vals = df[manual_lc].unique().tolist()
                    manual_bl = st.selectbox(
                        "اختر الكلاس الطبيعي:",
                        options=[str(v) for v in vals],
                        key="manual_bl_tab1")
                else:
                    manual_bl = st.text_input(
                        "اكتب الكلاس الطبيعي:",
                        value="normal",
                        key="manual_bl_text_tab1")
            use_lc   = manual_lc
            use_bl   = str(manual_bl)
            use_type = "manual"
        else:
            use_lc, use_bl, use_type = apply_detection(
                df, analysis_mode, detection,
                label_col, benign_label)

        # ── عرض الاختيار النهائي ──────────────────────────
        st.markdown(
            f"**الاختيار النهائي:** "
            f"Label=`{use_lc}` | "
            f"Benign=`{use_bl}` | "
            f"Type=`{use_type}`")

        if use_lc in df.columns:
            vc = df[use_lc].value_counts()
            st.info(f"✅ `{use_lc}`: "
                    f"{vc.head(5).to_dict()}")

        # ── زر التحليل ────────────────────────────────────
        if st.button("🚀 ابدأ التحليل",
                     type="primary",
                     use_container_width=True,
                     key="btn_analyze"):

            progress = st.progress(0)
            status   = st.empty()

            with st.spinner("🔮 جاري التحليل..."):

                from inference import (
                    run_inference,
                    run_inference_custom,
                    make_plots)

                # ── فحص التوافق + تدريب تلقائي ───────────
                status.text("🔍 فحص التوافق...")
                progress.progress(10)

                trained_new = auto_train_if_needed(
                    df, use_lc, use_bl, status)

                progress.progress(35)
                status.text("🤖 تشغيل التحليل...")
                progress.progress(50)

                # ── اختر النموذج المناسب ──────────────────
                if "_auto_custom_model" in st.session_state:
                    # نموذج مخصص تدرّب تلقائياً
                    auto_cm = st.session_state[
                        "_auto_custom_model"]
                    auto_lc = st.session_state[
                        "_auto_custom_meta"]["label_col"]
                    auto_bl = st.session_state[
                        "_auto_custom_meta"]["benign_label"]
                    results = run_inference_custom(
                        df, auto_cm,
                        auto_lc, auto_bl,
                        ft_unk_thr=threshold)
                    model_used = "🔧 Auto-trained (RandomForest)"
                else:
                    # النموذج الأصلي
                    models, err = load_models_cached(
                        version="v3")
                    if err:
                        st.error(f"❌ {err}")
                        st.stop()
                    results = run_inference(
                        df, models,
                        use_lc, use_bl,
                        ft_unk_thr=threshold)
                    model_used = "🔵 Original (FT-Transformer)"

                status.text("📊 إنتاج الرسوم...")
                progress.progress(80)

                final_bl = (
                    st.session_state["_auto_custom_meta"]
                    ["benign_label"]
                    if "_auto_custom_model" in
                    st.session_state
                    else use_bl)

                with tempfile.TemporaryDirectory() as tmp:
                    plot_paths = make_plots(
                        results, final_bl,
                        out_dir=tmp)
                    plot_bytes = []
                    for title, path in plot_paths:
                        try:
                            with open(path, "rb") as f:
                                plot_bytes.append(
                                    (title, f.read()))
                        except Exception:
                            pass

                progress.progress(100)
                status.text("✅ اكتمل!")

                st.session_state["results"]    = results
                st.session_state["plot_bytes"] = plot_bytes

                # ── إحصاءات ───────────────────────────────
                st.markdown("---")
                st.markdown("## 🏆 نتائج التحليل")

                total   = results["n_samples"]
                benign  = results["n_benign"]
                attacks = results["n_attacks"]
                unknown = results["n_unknown"]

                c1,c2,c3,c4 = st.columns(4)
                with c1:
                    st.metric("إجمالي العينات",
                              f"{total:,}")
                with c2:
                    st.metric("✅ Benign", f"{benign:,}",
                              f"{100*benign/total:.1f}%")
                with c3:
                    st.metric("🚫 هجمات", f"{attacks:,}",
                              f"{100*attacks/total:.1f}%",
                              delta_color="inverse")
                with c4:
                    st.metric("⚠️ Unknown", f"{unknown:,}",
                              f"{100*unknown/total:.1f}%",
                              delta_color="inverse")

                m = results["metrics"]
                if "accuracy" in m:
                    st.markdown("### 📈 دقة النموذج")
                    mc1,mc2,mc3 = st.columns(3)
                    for col, (name, val, hi, lo) in zip(
                            [mc1,mc2,mc3], [
                                ("Accuracy",
                                 m["accuracy"]*100,97,93),
                                ("Weighted F1",
                                 m["weighted_f1"]*100,95,90),
                                ("Macro F1",
                                 m["macro_f1"]*100,80,60),
                            ]):
                        with col:
                            cl = ("status-excellent"
                                  if val>hi else
                                  "status-good" if val>lo
                                  else "status-warning")
                            st.markdown(
                                f'<p class="metric-value {cl}">'
                                f'{val:.2f}%</p>'
                                f'<p class="metric-label">'
                                f'{name}</p>',
                                unsafe_allow_html=True)

                if results["atk_counts"]:
                    st.markdown("### 🔴 أنواع الهجمات")
                    atk_df = pd.DataFrame(
                        results["atk_counts"].most_common(20),
                        columns=["نوع الهجوم","العدد"])
                    atk_df["النسبة %"] = (
                        atk_df["العدد"]/total*100
                    ).round(2)
                    st.dataframe(atk_df,
                                 use_container_width=True)

                if plot_bytes:
                    st.markdown("### 📊 الرسوم البيانية")
                    for i in range(0, len(plot_bytes), 2):
                        cols = st.columns(2)
                        for j, (title, img) in enumerate(
                                plot_bytes[i:i+2]):
                            with cols[j]:
                                st.markdown(f"**{title}**")
                                st.image(img,
                                         use_column_width=True)

                if "report" in m:
                    with st.expander(
                            "📋 تقرير Classification كامل"):
                        st.code(m["report"])

                with st.expander("🔍 تفاصيل المعالجة"):
                    st.write(f"**النموذج:** {model_used}")
                    if trained_new:
                        st.info(
                            "✅ تم تدريب نموذج تلقائياً "
                            "لهذه الداتاست")
                    st.write(f"**وقت التنفيذ:** "
                             f"{results['elapsed_sec']}s")
                    st.write(f"**Label:** `{use_lc}`")
                    st.write(f"**Benign:** `{use_bl}`")
                    st.write(f"**Type:** `{use_type}`")
                    st.write(f"**Features مطابقة:** "
                             f"{results['matched_feats']}")
                    if results["missing_feats"]:
                        st.warning(
                            f"**مفقودة:** "
                            f"{results['missing_feats']}")
                    if results["removed_cols"]:
                        st.info(
                            f"**مستُبعدت:** "
                            f"{list(results['removed_cols'].keys())}")

                st.markdown("### 💾 تحميل النتائج")
                result_df = pd.DataFrame({
                    "prediction": results["y_pred"],
                    "confidence": results["y_conf"],
                    "is_unknown": results["y_unknown"],
                })
                st.download_button(
                    "⬇️ تحميل النتائج CSV",
                    result_df.to_csv(index=False),
                    f"ids_results_"
                    f"{datetime.now():%Y%m%d_%H%M}.csv",
                    "text/csv",
                    use_container_width=True,
                    key="download_tab1_csv")

                st.info(
                    "💡 انتقل لـ **📊 تقرير تفصيلي** "
                    "لرؤية كل الرسوم والتحليل الكامل")


# ══════════════════════════════════════════════════════════════
# Tab 2
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 📊 تقرير تفصيلي")

    if "results" not in st.session_state:
        st.info("🔍 ارفع ملف في **تحليل الشبكة** أولاً")
    else:
        results    = st.session_state["results"]
        plot_bytes = st.session_state["plot_bytes"]
        m          = results["metrics"]

        total   = results["n_samples"]
        benign  = results["n_benign"]
        attacks = results["n_attacks"]
        unknown = results["n_unknown"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("✅ ALLOW", f"{benign:,}",
                      f"{100*benign/total:.1f}%")
        with col2:
            st.metric("🚫 BLOCK", f"{attacks:,}",
                      f"{100*attacks/total:.1f}%",
                      delta_color="inverse")
        with col3:
            st.metric("⚠️ QUARANTINE", f"{unknown:,}",
                      f"{100*unknown/total:.1f}%",
                      delta_color="inverse")

        st.markdown("---")
        st.markdown("### 🎯 Decision Analysis")
        pie_imgs = [(t,img) for t,img in plot_bytes
                    if "Distribution" in t or "Pie" in t]
        atk_imgs = [(t,img) for t,img in plot_bytes
                    if "Attack" in t]
        dc1, dc2 = st.columns(2)
        with dc1:
            if pie_imgs:
                st.markdown("**Decision Distribution**")
                st.image(pie_imgs[0][1],
                         use_column_width=True)
        with dc2:
            if atk_imgs:
                st.markdown("**Attack Types**")
                st.image(atk_imgs[0][1],
                         use_column_width=True)

        if "accuracy" in m:
            st.markdown("---")
            st.markdown(
                "### 📈 Classification Performance")
            mc1,mc2,mc3 = st.columns(3)
            for col, (name, val, hi, lo) in zip(
                    [mc1,mc2,mc3], [
                        ("Accuracy",
                         m["accuracy"]*100,97,93),
                        ("Weighted F1",
                         m["weighted_f1"]*100,95,90),
                        ("Macro F1",
                         m["macro_f1"]*100,80,60),
                    ]):
                with col:
                    cl = ("status-excellent" if val>hi
                          else "status-good" if val>lo
                          else "status-warning")
                    st.markdown(
                        f'<p class="metric-value {cl}">'
                        f'{val:.2f}%</p>'
                        f'<p class="metric-label">'
                        f'{name}</p>',
                        unsafe_allow_html=True)

            conf_imgs = [(t,img) for t,img in plot_bytes
                         if "Confusion" in t]
            f1_imgs   = [(t,img) for t,img in plot_bytes
                         if "F1" in t]
            if conf_imgs or f1_imgs:
                pfc1, pfc2 = st.columns(2)
                with pfc1:
                    if conf_imgs:
                        st.markdown("**Confusion Matrix**")
                        st.image(conf_imgs[0][1],
                                 use_column_width=True)
                with pfc2:
                    if f1_imgs:
                        st.markdown("**Per-Class F1**")
                        st.image(f1_imgs[0][1],
                                 use_column_width=True)

            if "report" in m:
                st.markdown("---")
                st.markdown("### 📋 Classification Report")
                st.code(m["report"])

        st.markdown("---")
        st.markdown("### 🎚️ Confidence Analysis")
        conf_imgs = [(t,img) for t,img in plot_bytes
                     if "Confidence" in t]
        if conf_imgs:
            st.image(conf_imgs[0][1],
                     use_column_width=True)

        st.markdown("---")
        st.markdown("### 🌊 Drift Analysis")
        drift_imgs = [(t,img) for t,img in plot_bytes
                      if "Drift" in t]
        if drift_imgs:
            st.image(drift_imgs[0][1],
                     use_column_width=True)

        st.markdown("---")
        st.markdown("### 🗂️ Summary Dashboard")
        dash_imgs = [(t,img) for t,img in plot_bytes
                     if "Dashboard" in t or "Summary" in t]
        if dash_imgs:
            st.image(dash_imgs[0][1],
                     use_column_width=True)

        st.markdown("---")
        st.markdown("### 🖼️ كل الرسوم البيانية")
        for i in range(0, len(plot_bytes), 2):
            cols = st.columns(2)
            for j, (title, img) in enumerate(
                    plot_bytes[i:i+2]):
                with cols[j]:
                    st.markdown(f"**{title}**")
                    st.image(img, use_column_width=True)

        st.markdown("---")
        st.markdown("### 📥 تحميل النتائج")
        result_df = pd.DataFrame({
            "prediction": results["y_pred"],
            "confidence": results["y_conf"],
            "is_unknown": results["y_unknown"],
        })
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "⬇️ تحميل النتائج CSV",
                result_df.to_csv(index=False),
                f"ids_results_"
                f"{datetime.now():%Y%m%d_%H%M}.csv",
                "text/csv",
                use_container_width=True,
                key="download_tab2_csv")
        with c2:
            if "report" in m:
                st.download_button(
                    "⬇️ تحميل التقرير TXT",
                    m["report"],
                    f"ids_report_"
                    f"{datetime.now():%Y%m%d_%H%M}.txt",
                    "text/plain",
                    use_container_width=True,
                    key="download_tab2_txt")


# ══════════════════════════════════════════════════════════════
# Tab 3
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 🤖 كيف يعمل النظام")
    st.markdown("""
    ### 🧠 الفكرة الأساسية
    ارفع أي داتاست → اضغط **ابدأ التحليل** → النتائج فوراً.
    النظام يقرر تلقائياً هل يدرّب نموذجاً جديداً أم يستخدم
    النموذج الحالي — بدون تدخل من المستخدم.
    """)

    agents = [
        ("1","Universal Preprocessor","#2ecc71",
         "يقرأ أي CSV ويُنظّفه تلقائياً — يكتشف timestamps "
         "وIPs وdata leakage بالمنطق لا بالأسماء. "
         "يقسم البيانات لـ Train/Test ثم SMOTE على Train فقط."),
        ("2","Smart Feature Selector","#3498db",
         "Boruta + XGBoost + SHAP يختار أهم الـ features. "
         "من 46 عمود يختار 10 فقط."),
        ("3","Behavioral Anomaly Detector","#9b59b6",
         "Autoencoder يتعلم شكل الـ normal مع Data Augmentation "
         "+ IsolationForest."),
        ("4","FT-Transformer Classifier","#e74c3c",
         "يصنّف 10-34 نوع هجوم بدقة 93-98%. "
         "إذا كانت الداتاست جديدة يُستبدل بـ RandomForest "
         "تلقائياً."),
        ("5","Adaptive Learner","#f39c12",
         "KS Test يكتشف Concept Drift. "
         "EWC يحمي الأوزان. Reservoir Sampling للتنوع."),
        ("6","Decision Maker","#1abc9c",
         "يجمع نتائج الـ Agents: ALLOW / BLOCK / QUARANTINE."),
    ]

    for num, name, color, desc in agents:
        st.markdown(
            f'<div class="agent-card" '
            f'style="border-color:{color}">'
            f'<strong style="color:{color};'
            f'font-size:1.1rem;">'
            f'Agent {num} — {name}</strong><br><br>'
            f'{desc}</div>',
            unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔄 الـ Pipeline الكامل")
    st.code("""
ارفع أي CSV → اضغط "ابدأ التحليل"
     ↓
🔍 فحص التوافق مع النموذج الحالي
     ↓ متوافق (≥70% features)
  → تحليل فوري بالنموذج الأصلي

     ↓ غير متوافق (داتاست جديدة)
  → تدريب تلقائي RandomForest (خلف الكواليس)
  → تحليل بالنموذج الجديد
     ↓
النتائج: ALLOW / BLOCK / QUARANTINE + 10 رسوم
    """)

    st.markdown("### 📊 مقارنة الإصدارات")
    comparison = pd.DataFrame({
        "الميزة": [
            "أي داتاست","SMOTE صحيح",
            "Boruta+SHAP","Behavioral AE",
            "Persistent Memory","Auto Training",
            "Smart Detection","Accuracy"],
        "v1": ["❌","❌","✅","❌","✅",
               "❌","❌","97.7%"],
        "v2": ["✅","❌","❌","✅","❌",
               "❌","❌","68%"],
        "v3": ["✅","✅","✅","✅","✅",
               "✅","✅","93-99%"],
    })
    st.dataframe(comparison, use_container_width=True,
                 hide_index=True)


# ══════════════════════════════════════════════════════════════
# Tab 4
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 📋 عن المشروع")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🎯 الهدف
        نظام كشف هجمات شبكية يعمل على **أي داتاست**
        بضغطة زر واحدة — بدون تعديل يدوي.

        ### 📊 الداتاسيت المدعومة
        - **TON-IoT** — 10 أنواع هجمات
        - **CIC-IoT-2023** — 34 نوع هجوم
        - **UNSW-NB15** — 9 أنواع هجمات
        - **Bot-IoT** — 5 categories
        - **CICIDS2017** — 14 نوع هجوم
        - **أي CSV** — تدريب تلقائي

        ### ⚡ الأداء
        - Accuracy: **93-99%**
        - Macro F1: **80-92%**
        """)
    with col2:
        st.markdown("""
        ### 🛠️ التقنيات
        - **FT-Transformer** — Tabular Transformer
        - **Boruta + SHAP** — Feature Selection
        - **SMOTE-ENN** — Class Balancing
        - **Autoencoder + IForest** — Anomaly Detection
        - **ONNX Runtime** — Fast Inference
        - **RandomForest** — Auto Custom Training
        - **Smart Detection** — Auto label detection
        - **Persistent Memory** — تحسين مستمر

        ### 📞 التواصل

        - [🐙 GitHub](https://github.com/Muoz22/ids-network-analyzer)

        - [🎓 Google Scholar](https://scholar.google.com/citations?user=J35vcAIAAAAJ&hl=en)
        - [🐙 Researchgate](https://www.researchgate.net/profile/Muaadh-Alsoufi?ev=hdr_xprf)
   

    if models_global and "meta" in models_global:
        meta = models_global["meta"]
        st.markdown("---")
        st.markdown("### 🔍 معلومات النموذج الحالي")
        info_df = pd.DataFrame({
            "المعلومة": [
                "Accuracy","Weighted F1","Macro F1",
                "Features","Classes","Session #",
                "Trained on","Trained at"],
            "القيمة": [
                f"{meta.get('accuracy',0)*100:.2f}%",
                f"{meta.get('weighted_f1',0)*100:.2f}%",
                f"{meta.get('macro_f1',0)*100:.2f}%",
                str(meta.get('n_features',0)),
                str(meta.get('n_classes',0)),
                str(meta.get('session',0)),
                meta.get('trained_on','').split('/')[-2],
                meta.get('trained_at','')[:10],
            ]
        })
        st.dataframe(info_df, use_container_width=True,
                     hide_index=True)

    st.markdown("---")
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;padding:1rem;
    background:linear-gradient(135deg,#1a1a2e,#0f3460);
    border-radius:10px;">
    <p style="color:#00d4ff;font-size:1.2rem;
    font-weight:bold;margin:0;">
    🛡️ AI Agents IDS v3 Universal</p>
    <p style="color:#a0aec0;margin:0.3rem 0;">
    © 2025 <strong style="color:white;">
    Muaz Al-Soufi</strong> — All Rights Reserved</p>
    <p style="color:#a0aec0;margin:0.3rem 0;">
    <a href="https://github.com/Muoz22/ids-network-analyzer"
    style="color:#00d4ff;">
    github.com/Muoz22/ids-network-analyzer</a></p>
    <p style="color:#666;font-size:0.8rem;margin:0.3rem 0;">
    Built with ❤️ using Streamlit · ONNX Runtime ·
    FT-Transformer · RandomForest</p>
    </div>
    """, unsafe_allow_html=True)
