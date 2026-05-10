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
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models_cached(version="v3"):
    try:
        from inference import load_models
        models = load_models(model_dir="models/")
        return models, None
    except Exception as e:
        return None, str(e)

models_global, models_err = load_models_cached(version="v3")

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

    # عرض معلومات النموذج النشط
    if "custom_meta" in st.session_state:
        st.warning("🔧 النموذج المخصص نشط")
        cm = st.session_state["custom_meta"]
        st.write(f"**Classes:** {cm['n_classes']}")
        st.write(f"**Features:** "
                 f"{len(cm['features'])}")
        acc = cm["metrics"].get("accuracy", 0)*100
        st.write(f"**Accuracy:** {acc:.2f}%")
    elif models_global and "meta" in models_global:
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


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 تحليل الشبكة",
    "📊 تقرير تفصيلي",
    "📖 كيف يعمل",
    "📋 عن المشروع",
    "🔧 Train Custom Model",
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

        if label_col in df.columns:
            st.info(
                f"✅ عمود '{label_col}' موجود — "
                f"{df[label_col].value_counts().head(5).to_dict()}")
        else:
            st.warning(
                f"⚠️ عمود '{label_col}' غير موجود")

        if st.button("🚀 ابدأ التحليل",
                     type="primary",
                     use_container_width=True,
                     key="btn_analyze"):

            progress = st.progress(0)
            status   = st.empty()

            with st.spinner("🔮 جاري التحليل..."):
                status.text("⚙️ تنظيف البيانات...")
                progress.progress(20)

                from inference import (
                    run_inference,
                    run_inference_custom,
                    make_plots)

                status.text("🤖 تشغيل النماذج...")
                progress.progress(50)

                # ── اختيار النموذج ────────────────────────
                if "custom_model" in st.session_state:
                    results = run_inference_custom(
                        df,
                        st.session_state["custom_model"],
                        label_col,
                        benign_label,
                        ft_unk_thr=threshold)
                    st.sidebar.success(
                        "🔧 يستخدم النموذج المخصص")
                else:
                    models, err = load_models_cached(
                        version="v3")
                    if err:
                        st.error(f"❌ {err}")
                        st.stop()
                    results = run_inference(
                        df, models, label_col,
                        benign_label,
                        ft_unk_thr=threshold)
                    st.sidebar.info(
                        "🔵 يستخدم النموذج الأصلي (TON-IoT)")

                status.text("📊 إنتاج الرسوم...")
                progress.progress(80)

                with tempfile.TemporaryDirectory() as tmp:
                    plot_paths = make_plots(
                        results, benign_label,
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

                # ── دقة النموذج ───────────────────────────
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

                # ── أنواع الهجمات ─────────────────────────
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

                # ── الرسوم ────────────────────────────────
                if plot_bytes:
                    st.markdown("### 📊 الرسوم البيانية")
                    for i in range(0, len(plot_bytes), 2):
                        cols = st.columns(2)
                        for j, (title, img) in enumerate(
                                plot_bytes[i:i+2]):
                            with cols[j]:
                                st.markdown(f"**{title}**")
                                st.image(
                                    img,
                                    use_column_width=True)

                # ── تقرير ─────────────────────────────────
                if "report" in m:
                    with st.expander(
                            "📋 تقرير Classification كامل"):
                        st.code(m["report"])

                # ── تفاصيل ────────────────────────────────
                with st.expander("🔍 تفاصيل المعالجة"):
                    st.write(f"**وقت التنفيذ:** "
                             f"{results['elapsed_sec']}s")
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

                # ── تحميل ─────────────────────────────────
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

        # ── Decision Analysis ──────────────────────────────
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

        # ── Classification Performance ─────────────────────
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
                st.markdown(
                    "### 📋 Classification Report")
                st.code(m["report"])

        # ── Confidence Analysis ────────────────────────────
        st.markdown("---")
        st.markdown("### 🎚️ Confidence Analysis")
        conf_imgs = [(t,img) for t,img in plot_bytes
                     if "Confidence" in t]
        if conf_imgs:
            st.image(conf_imgs[0][1],
                     use_column_width=True)

        # ── Drift Analysis ─────────────────────────────────
        st.markdown("---")
        st.markdown("### 🌊 Drift Analysis")
        drift_imgs = [(t,img) for t,img in plot_bytes
                      if "Drift" in t]
        if drift_imgs:
            st.image(drift_imgs[0][1],
                     use_column_width=True)

        # ── Summary Dashboard ──────────────────────────────
        st.markdown("---")
        st.markdown("### 🗂️ Summary Dashboard")
        dash_imgs = [(t,img) for t,img in plot_bytes
                     if "Dashboard" in t or
                     "Summary" in t]
        if dash_imgs:
            st.image(dash_imgs[0][1],
                     use_column_width=True)

        # ── All Plots ──────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🖼️ كل الرسوم البيانية")
        for i in range(0, len(plot_bytes), 2):
            cols = st.columns(2)
            for j, (title, img) in enumerate(
                    plot_bytes[i:i+2]):
                with cols[j]:
                    st.markdown(f"**{title}**")
                    st.image(img,
                             use_column_width=True)

        # ── Download ───────────────────────────────────────
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
    بدل نموذج يتعلم "هذا dos لأن src_ip_bytes=X"،
    النظام يتعلم **شكل الـ normal traffic** ثم يكتشف
    أي انحراف عنه — بغض النظر عن اسم الـ feature.
    """)

    agents = [
        ("1","Universal Preprocessor","#2ecc71",
         "يقرأ أي CSV ويُنظّفه تلقائياً — يكتشف timestamps "
         "وIPs وdata leakage بالمنطق لا بالأسماء. "
         "يقسم البيانات لـ Train/Test ثم SMOTE على Train فقط."),
        ("2","Smart Feature Selector","#3498db",
         "Boruta + XGBoost + SHAP يختار أهم الـ features. "
         "من 46 عمود يختار 10 فقط بناءً على الأهمية الحقيقية."),
        ("3","Behavioral Anomaly Detector","#9b59b6",
         "Autoencoder يتعلم شكل الـ normal مع Data Augmentation "
         "+ IsolationForest. الـ threshold من الذاكرة."),
        ("4","FT-Transformer Classifier","#e74c3c",
         "Feature Tokenizer + CLS Token + Multi-Head Attention "
         "يصنّف 10-34 نوع هجوم بدقة 93-98%."),
        ("5","Adaptive Learner","#f39c12",
         "KS Test يكتشف Concept Drift. "
         "EWC يحمي الأوزان. Reservoir Sampling للتنوع."),
        ("6","Decision Maker","#1abc9c",
         "يجمع نتائج الـ Agents: ALLOW / BLOCK / QUARANTINE. "
         "Priority Attacks من الذاكرة تُحجب فوراً."),
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
أي CSV جديد
     ↓
Agent 1: Auto Filter + RobustScaler + SMOTE (Train only)
     ↓
Agent 2: Boruta + XGBoost + SHAP → Top K Features
     ↓
Agent 3: AE + IForest → Anomaly Score
     ↓
Agent 4: FT-Transformer → Attack Classification
     ↓
Agent 5: KS Drift Detection + EWC + Reservoir
     ↓
Agent 6: ALLOW / BLOCK / QUARANTINE
    """)

    st.markdown("### 📊 مقارنة الإصدارات")
    comparison = pd.DataFrame({
        "الميزة": [
            "أي داتاست","SMOTE صحيح",
            "Boruta+SHAP","Behavioral AE",
            "Persistent Memory","Train Custom",
            "Accuracy"],
        "v1": ["❌","❌","✅","❌","✅","❌","97.7%"],
        "v2": ["✅","❌","❌","✅","❌","❌","68%"],
        "v3": ["✅","✅","✅","✅","✅","✅","93-98%"],
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
        نظام كشف هجمات شبكية مبني على 6 AI Agents
        يعمل على **أي داتاست** بدون تعديل يدوي.

        ### 📊 الداتاسيت المستخدمة
        - **TON-IoT** — 10 أنواع هجمات
        - **CIC-IoT-2023** — 34 نوع هجوم

        ### ⚡ الأداء
        - Accuracy: **93-98%**
        - Macro F1: **82-92%**
        """)
    with col2:
        st.markdown("""
        ### 🛠️ التقنيات
        - **FT-Transformer** — Tabular Transformer
        - **Boruta + SHAP** — Feature Selection
        - **SMOTE-ENN** — Class Balancing (Train only)
        - **Autoencoder + IForest** — Anomaly Detection
        - **ONNX Runtime** — Fast Inference
        - **XGBoost** — Custom Model Training
        - **Persistent Memory** — تحسين مستمر

        ### 📞 التواصل
        - [GitHub](https://github.com/Muoz22/ids-network-analyzer)
        """)

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
    st.markdown(
        '<p style="text-align:center;color:#666;">'
        'Built with ❤️ using Streamlit + ONNX Runtime'
        '</p>',
        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# Tab 5 — Train Custom Model
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## 🔧 Train Custom Model")
    st.info(
        "ارفع داتاست جديدة وسيتدرب النظام عليها "
        "تلقائياً في 30-60 ثانية باستخدام XGBoost. "
        "النموذج الأصلي لن يتأثر.")

    if "custom_model" in st.session_state:
        cm = st.session_state["custom_meta"]
        st.success(
            f"✅ نموذج مخصص محمّل — "
            f"Classes: {cm['classes']}  "
            f"Accuracy: "
            f"{cm['metrics'].get('accuracy',0)*100:.1f}%")
        if st.button(
                "🗑️ حذف النموذج المخصص والرجوع "
                "للنموذج الأصلي",
                key="btn_delete_custom"):
            del st.session_state["custom_model"]
            del st.session_state["custom_meta"]
            st.success(
                "✅ تم الحذف — النموذج الأصلي نشط")
            st.rerun()

    st.markdown("---")
    st.markdown("### 📁 ارفع داتاست التدريب")

    col1, col2 = st.columns([2, 1])
    with col1:
        train_file = st.file_uploader(
            "اختر ملف CSV للتدريب",
            type=["csv"],
            key="train_uploader")
    with col2:
        st.markdown("""
        **متطلبات الملف:**
        - يحتوي على عمود الـ label
        - على الأقل 500 صف
        - يفضل 2,000-10,000 صف
        - أي network traffic features
        """)

    if train_file is not None:
        with st.spinner("جاري قراءة الملف..."):
            try:
                df_train = pd.read_csv(
                    train_file, low_memory=False)
                st.success(
                    f"✅ تم تحميل الملف: "
                    f"{df_train.shape[0]:,} صف × "
                    f"{df_train.shape[1]} عمود")
            except Exception as e:
                st.error(f"❌ خطأ: {e}")
                st.stop()

        with st.expander("👁️ معاينة البيانات",
                         expanded=False):
            st.dataframe(df_train.head(5),
                         use_container_width=True)

        st.markdown("### ⚙️ إعدادات التدريب")
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            train_label = st.text_input(
                "عمود الـ Label",
                value=label_col,
                key="train_label_col")
        with tc2:
            train_benign = st.text_input(
                "الكلاس الطبيعي",
                value=benign_label,
                key="train_benign_label")
        with tc3:
            max_rows = st.number_input(
                "أقصى عدد صفوف",
                min_value=500,
                max_value=50000,
                value=10000,
                step=500)

        if train_label in df_train.columns:
            vc = df_train[train_label].value_counts()
            st.info(
                f"✅ عمود '{train_label}' موجود — "
                f"{len(vc)} كلاس: "
                f"{vc.head(5).to_dict()}")
            fig, ax = plt.subplots(figsize=(10, 3))
            vc.plot(kind="barh", ax=ax,
                    color="#3498db", alpha=0.8)
            ax.set_title("توزيع الكلاسات",
                         fontweight="bold")
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.warning(
                f"⚠️ عمود '{train_label}' غير موجود")

        st.markdown("---")
        if st.button("🚀 ابدأ التدريب",
                     type="primary",
                     use_container_width=True,
                     key="btn_train"):

            if train_label not in df_train.columns:
                st.error(
                    f"❌ عمود '{train_label}' "
                    f"غير موجود")
                st.stop()

            if len(df_train) < 100:
                st.error(
                    "❌ الداتاست صغيرة جداً — "
                    "يجب على الأقل 100 صف")
                st.stop()

            from inference import train_custom_model

            progress = st.progress(0)
            status   = st.empty()

            with st.spinner(
                    "🔧 جاري التدريب على CPU..."):
                status.text("⚙️ تحليل البيانات...")
                progress.progress(10)
                status.text("🔬 Feature Selection...")
                progress.progress(30)
                status.text("🤖 تدريب XGBoost...")
                progress.progress(50)

                train_results = train_custom_model(
                    df_train,
                    label_col    = train_label,
                    benign_label = train_benign,
                    max_rows     = max_rows)

                progress.progress(90)

                if train_results["success"]:
                    st.session_state[
                        "custom_model"] = {
                        "model"   : train_results["model"],
                        "scaler"  : train_results["scaler"],
                        "le"      : train_results["le"],
                        "features": train_results["features"],
                    }
                    st.session_state[
                        "custom_meta"] = {
                        "classes"     : train_results["classes"],
                        "n_classes"   : train_results["n_classes"],
                        "n_samples"   : train_results["n_samples"],
                        "features"    : train_results["features"],
                        "metrics"     : train_results["metrics"],
                        "label_col"   : train_label,
                        "benign_label": train_benign,
                    }

                    progress.progress(100)
                    status.text("✅ اكتمل التدريب!")

                    st.markdown("---")
                    st.markdown("## 🏆 نتائج التدريب")

                    m = train_results["metrics"]
                    rc1,rc2,rc3 = st.columns(3)
                    for col, (name, val, hi, lo) in zip(
                            [rc1,rc2,rc3], [
                                ("Accuracy",
                                 m["accuracy"]*100,90,80),
                                ("Weighted F1",
                                 m["weighted_f1"]*100,90,80),
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

                    ic1,ic2,ic3 = st.columns(3)
                    with ic1:
                        st.metric(
                            "عدد الكلاسات",
                            train_results["n_classes"])
                    with ic2:
                        st.metric(
                            "عدد الـ Features",
                            len(train_results["features"]))
                    with ic3:
                        st.metric(
                            "عدد العينات",
                            f"{train_results['n_samples']:,}")

                    st.markdown(
                        "### 📋 Classification Report")
                    st.code(m["report"])

                    if train_results["removed_cols"]:
                        st.info(
                            f"**أعمدة مستُبعدت:** "
                            f"{list(train_results['removed_cols'].keys())}")

                    st.success(
                        "✅ النموذج المخصص جاهز! "
                        "انتقل لـ **🔍 تحليل الشبكة** "
                        "وارفع ملفاً من نفس الداتاست.")
                else:
                    progress.progress(0)
                    st.error(train_results["message"])
