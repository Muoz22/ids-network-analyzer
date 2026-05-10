# ================================================================
# app.py — AI Agents IDS v3 — Network Security Analyzer
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import os, tempfile, json
from datetime import datetime

# ── إعداد الصفحة ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Agents IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────
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

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🛡️ AI Agents IDS</h1>
    <p>Network Intrusion Detection System — Powered by 6 AI Agents | v3 Universal</p>
</div>
""", unsafe_allow_html=True)

# ── Load Models ───────────────────────────────────────────────
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

    # اقرأ الـ defaults من metadata
    default_lc = "type"
    default_bl = "normal"
    if models_global and "meta" in models_global:
        meta = models_global["meta"]
        default_lc = meta.get("label_col", "type")
        default_bl = meta.get("benign_label", "normal")

    label_col = st.text_input(
        "اسم عمود الـ Label", value=default_lc)
    benign_label = st.text_input(
        "اسم الكلاس الطبيعي", value=default_bl)
    threshold = st.slider(
        "Unknown Threshold",
        min_value=0.3, max_value=0.9,
        value=0.6, step=0.05)

    st.markdown("---")
    st.markdown("### 📊 النموذج")
    if models_global and "meta" in models_global:
        meta = models_global["meta"]
        st.success("✅ النموذج محمّل")
        st.write(f"**Accuracy:** {meta.get('accuracy',0)*100:.2f}%")
        st.write(f"**Classes:** {meta.get('n_classes',0)}")
        st.write(f"**Features:** {meta.get('n_features',0)}")
        with st.expander("🔍 التفاصيل"):
            st.write(f"**Class Names:**")
            for c in meta.get("class_names",[]):
                st.write(f"  • {c}")
            st.write(f"**Features:**")
            for f in meta.get("feat_cols",[]):
                st.write(f"  • {f}")
    elif models_err:
        st.error(f"❌ {models_err}")

    st.markdown("---")
    st.markdown("### 🔗 روابط")
    st.markdown("[GitHub](https://github.com/Muoz22/ids-network-analyzer)")

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 تحليل الشبكة",
    "📊 تقرير تفصيلي",
    "📖 كيف يعمل",
    "📋 عن المشروع",
])

# ══════════════════════════════════════════════════════════════
# Tab 1 — التحليل السريع
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
                df = pd.read_csv(
                    uploaded_file, low_memory=False)
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
            st.write(f"**الأعمدة:** {df.columns.tolist()}")

        if label_col in df.columns:
            st.info(f"✅ عمود '{label_col}' موجود — "
                    f"{df[label_col].value_counts().head(5).to_dict()}")
        else:
            st.warning(f"⚠️ عمود '{label_col}' غير موجود")

        if st.button("🚀 ابدأ التحليل",
                     type="primary",
                     use_container_width=True):

            models, err = load_models_cached(version="v3")
            if err:
                st.error(f"❌ {err}")
                st.stop()

            progress = st.progress(0)
            status   = st.empty()

            with st.spinner("🔮 جاري التحليل..."):
                status.text("⚙️ تنظيف البيانات...")
                progress.progress(20)

                from inference import run_inference, make_plots
                with tempfile.TemporaryDirectory() as tmp:
                    status.text("🤖 تشغيل النماذج...")
                    progress.progress(50)

                    results = run_inference(
                        df, models, label_col,
                        benign_label,
                        ft_unk_thr=threshold)

                    status.text("📊 إنتاج الرسوم...")
                    progress.progress(80)

                    plot_paths = make_plots(
                        results, benign_label,
                        out_dir=tmp)

                    progress.progress(100)
                    status.text("✅ اكتمل!")

                    # حفظ النتائج في session state
                    st.session_state["results"]    = results
                    st.session_state["plot_paths"] = plot_paths
                    st.session_state["df"]         = df

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
                        st.metric("✅ Benign",
                                  f"{benign:,}",
                                  f"{100*benign/total:.1f}%")
                    with c3:
                        st.metric("🚫 هجمات",
                                  f"{attacks:,}",
                                  f"{100*attacks/total:.1f}%",
                                  delta_color="inverse")
                    with c4:
                        st.metric("⚠️ Unknown",
                                  f"{unknown:,}",
                                  f"{100*unknown/total:.1f}%",
                                  delta_color="inverse")

                    # ── دقة النموذج ───────────────────────────
                    m = results["metrics"]
                    if "accuracy" in m:
                        st.markdown("### 📈 دقة النموذج")
                        mc1,mc2,mc3 = st.columns(3)
                        with mc1:
                            acc = m["accuracy"]*100
                            cl  = ("status-excellent"
                                   if acc>97 else
                                   "status-good" if acc>93
                                   else "status-warning")
                            st.markdown(
                                f'<p class="metric-value {cl}">'
                                f'{acc:.2f}%</p>'
                                f'<p class="metric-label">'
                                f'Accuracy</p>',
                                unsafe_allow_html=True)
                        with mc2:
                            f1 = m["weighted_f1"]*100
                            st.markdown(
                                f'<p class="metric-value '
                                f'status-good">{f1:.2f}%</p>'
                                f'<p class="metric-label">'
                                f'Weighted F1</p>',
                                unsafe_allow_html=True)
                        with mc3:
                            mf1 = m["macro_f1"]*100
                            cl2 = ("status-excellent"
                                   if mf1>80 else
                                   "status-good" if mf1>60
                                   else "status-warning")
                            st.markdown(
                                f'<p class="metric-value {cl2}">'
                                f'{mf1:.2f}%</p>'
                                f'<p class="metric-label">'
                                f'Macro F1</p>',
                                unsafe_allow_html=True)

                    # ── أنواع الهجمات ─────────────────────────
                    if results["atk_counts"]:
                        st.markdown(
                            "### 🔴 أنواع الهجمات المكتشفة")
                        atk_df = pd.DataFrame(
                            results["atk_counts"].most_common(20),
                            columns=["نوع الهجوم","العدد"])
                        atk_df["النسبة %"] = (
                            atk_df["العدد"]/total*100
                        ).round(2)
                        st.dataframe(atk_df,
                                     use_container_width=True)

                    # ── الرسوم الأساسية ───────────────────────
                    if plot_paths:
                        st.markdown("### 📊 الرسوم البيانية")
                        for i in range(0, len(plot_paths), 2):
                            cols = st.columns(2)
                            for j, (title, path) in \
                                    enumerate(plot_paths[i:i+2]):
                                with cols[j]:
                                    st.markdown(f"**{title}**")
                                    st.image(path,
                                             use_column_width=True)

                    # ── تقرير Classification ───────────────────
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
                        use_container_width=True)

                    st.info("💡 انتقل لـ **📊 تقرير تفصيلي** "
                            "لرؤية كل الرسوم والتحليل الكامل")

# ══════════════════════════════════════════════════════════════
# Tab 2 — تقرير تفصيلي
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 📊 تقرير تفصيلي")

    if "results" not in st.session_state:
        st.info("🔍 ارفع ملف في **تحليل الشبكة** أولاً")
    else:
        results    = st.session_state["results"]
        plot_paths = st.session_state["plot_paths"]
        m          = results["metrics"]

        # ── ملخص سريع ─────────────────────────────────────────
        total   = results["n_samples"]
        benign  = results["n_benign"]
        attacks = results["n_attacks"]
        unknown = results["n_unknown"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("✅ ALLOW",  f"{benign:,}",
                      f"{100*benign/total:.1f}%")
        with col2:
            st.metric("🚫 BLOCK",  f"{attacks:,}",
                      f"{100*attacks/total:.1f}%",
                      delta_color="inverse")
        with col3:
            st.metric("⚠️ QUARANTINE", f"{unknown:,}",
                      f"{100*unknown/total:.1f}%",
                      delta_color="inverse")

        # ── Section 1: Decision Analysis ──────────────────────
        st.markdown("---")
        st.markdown("### 🎯 Decision Analysis")

        dc1, dc2 = st.columns(2)
        with dc1:
            # Pie chart
            pie_plots = [(t,p) for t,p in plot_paths
                         if "Distribution" in t or
                         "Pie" in t or "pie" in p.lower()]
            if pie_plots:
                st.markdown("**Decision Distribution**")
                st.image(pie_plots[0][1],
                         use_column_width=True)
        with dc2:
            # Attack bar
            atk_plots = [(t,p) for t,p in plot_paths
                         if "Attack" in t or
                         "attack" in p.lower()]
            if atk_plots:
                st.markdown("**Attack Types**")
                st.image(atk_plots[0][1],
                         use_column_width=True)

        # ── Section 2: Classification Performance ─────────────
        if "accuracy" in m:
            st.markdown("---")
            st.markdown("### 📈 Classification Performance")

            mc1, mc2, mc3 = st.columns(3)
            metrics_list = [
                ("Accuracy",    m["accuracy"]*100,    97, 93),
                ("Weighted F1", m["weighted_f1"]*100, 95, 90),
                ("Macro F1",    m["macro_f1"]*100,    80, 60),
            ]
            for col, (name, val, hi, lo) in zip(
                    [mc1,mc2,mc3], metrics_list):
                with col:
                    cl = ("status-excellent" if val>hi
                          else "status-good" if val>lo
                          else "status-warning")
                    status = ("🟢 Excellent" if val>hi
                              else "🟡 Good" if val>lo
                              else "🔴 Fair")
                    st.markdown(
                        f'<p class="metric-value {cl}">'
                        f'{val:.2f}%</p>'
                        f'<p class="metric-label">'
                        f'{name}<br>{status}</p>',
                        unsafe_allow_html=True)

            # Confusion + F1
            conf_plots = [(t,p) for t,p in plot_paths
                          if "Confusion" in t or
                          "confusion" in p.lower()]
            f1_plots   = [(t,p) for t,p in plot_paths
                          if "F1" in t or "f1" in p.lower()]

            if conf_plots or f1_plots:
                pfc1, pfc2 = st.columns(2)
                with pfc1:
                    if conf_plots:
                        st.markdown("**Confusion Matrix**")
                        st.image(conf_plots[0][1],
                                 use_column_width=True)
                with pfc2:
                    if f1_plots:
                        st.markdown("**Per-Class F1**")
                        st.image(f1_plots[0][1],
                                 use_column_width=True)

            # Classification Report
            if "report" in m:
                st.markdown("---")
                st.markdown("### 📋 Classification Report")
                st.code(m["report"])

        # ── Section 3: Confidence Analysis ────────────────────
        st.markdown("---")
        st.markdown("### 🎚️ Confidence Analysis")

        conf_plots = [(t,p) for t,p in plot_paths
                      if "Confidence" in t or
                      "confidence" in p.lower()]
        if conf_plots:
            st.image(conf_plots[0][1],
                     use_column_width=True)

        # ── Section 4: All Plots ───────────────────────────────
        st.markdown("---")
        st.markdown("### 🖼️ كل الرسوم البيانية")

        for i in range(0, len(plot_paths), 2):
            cols = st.columns(2)
            for j, (title, path) in enumerate(
                    plot_paths[i:i+2]):
                with cols[j]:
                    st.markdown(f"**{title}**")
                    st.image(path, use_column_width=True)

        # ── Section 5: Raw Data ───────────────────────────────
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
                use_container_width=True)
        with c2:
            if "report" in m:
                st.download_button(
                    "⬇️ تحميل التقرير TXT",
                    m["report"],
                    f"ids_report_"
                    f"{datetime.now():%Y%m%d_%H%M}.txt",
                    "text/plain",
                    use_container_width=True)

# ══════════════════════════════════════════════════════════════
# Tab 3 — كيف يعمل
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
        ("1", "Universal Preprocessor", "#2ecc71",
         "يقرأ أي CSV ويُنظّفه تلقائياً — يكتشف timestamps وIPs "
         "وdata leakage بالمنطق لا بالأسماء. "
         "يقسم البيانات لـ Train/Test ثم يطبق SMOTE-ENN على Train فقط."),
        ("2", "Smart Feature Selector", "#3498db",
         "Boruta + XGBoost + SHAP يختار أهم الـ features تلقائياً. "
         "من 46 عمود يختار 10 فقط بناءً على الأهمية الحقيقية."),
        ("3", "Behavioral Anomaly Detector", "#9b59b6",
         "Autoencoder يتعلم شكل الـ normal traffic مع Data Augmentation. "
         "+ IsolationForest لكشف النقاط الشاذة. "
         "الـ threshold يتحسّن تلقائياً من الذاكرة."),
        ("4", "FT-Transformer Classifier", "#e74c3c",
         "Feature Tokenizer + CLS Token + Multi-Head Attention "
         "يصنّف 10-34 نوع هجوم بدقة 93-98%. "
         "يعمل على أي عدد features وأي عدد classes."),
        ("5", "Adaptive Learner", "#f39c12",
         "KS Test يكتشف Concept Drift. "
         "EWC يحمي الأوزان القديمة. "
         "Reservoir Sampling يحتفظ بعينات متنوعة."),
        ("6", "Decision Maker", "#1abc9c",
         "يجمع نتائج كل الـ Agents ويتخذ قرار: "
         "ALLOW / BLOCK / QUARANTINE. "
         "Priority Attacks من الذاكرة تحصل على BLOCK فوري."),
    ]

    for num, name, color, desc in agents:
        st.markdown(
            f'<div class="agent-card" '
            f'style="border-color:{color}">'
            f'<strong style="color:{color}; font-size:1.1rem;">'
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
            "أي داتاست", "SMOTE صحيح",
            "Boruta+SHAP", "Behavioral AE",
            "Persistent Memory", "Accuracy"
        ],
        "v1": ["❌", "❌", "✅", "❌", "✅", "97.7%"],
        "v2": ["✅", "❌", "❌", "✅", "❌", "68%"],
        "v3": ["✅", "✅", "✅", "✅", "✅", "93-98%"],
    })
    st.dataframe(comparison, use_container_width=True,
                 hide_index=True)

# ══════════════════════════════════════════════════════════════
# Tab 4 — عن المشروع
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
        - يعمل على أي CSV بدون إعداد مسبق
        """)

    with col2:
        st.markdown("""
        ### 🛠️ التقنيات
        - **FT-Transformer** — Tabular Transformer
        - **Boruta + SHAP** — Feature Selection
        - **SMOTE-ENN** — Class Balancing (Train only)
        - **Autoencoder + IForest** — Anomaly Detection
        - **ONNX Runtime** — Fast Inference
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
                "Accuracy", "Weighted F1", "Macro F1",
                "Features", "Classes", "Session #",
                "Trained on", "Trained at"
            ],
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
