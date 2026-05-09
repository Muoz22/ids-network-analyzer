# ================================================================
# app.py — AI Agents IDS — Network Security Analyzer
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

# ── CSS مخصص ──────────────────────────────────────────────────
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    text-align: center;
}
.main-header h1 { color: #00d4ff; font-size: 2.5rem; margin: 0; }
.main-header p  { color: #a0aec0; font-size: 1.1rem; margin: 0.5rem 0 0; }
.metric-card {
    background: #1a1a2e;
    border: 1px solid #0f3460;
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
}
.metric-value { font-size: 2rem; font-weight: bold; margin: 0; }
.metric-label { color: #a0aec0; font-size: 0.9rem; }
.status-excellent { color: #00d4ff; }
.status-good      { color: #48bb78; }
.status-warning   { color: #ed8936; }
.attack-bar {
    background: #2d3748;
    border-left: 4px solid #e74c3c;
    padding: 0.5rem 1rem;
    margin: 0.3rem 0;
    border-radius: 0 6px 6px 0;
}
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🛡️ AI Agents IDS</h1>
    <p>Network Intrusion Detection System — Powered by 6 AI Agents</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/AI%20Agents-6%20Agents-blue",
             use_column_width=True)
    st.markdown("---")
    st.markdown("### ⚙️ إعدادات التحليل")

    label_col = st.text_input(
        "اسم عمود الـ Label",
        value="type",
        help="اسم العمود الذي يحتوي على نوع الهجوم أو Normal")

    benign_label = st.text_input(
        "اسم الكلاس الطبيعي",
        value="normal",
        help="القيمة التي تمثل الـ traffic الطبيعي")

    threshold = st.slider(
        "Unknown Threshold",
        min_value=0.3,
        max_value=0.9,
        value=0.6,
        step=0.05,
        help="الـ confidence أقل من هذه القيمة يُصنَّف كـ Unknown")

    st.markdown("---")
    st.markdown("### 📊 النموذج")
    st.markdown("""
    - **Architecture**: FT-Transformer
    - **Agents**: 6 AI Agents
    - **Datasets**: TON-IoT, CIC-IoT
    - **Accuracy**: 97-99%
    """)

    st.markdown("---")
    st.markdown("### 🔗 روابط")
    st.markdown("[GitHub](https://github.com/Muoz22/ids-network-analyzer)")


# ── تحميل النماذج ──────────────────────────────────────────────
@st.cache_resource
def load_models_cached():
    """تحميل النماذج مرة واحدة فقط"""
    try:
        from inference import load_models
        models = load_models(model_dir="models/")
        return models, None
    except Exception as e:
        return None, str(e)


# ── Main Content ───────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔍 تحليل الشبكة",
    "📖 كيف يعمل النظام",
    "📋 عن المشروع"
])


# ══════════════════════════════════════════════════════
# Tab 1 — التحليل
# ══════════════════════════════════════════════════════
with tab1:
    st.markdown("## 📁 ارفع ملف CSV للتحليل")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "اختر ملف CSV يحتوي على network traffic data",
            type=["csv"],
            help="يدعم أي داتاست network traffic بأي format")

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
        # ── اقرأ الملف ────────────────────────────────────────
        with st.spinner("جاري قراءة الملف..."):
            try:
                df = pd.read_csv(uploaded_file, low_memory=False)
                st.success(f"✅ تم تحميل الملف: "
                           f"{df.shape[0]:,} صف × {df.shape[1]} عمود")
            except Exception as e:
                st.error(f"❌ خطأ في قراءة الملف: {e}")
                st.stop()

        # ── معاينة البيانات ───────────────────────────────────
        with st.expander("👁️ معاينة البيانات", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
            st.write(f"**الأعمدة:** {df.columns.tolist()}")

        # ── فحص عمود الـ label ────────────────────────────────
        if label_col in df.columns:
            st.info(f"✅ عمود '{label_col}' موجود — "
                    f"القيم: {df[label_col].value_counts().head(5).to_dict()}")
        else:
            st.warning(f"⚠️ عمود '{label_col}' غير موجود — "
                       f"سيعمل النظام بدون تقرير الدقة")

        # ── زر التحليل ────────────────────────────────────────
        if st.button("🚀 ابدأ التحليل", type="primary",
                     use_container_width=True):

            # تحميل النماذج
            models, err = load_models_cached()
            if err:
                st.error(f"❌ خطأ في تحميل النماذج: {err}")
                st.info("💡 تأكد من رفع ملفات النماذج في مجلد models/")
                st.stop()

            # تشغيل الـ inference
            progress = st.progress(0)
            status   = st.empty()

            with st.spinner("🔮 جاري تحليل البيانات..."):
                status.text("⚙️ تنظيف البيانات...")
                progress.progress(20)

                from inference import run_inference, make_plots
                with tempfile.TemporaryDirectory() as tmp:
                    status.text("🤖 تشغيل النماذج...")
                    progress.progress(50)

                    results = run_inference(
                        df, models, label_col, benign_label,
                        ft_unk_thr=threshold)

                    status.text("📊 إنتاج التقارير...")
                    progress.progress(80)

                    plot_paths = make_plots(
                        results, benign_label, out_dir=tmp)

                    progress.progress(100)
                    status.text("✅ اكتمل التحليل!")

                    # ── عرض النتائج ───────────────────────────
                    st.markdown("---")
                    st.markdown("## 🏆 نتائج التحليل")

                    # بطاقات الإحصاءات
                    c1, c2, c3, c4 = st.columns(4)
                    total   = results["n_samples"]
                    benign  = results["n_benign"]
                    attacks = results["n_attacks"]
                    unknown = results["n_unknown"]

                    with c1:
                        st.metric("إجمالي العينات", f"{total:,}")
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

                    # دقة النموذج لو عندنا labels
                    m = results["metrics"]
                    if "accuracy" in m:
                        st.markdown("### 📈 دقة النموذج")
                        mc1, mc2, mc3 = st.columns(3)
                        with mc1:
                            acc = m["accuracy"]*100
                            color = ("status-excellent" if acc > 97
                                     else "status-good" if acc > 93
                                     else "status-warning")
                            st.markdown(
                                f'<p class="metric-value {color}">'
                                f'{acc:.2f}%</p>'
                                f'<p class="metric-label">Accuracy</p>',
                                unsafe_allow_html=True)
                        with mc2:
                            f1 = m["weighted_f1"]*100
                            st.markdown(
                                f'<p class="metric-value status-good">'
                                f'{f1:.2f}%</p>'
                                f'<p class="metric-label">Weighted F1</p>',
                                unsafe_allow_html=True)
                        with mc3:
                            mf1 = m["macro_f1"]*100
                            st.markdown(
                                f'<p class="metric-value">'
                                f'{mf1:.2f}%</p>'
                                f'<p class="metric-label">Macro F1</p>',
                                unsafe_allow_html=True)

                    # أنواع الهجمات
                    if results["atk_counts"]:
                        st.markdown("### 🔴 أنواع الهجمات المكتشفة")
                        atk_df = pd.DataFrame(
                            results["atk_counts"].most_common(20),
                            columns=["نوع الهجوم", "العدد"])
                        atk_df["النسبة %"] = (
                            atk_df["العدد"] / total * 100
                        ).round(2)
                        st.dataframe(
                            atk_df, use_container_width=True)

                    # الرسوم البيانية
                    if plot_paths:
                        st.markdown("### 📊 الرسوم البيانية")
                        for i in range(0, len(plot_paths), 2):
                            cols = st.columns(2)
                            for j, (title, path) in enumerate(
                                    plot_paths[i:i+2]):
                                with cols[j]:
                                    st.markdown(f"**{title}**")
                                    st.image(path,
                                             use_column_width=True)

                    # تقرير مفصّل
                    if "report" in m:
                        with st.expander(
                                "📋 تقرير Classification كامل"):
                            st.code(m["report"])

                    # معلومات إضافية
                    with st.expander("🔍 تفاصيل المعالجة"):
                        st.write(f"**وقت التنفيذ:** "
                                 f"{results['elapsed_sec']} ثانية")
                        st.write(f"**Features مطابقة:** "
                                 f"{results['matched_feats']}")
                        if results["missing_feats"]:
                            st.warning(
                                f"**Features مفقودة:** "
                                f"{results['missing_feats']}")
                        if results["removed_cols"]:
                            st.info(
                                f"**أعمدة مستُبعدت تلقائياً:** "
                                f"{list(results['removed_cols'].keys())}")

                    # تحميل النتائج
                    st.markdown("### 💾 تحميل النتائج")
                    result_df = pd.DataFrame({
                        "prediction" : results["y_pred"],
                        "confidence" : results["y_conf"],
                        "is_unknown" : results["y_unknown"],
                    })
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="⬇️ تحميل النتائج CSV",
                        data=csv,
                        file_name=f"ids_results_{datetime.now():%Y%m%d_%H%M}.csv",
                        mime="text/csv",
                        use_container_width=True)


# ══════════════════════════════════════════════════════
# Tab 2 — كيف يعمل النظام
# ══════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🤖 كيف يعمل النظام")

    agents = [
        ("1", "Data Preprocessor", "#2ecc71",
         "يُنظّف البيانات تلقائياً ويكتشف الأعمدة الضارة "
         "(timestamps, IPs, data leakage) بالمنطق لا بالأسماء."),
        ("2", "Feature Selector", "#3498db",
         "يختار أهم الـ features باستخدام Boruta + XGBoost + SHAP. "
         "من 46 عمود يختار 12 فقط."),
        ("3", "Anomaly Detector", "#9b59b6",
         "يتعلم شكل الـ traffic الطبيعي عبر Autoencoder + IForest. "
         "أي شيء مختلف يُعلَّم عليه."),
        ("4", "Attack Classifier", "#e74c3c",
         "FT-Transformer يصنّف أنواع الهجمات بدقة 97-99%. "
         "يدعم 34 نوع هجوم مختلف."),
        ("5", "Adaptive Learner", "#f39c12",
         "يراقب Concept Drift ويُنبّه عند تغيّر طبيعة البيانات. "
         "يستخدم KS test + EWC."),
        ("6", "Decision Maker", "#1abc9c",
         "يأخذ قرار نهائي: ALLOW / BLOCK / QUARANTINE. "
         "يُنتج تقريراً كاملاً بـ 22 رسماً."),
    ]

    for num, name, color, desc in agents:
        st.markdown(
            f'<div style="border-left: 4px solid {color}; '
            f'padding: 1rem; margin: 0.5rem 0; '
            f'background: rgba(0,0,0,0.05); border-radius: 0 8px 8px 0;">'
            f'<strong style="color:{color}">Agent {num} — {name}</strong><br>'
            f'{desc}</div>',
            unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# Tab 3 — عن المشروع
# ══════════════════════════════════════════════════════
with tab3:
    st.markdown("## 📋 عن المشروع")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🎯 الهدف
        نظام كشف هجمات شبكية مبني على 6 AI Agents
        تعمل معاً لتحليل الـ network traffic وكشف الهجمات
        بدقة تصل لـ 99%.

        ### 📊 الداتاسيت المستخدمة في التدريب
        - **TON-IoT** — 22M سجل، 10 أنواع هجمات
        - **CIC-IoT-2023** — 1.5M سجل، 34 نوع هجوم

        ### ⚡ الأداء
        - Accuracy: **97-99%**
        - Inference time: **< 30 ثانية** لمليون سجل
        - يعمل على أي داتاست بدون تعديل يدوي
        """)

    with col2:
        st.markdown("""
        ### 🛠️ التقنيات المستخدمة
        - **FT-Transformer** — نموذج Transformer للـ tabular data
        - **Boruta + SHAP** — اختيار الـ features
        - **SMOTE-ENN** — معالجة الـ imbalance
        - **IsolationForest** — كشف الـ anomalies
        - **Persistent Memory** — تحسين مستمر بين الجلسات

        ### 📞 التواصل
        - GitHub: [Muoz22/ids-network-analyzer](https://github.com/Muoz22/ids-network-analyzer)
        """)

    st.markdown("---")
    st.markdown(
        '<p style="text-align:center; color:#666;">'
        'Built with ❤️ using Streamlit + TensorFlow</p>',
        unsafe_allow_html=True)
