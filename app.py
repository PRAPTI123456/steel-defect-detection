"""
Streamlit dashboard.
Run:  streamlit run app.py
"""
import cv2
import numpy as np
import pandas as pd
import streamlit as st

import db
import severity
from inspector import inspect_image, load_model

st.set_page_config(page_title="Defect Inspection", page_icon="🏭", layout="wide")
db.init_db()

st.title("🏭 Industrial Defect Detection & Quality Inspection")

with st.sidebar:
    st.header("Settings")
    weights = st.text_input("Model weights path (optional)", "runs/neu_det/weights/best.pt")
    conf = st.slider("YOLO confidence threshold", 0.05, 0.95, 0.25, 0.05)
    st.caption("Lower threshold = more detections (more false alarms).")
    use_classic = st.checkbox("Also use classical scratch detector", value=True,
                              help="Works on any steel photo without training. Shown in red.")
    classic_z = st.slider("Classical sensitivity", 5.0, 15.0, 9.0, 0.5,
                          help="Lower = more sensitive (finds fainter scratches, more false alarms).")
    strict = st.checkbox("Strict mode: any defect = REJECT", value=True)
    severity.STRICT_MODE = strict
    st.caption("Green boxes = YOLO model, red boxes = classical detector.")

tab_inspect, tab_dash = st.tabs(["🔍 Inspect", "📊 Dashboard"])

VERDICT_ICON = {"PASS": "✅", "REVIEW": "⚠️", "REJECT": "❌"}

with tab_inspect:
    files = st.file_uploader("Upload product images", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=True)
    if files:
        model = load_model(weights)
        if model is None:
            st.warning(f"No YOLO weights found at '{weights}'. Using the classical detector only. "
                       "Train the model (train.py) and put best.pt there for full detection.")
            if not use_classic:
                st.error("Enable the classical detector or provide model weights.")
                st.stop()

        for f in files:
            img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
            annotated, defects, verdict = inspect_image(model, img, conf, use_classic=use_classic, classic_z=classic_z)
            db.save_inspection(f.name, verdict, defects)

            c1, c2 = st.columns(2)
            c1.image(img[:, :, ::-1], caption=f"Original: {f.name}")
            c2.image(annotated[:, :, ::-1], caption="Detected defects")
            st.subheader(f"{VERDICT_ICON[verdict]} Verdict: {verdict}")
            if defects:
                st.dataframe(pd.DataFrame(defects)[["class_name", "source", "confidence", "area_pct", "severity"]])
            st.divider()

with tab_dash:
    inspections = db.fetch_inspections()
    defects = db.fetch_defects()
    if not inspections:
        st.info("No inspections yet. Upload images in the Inspect tab.")
    else:
        df_i = pd.DataFrame(inspections)
        df_d = pd.DataFrame(defects)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Products inspected", len(df_i))
        m2.metric("Passed", int((df_i.verdict == "PASS").sum()))
        m3.metric("Need review", int((df_i.verdict == "REVIEW").sum()))
        m4.metric("Rejected", int((df_i.verdict == "REJECT").sum()))

        c1, c2 = st.columns(2)
        if not df_d.empty:
            c1.subheader("Defects by type")
            c1.bar_chart(df_d.class_name.value_counts())
            c2.subheader("Defects by severity")
            c2.bar_chart(df_d.severity.value_counts())

        st.subheader("Inspection history")
        st.dataframe(df_i)
