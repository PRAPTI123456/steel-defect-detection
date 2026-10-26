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
from inspector import inspect_image, load_model, to_grayscale_bgr

st.set_page_config(page_title="Defect Inspection", page_icon="🏭", layout="wide")
db.init_db()

st.title("🏭 Industrial Defect Detection & Quality Inspection")

with st.sidebar:
    st.header("Settings")
    weights = st.text_input("Model weights path", "runs/neu_det/weights/best.pt")
    conf = st.slider("YOLO confidence threshold", 0.01, 0.95, 0.10, 0.01)
    st.caption("Lower threshold = more detections (more false alarms).")
    imgsz = st.selectbox("Image size (match training)", [200, 320, 416, 640], index=3)
    gray = st.checkbox("Convert to grayscale before testing", value=True)
    strict = st.checkbox("Strict mode: any defect = REJECT", value=True)
    severity.STRICT_MODE = strict

tab_inspect, tab_dash = st.tabs(["🔍 Inspect", "📊 Dashboard"])

VERDICT_ICON = {"PASS": "✅", "REVIEW": "⚠️", "REJECT": "❌"}

with tab_inspect:
    files = st.file_uploader("Upload product images", type=["jpg", "jpeg", "png", "bmp"],
                             accept_multiple_files=True)
    if files:
        model = load_model(weights)
        if model is None:
            st.error(f"No YOLO weights found at '{weights}'. Check the path in the sidebar.")
            st.stop()

        for f in files:
            img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
            annotated, defects, verdict = inspect_image(model, img, conf, imgsz=imgsz, grayscale=gray)
            db.save_inspection(f.name, verdict, defects)

            c1, c2 = st.columns(2)
            c1.image(img[:, :, ::-1], caption=f"Original: {f.name}")
            c2.image(annotated[:, :, ::-1], caption="Detected defects (model input was grayscale)"
                     if gray else "Detected defects")
            st.subheader(f"{VERDICT_ICON[verdict]} Verdict: {verdict}")
            if defects:
                st.dataframe(pd.DataFrame(defects)[["class_name", "source", "confidence", "area_pct", "severity"]])
            else:
                st.info("No defects detected at this confidence.")
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
