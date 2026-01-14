import os

import joblib
import pandas as pd
import shap
import streamlit as st


st.set_page_config(page_title="临床预测与个体解释", layout="wide")

st.title("临床应用：预测与个体可解释性")
st.write(
    "使用本地模型文件并手动输入变量，生成预测与 SHAP 解释（含力图）。"
)

with st.sidebar:
    st.header("数据与模型")
    st.markdown("**使用本地模型文件**")
    model_path = st.text_input("模型路径", value="xgb_model.joblib")
    st.markdown("**手动输入变量**")
    input_cols = st.columns(2)
    with input_cols[0]:
        glu_val = st.number_input("GLU", value=0.0)
        pulse_pressure_val = st.number_input("脉压", value=0.0)
        gender_label = st.selectbox("性别（男=1，女=0）", ["男", "女"], index=0)
        egfr_val = st.number_input("eGFR", value=0.0)
    with input_cols[1]:
        plt_val = st.number_input("PLT", value=0.0)
        dbp_val = st.number_input("舒张压", value=0.0)
        mcv_val = st.number_input("MCV", value=0.0)
        ldl_val = st.number_input("LDL-C", value=0.0)


def load_model(path):
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型文件不存在: {path}")
    return joblib.load(path)


model = None
try:
    model = load_model(model_path)
except Exception as exc:
    st.error(f"模型加载失败: {exc}")

if model is None:
    st.info("请确认模型路径。")
    st.stop()

st.subheader("个体输入")

feature_columns = ["GLU", "脉压", "性别", "eGFR", "PLT", "舒张压", "MCV", "LDL-C"]
manual_values = {
    "GLU": glu_val,
    "脉压": pulse_pressure_val,
    "性别": 1 if gender_label == "男" else 0,
    "eGFR": egfr_val,
    "PLT": plt_val,
    "舒张压": dbp_val,
    "MCV": mcv_val,
    "LDL-C": ldl_val,
}
row = pd.DataFrame([manual_values], columns=feature_columns)
st.dataframe(row, use_container_width=True)

try:
    prediction = model.predict(row)[0]
    st.subheader("预测结果")
    st.write(f"预测值: {prediction}")
except Exception as exc:
    st.error(f"预测失败: {exc}")
    st.stop()

st.subheader("个体 SHAP 解释")

background = row.copy()

try:
    explainer = shap.Explainer(model, background)
    shap_values = explainer(row)
except Exception as exc:
    st.error(f"SHAP 计算失败: {exc}")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.markdown("**SHAP Waterfall 图**")
    try:
        fig = shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig, clear_figure=True)
    except Exception as exc:
        st.error(f"Waterfall 图绘制失败: {exc}")

with col2:
    st.markdown("**SHAP 力图**")
    try:
        force = shap.force_plot(
            explainer.expected_value, shap_values.values[0], row, matplotlib=False
        )
        st.components.v1.html(force.html(), height=360)
    except Exception as exc:
        st.error(f"力图绘制失败: {exc}")
