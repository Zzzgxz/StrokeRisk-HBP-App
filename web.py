import os

import joblib
import pandas as pd
import shap
import streamlit as st


st.set_page_config(page_title="临床预测与个体解释", layout="wide")

st.title("临床应用：预测与个体可解释性")
st.write(
    "使用本地模型文件并上传结构化临床数据，选择个体后生成预测与 SHAP 解释（含力图）。"
)

with st.sidebar:
    st.header("数据与模型")
    st.markdown("**使用本地模型文件**")
    model_path = st.text_input("模型路径", value="xgb_model.joblib")
    data_file = st.file_uploader("上传数据文件（CSV）", type=["csv"])
    target_col = st.text_input("目标列名（用于排除）", value="")
    row_index = st.number_input("个体行号（从 0 开始）", min_value=0, value=0, step=1)


def load_model(path):
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型文件不存在: {path}")
    return joblib.load(path)


def load_data(uploaded_file):
    if uploaded_file is None:
        return None
    return pd.read_csv(uploaded_file)


model = None
try:
    model = load_model(model_path)
except Exception as exc:
    st.error(f"模型加载失败: {exc}")

data = load_data(data_file)

if model is None or data is None:
    st.info("请确认模型路径并上传数据。")
    st.stop()

if target_col and target_col in data.columns:
    feature_data = data.drop(columns=[target_col])
else:
    feature_data = data.copy()

if len(feature_data) == 0:
    st.error("数据为空，无法进行预测。")
    st.stop()

if row_index >= len(feature_data):
    st.warning("行号超出范围，已自动调整为最后一行。")
    row_index = len(feature_data) - 1

row = feature_data.iloc[[row_index]]

st.subheader("个体输入")
st.dataframe(row, use_container_width=True)

try:
    prediction = model.predict(row)[0]
    st.subheader("预测结果")
    st.write(f"预测值: {prediction}")
except Exception as exc:
    st.error(f"预测失败: {exc}")
    st.stop()

st.subheader("个体 SHAP 解释")

background = feature_data.sample(
    n=min(100, len(feature_data)), random_state=0, replace=False
)

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
        st.components.v1.html(force.html(), height=320)
    except Exception:
        try:
            force = shap.force_plot(
                explainer.expected_value,
                shap_values.values[0],
                row,
                matplotlib=True,
                show=False,
            )
            st.pyplot(force, clear_figure=True)
        except Exception as exc:
            st.error(f"力图绘制失败: {exc}")
