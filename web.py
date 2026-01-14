import os

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import xgboost as xgb

st.set_page_config(page_title="临床预测与个体解释", layout="wide")

st.title("临床应用：预测与个体可解释性")
st.write("使用本地模型文件并手动输入变量，生成预测与 SHAP 解释（含力图）。")

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


def load_model(path: str):
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型文件不存在: {path}")
    mdl = joblib.load(path)

    # xgboost 版本兼容：某些序列化模型在新版本环境中缺少该字段
    if mdl.__class__.__name__ == "XGBClassifier" and not hasattr(mdl, "use_label_encoder"):
        mdl.use_label_encoder = False

    return mdl


model = None
try:
    model = load_model(model_path)
except Exception as exc:
    st.error(f"模型加载失败: {exc}")

if model is None:
    st.info("请确认模型路径。")
    st.stop()

st.subheader("个体输入")

# 注意：特征名与顺序必须与训练时一致
feature_columns = ["GLU", "脉压", "性别", "eGFR", "PLT", "舒张压", "MCV", "LDL-C"]
manual_values = {
    "GLU": float(glu_val),
    "脉压": float(pulse_pressure_val),
    "性别": 1 if gender_label == "男" else 0,
    "eGFR": float(egfr_val),
    "PLT": float(plt_val),
    "舒张压": float(dbp_val),
    "MCV": float(mcv_val),
    "LDL-C": float(ldl_val),
}
row = pd.DataFrame([manual_values], columns=feature_columns)
st.dataframe(row, use_container_width=True)

# 预测：建议同时给出类别与概率（临床更直观）
try:
    pred_class = int(model.predict(row)[0])
    pred_prob = float(model.predict_proba(row)[0, 1]) if hasattr(model, "predict_proba") else None

    st.subheader("预测结果")
    st.write(f"预测类别(0/1): {pred_class}")
    if pred_prob is not None:
        st.write(f"预测概率P(卒中=1): {pred_prob:.4f}")
except Exception as exc:
    st.error(f"预测失败: {exc}")
    st.stop()

st.subheader("个体 SHAP 解释（力图）")

# 使用 XGBoost 原生 pred_contribs 计算 SHAP，避免编码问题
try:
    feature_names = row.columns.tolist()
    row_np = row.to_numpy(dtype=float)

    # 显式覆盖特征名，规避编码异常
    try:
        model.get_booster().feature_names = feature_names
    except Exception:
        pass

    dmatrix = xgb.DMatrix(row_np, feature_names=feature_names)
    contribs = model.get_booster().predict(dmatrix, pred_contribs=True)
    shap_values_use = contribs[:, :-1]
    expected_value_use = float(contribs[0, -1])

    # 构造 Explanation 以便使用新版 shap.plots.waterfall
    exp = shap.Explanation(
        values=shap_values_use[0],
        base_values=expected_value_use,
        data=row_np[0],
        feature_names=feature_names,
    )
except Exception as exc:
    st.error(f"SHAP 计算失败: {exc}")
    st.stop()

st.markdown("**SHAP 力图**")
try:
    force = shap.force_plot(
        expected_value_use,
        shap_values_use[0],
        row_np[0],
        feature_names=feature_names,
        matplotlib=False,
    )
    st.components.v1.html(force.html(), height=360)
except Exception as exc:
    st.error(f"力图绘制失败: {exc}")
