import os

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import xgboost as xgb

st.set_page_config(page_title="卒中风险评估", layout="wide")

st.title("卒中风险评估")
st.write("请输入以下指标，获取个体化卒中风险评估与解释。")

model_path = "xgb_model.joblib"

st.markdown("**请输入指标（含单位）**")
input_cols = st.columns(2)
with input_cols[0]:
    glu_val = st.number_input("GLU (mmol/L)", value=0.0)
    pulse_pressure_val = st.number_input("脉压 PP (mmHg)", value=0.0)
    gender_label = st.selectbox("性别（男=1，女=2）", ["男", "女"], index=0)
    egfr_val = st.number_input("eGFR (mL/min/1.73m²)", value=0.0)
with input_cols[1]:
    plt_val = st.number_input("PLT (10^9/L)", value=0.0)
    dbp_val = st.number_input("舒张压 DBP (mmHg)", value=0.0)
    mcv_val = st.number_input("MCV (fL)", value=0.0)
    ldl_val = st.number_input("LDL-C (mmol/L)", value=0.0)


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
    st.info("请稍后重试。")
    st.stop()

# 注意：特征名与顺序必须与训练时一致
feature_columns = ["GLU", "脉压", "性别", "eGFR", "PLT", "舒张压", "MCV", "LDL-C"]
manual_values = {
    "GLU": float(glu_val),
    "脉压": float(pulse_pressure_val),
    "性别": 1 if gender_label == "男" else 2,
    "eGFR": float(egfr_val),
    "PLT": float(plt_val),
    "舒张压": float(dbp_val),
    "MCV": float(mcv_val),
    "LDL-C": float(ldl_val),
}
row = pd.DataFrame([manual_values], columns=feature_columns)

predict_clicked = st.button("预测发病概率")
if not predict_clicked:
    st.stop()

# 预测：建议同时给出类别与概率（临床更直观）
try:
    pred_class = int(model.predict(row)[0])
    pred_prob = float(model.predict_proba(row)[0, 1]) if hasattr(model, "predict_proba") else None

    st.subheader("评估结果")
    if pred_prob is not None:
        st.write(f"该个体有 {pred_prob:.2%} 的可能后续发生卒中。")
    else:
        st.write(f"该个体后续发生卒中的风险评估为：{pred_class}。")
except Exception as exc:
    st.error(f"预测失败: {exc}")
    st.stop()

st.subheader("个体解释（力图）")

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
    force_html = f"{shap.getjs()}{force.html()}"
    st.components.v1.html(force_html, height=360)
except Exception as exc:
    st.error(f"力图绘制失败: {exc}")
