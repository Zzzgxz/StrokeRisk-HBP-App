import os

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

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

st.subheader("个体 SHAP 解释")

# 对 XGBoost 明确使用 TreeExplainer，避免 shap.Explainer 自动推断失败
try:
    explainer = shap.TreeExplainer(model)

    # 二分类：shap_values 常见为 (n_samples, n_features)；部分版本可能返回 list
    shap_values = explainer.shap_values(row)

    expected_value = explainer.expected_value
    # 兼容：expected_value 可能是数组（例如 [base0, base1]），取正类
    if isinstance(expected_value, (list, np.ndarray)) and np.ndim(expected_value) > 0:
        expected_value_use = float(np.array(expected_value).ravel()[-1])
    else:
        expected_value_use = float(expected_value)

    # 兼容：shap_values 可能是 list（如 [class0, class1]），取正类
    if isinstance(shap_values, list):
        shap_values_use = shap_values[-1]
    else:
        shap_values_use = shap_values

    # 构造 Explanation 以便使用新版 shap.plots.waterfall
    exp = shap.Explanation(
        values=shap_values_use[0],
        base_values=expected_value_use,
        data=row.iloc[0].values,
        feature_names=row.columns.tolist(),
    )
except Exception as exc:
    st.error(f"SHAP 计算失败: {exc}")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.markdown("**SHAP Waterfall 图**")
    try:
        fig = shap.plots.waterfall(exp, show=False)
        st.pyplot(fig, clear_figure=True)
    except Exception as exc:
        st.error(f"Waterfall 图绘制失败: {exc}")

with col2:
    st.markdown("**SHAP 力图**")
    try:
        force = shap.force_plot(
            expected_value_use,
            shap_values_use[0],
            row.iloc[0],
            feature_names=row.columns.tolist(),
            matplotlib=False,
        )
        st.components.v1.html(force.html(), height=360)
    except Exception as exc:
        st.error(f"力图绘制失败: {exc}")
