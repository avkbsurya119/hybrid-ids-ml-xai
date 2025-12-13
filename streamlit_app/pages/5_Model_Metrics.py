import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="Model Metrics", layout="wide")
st.title("📊 Model Metrics & Evaluation")

# -------------------------------------------------
# Safety checks
# -------------------------------------------------
if "uploaded_df" not in st.session_state or "inference_df" not in st.session_state:
    st.warning("Please upload a dataset and run inference first.")
    st.stop()

df = st.session_state["uploaded_df"]
pred = st.session_state["inference_df"]

# Detect label column
label_col = None
if "Label" in df.columns:
    label_col = "Label"
elif " Label" in df.columns:
    label_col = " Label"

if label_col is None:
    st.error("No ground-truth labels found in dataset. Metrics unavailable.")
    st.stop()

y_true = df[label_col].values
y_pred_final = pred["Final_Prediction"].values
y_pred_binary = pred["Binary_Route"].values

# -------------------------------------------------
# Section 1: Binary Model Metrics
# -------------------------------------------------
st.header("1️⃣ Binary Classifier Metrics (BENIGN vs ATTACK)")

y_true_binary = np.where(y_true == "BENIGN", "BENIGN", "ATTACK")

cm_bin = confusion_matrix(y_true_binary, y_pred_binary, labels=["BENIGN", "ATTACK"])

# ---- Confusion Matrix ----
fig, ax = plt.subplots()
im = ax.imshow(cm_bin, cmap="Blues")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["BENIGN", "ATTACK"])
ax.set_yticklabels(["BENIGN", "ATTACK"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

for i in range(2):
    for j in range(2):
        ax.text(j, i, cm_bin[i, j], ha="center", va="center")

st.subheader("Confusion Matrix (Binary)")
st.pyplot(fig)

# ---- Classification report ----
st.subheader("Classification Report")
st.code(
    classification_report(
        y_true_binary,
        y_pred_binary,
        digits=4
    )
)

# -------------------------------------------------
# Section 2: ROC Curve
# -------------------------------------------------
st.header("2️⃣ ROC Curve (Binary Model)")

# Convert labels
y_true_roc = np.where(y_true_binary == "ATTACK", 1, 0)

# Approx confidence proxy
binary_conf = np.where(
    pred["Binary_Route"] == "ATTACK",
    1 - pred["Confidence"],
    pred["Confidence"]
)

fpr, tpr, _ = roc_curve(y_true_roc, binary_conf)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
ax.plot([0, 1], [0, 1], linestyle="--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()

st.pyplot(fig)

# -------------------------------------------------
# Section 3: Attack Classifier Metrics
# -------------------------------------------------
st.header("3️⃣ Attack Classifier Metrics")

attack_mask = y_true != "BENIGN"

if attack_mask.sum() == 0:
    st.warning("No attack samples available for attack-class metrics.")
    st.stop()

y_true_attack = y_true[attack_mask]
y_pred_attack = y_pred_final[attack_mask]

# ---- Confusion Matrix ----
labels = sorted(np.unique(y_true_attack))
cm_atk = confusion_matrix(y_true_attack, y_pred_attack, labels=labels)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm_atk, cmap="Oranges")

ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_yticklabels(labels)

for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, cm_atk[i, j], ha="center", va="center", fontsize=8)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.subheader("Attack Confusion Matrix")
st.pyplot(fig)

# ---- Classification report ----
st.subheader("Attack Classification Report")
st.code(
    classification_report(
        y_true_attack,
        y_pred_attack,
        digits=4
    )
)

# -------------------------------------------------
# Section 4: Class Distribution
# -------------------------------------------------
st.header("4️⃣ Class Distribution")

dist_df = pd.DataFrame({
    "Actual": pd.Series(y_true_attack).value_counts(),
    "Predicted": pd.Series(y_pred_attack).value_counts()
}).fillna(0)

st.bar_chart(dist_df)
