import shap
import pandas as pd

def explain_binary_flow(
    model,
    X_background,
    X_single,
    feature_names
):
    """
    Robust SHAP explanation for ONE flow.
    Uses interventional perturbation (production-safe).
    """

    explainer = shap.TreeExplainer(
        model,
        data=X_background,
        feature_perturbation="interventional"  # 🔥 FIX
    )

    shap_values = explainer.shap_values(X_single)

    # LightGBM binary handling
    if isinstance(shap_values, list):
        values = shap_values[1]  # positive class
    else:
        values = shap_values

    df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": values[0]
    })

    df["abs"] = df["shap_value"].abs()
    df = df.sort_values("abs", ascending=False)

    return df

def explain_attack_flow(
    model,
    X_background,
    X_single,
    feature_names,
    class_index
):
    """
    SHAP explanation for attack classifier (multiclass).
    Explains ONE flow for ONE predicted attack class.
    """

    explainer = shap.TreeExplainer(
        model,
        data=X_background,
        feature_perturbation="interventional"
    )

    shap_values = explainer.shap_values(X_single)

    # Handle different SHAP value formats for multiclass
    import numpy as np

    try:
        if isinstance(shap_values, list):
            # Old format: list of arrays [class0_values, class1_values, ...]
            if class_index < len(shap_values):
                values = shap_values[class_index][0]
            else:
                raise IndexError(f"class_index {class_index} >= len(shap_values) {len(shap_values)}")
        else:
            # New format: 3D array (n_samples, n_classes, n_features) or 2D
            shap_array = np.array(shap_values)

            if shap_array.ndim == 3:
                # Shape: (n_samples, n_classes, n_features)
                values = shap_array[0, class_index, :]
            elif shap_array.ndim == 2:
                # Shape: (n_samples, n_features) - single output
                # For multiclass with single output, use all values
                values = shap_array[0, :]
            else:
                # Fallback
                values = shap_array.flatten()

    except (IndexError, KeyError) as e:
        # Debug information
        print(f"SHAP format error: {e}")
        print(f"shap_values type: {type(shap_values)}")
        print(f"class_index: {class_index}")

        if isinstance(shap_values, list):
            print(f"shap_values is list, length: {len(shap_values)}")
        else:
            print(f"shap_values shape: {np.array(shap_values).shape}")

        raise ValueError(f"Cannot extract SHAP values for class {class_index}. See console for details.") from e

    df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": values
    })

    df["abs"] = df["shap_value"].abs()
    df = df.sort_values("abs", ascending=False)

    return df
