from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np


def one_split(X, y, train_frac=0.8):
    """Single train/test split: first part = train, last part = test."""
    n_train = int(len(X) * train_frac)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def evaluate_lasso(X, y, feature_names, alpha=0.01):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", Lasso(alpha=alpha, max_iter=5000, random_state=42))
    ])
    Xtr, ytr, Xte, yte = one_split(X, y)
    model.fit(Xtr, ytr)
    ypred = model.predict(Xte)

    perf = {"R2": r2_score(yte, ypred),
            "MAPE": mean_absolute_percentage_error(yte, ypred)}

    coefs = np.abs(model.named_steps["lasso"].coef_)
    if coefs.sum() > 0:
        coefs = coefs / coefs.sum()
    importance = dict(zip(feature_names, coefs))

    return {"performance": perf, "feature_importance": importance}


def evaluate_ridge(X, y, feature_names, alpha=1.0):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha, random_state=42))
    ])
    Xtr, ytr, Xte, yte = one_split(X, y)
    model.fit(Xtr, ytr)
    ypred = model.predict(Xte)

    perf = {"R2": r2_score(yte, ypred),
            "MAPE": mean_absolute_percentage_error(yte, ypred)}

    coefs = np.abs(model.named_steps["ridge"].coef_)
    if coefs.sum() > 0:
        coefs = coefs / coefs.sum()
    importance = dict(zip(feature_names, coefs))

    return {"performance": perf, "feature_importance": importance}


def evaluate_rf(X, y, feature_names, n_estimators=200):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    Xtr, ytr, Xte, yte = one_split(X, y)
    model.fit(Xtr, ytr)
    ypred = model.predict(Xte)

    perf = {"R2": r2_score(yte, ypred),
            "MAPE": mean_absolute_percentage_error(yte, ypred)}

    importance = dict(zip(feature_names, model.feature_importances_))

    return {"performance": perf, "feature_importance": importance}


from xgboost import XGBRegressor

def evaluate_xgb(X, y, feature_names,
                 n_estimators=500, max_depth=5, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8):
    Xtr, ytr, Xte, yte = one_split(X, y)

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42,
        n_jobs=-1
    )
    model.fit(Xtr, ytr)
    ypred = model.predict(Xte)

    perf = {"R2": r2_score(yte, ypred),
            "MAPE": mean_absolute_percentage_error(yte, ypred)}

    # feature importances directly from the booster
    importances = model.feature_importances_
    importance = dict(zip(feature_names, importances))

    return {"performance": perf, "feature_importance": importance}



from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error

def evaluate_svr(X, y, feature_names, train_frac=0.8, C=1.0, epsilon=0.1, kernel="rbf"):
    """
    Evaluate Support Vector Regression (SVR) on a single train/test split.

    Args:
        X, y: features and target arrays
        feature_names: list of column names
        train_frac: fraction of data to use for training
        C: regularization parameter
        epsilon: epsilon-tube for regression
        kernel: 'rbf', 'linear', 'poly', etc.

    Returns:
        dict with performance metrics and (dummy) feature importance
    """
    n_train = int(len(X) * train_frac)
    Xtr, ytr = X[:n_train], y[:n_train]
    Xte, yte = X[n_train:], y[n_train:]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(C=C, epsilon=epsilon, kernel=kernel))
    ])
    model.fit(Xtr, ytr)
    ypred = model.predict(Xte)

    perf = {
        "R2": r2_score(yte, ypred),
        "MAPE": mean_absolute_percentage_error(yte, ypred)
    }

    # SVR (with RBF/poly kernels) doesn’t have native feature importances.
    # If you want them, you can use permutation importance later.
    importance = {f: 0.0 for f in feature_names}

    return {"performance": perf, "feature_importance": importance}




def evaluate_all(df, target_col):
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    feature_names = df.drop(columns=[target_col]).columns

    return {
        "Lasso": evaluate_lasso(X, y, feature_names),
        "Ridge": evaluate_ridge(X, y, feature_names),
        "SVR": evaluate_svr(X, y, feature_names),
        #"XGBoost": evaluate_xgb(X, y, feature_names),
        #"RandomForest": evaluate_rf(X, y, feature_names),
    }




import json
import google.generativeai as genai  # or swap out for OpenAI, Anthropic, etc.

class FeedbackAgent:
    def __init__(self, model="gemini-1.5-flash"):
        self.model = model

    def _call_model(self, prompt: str) -> str:
        if self.model.startswith("gemini"):
            model = genai.GenerativeModel(
                self.model,
                generation_config={"response_mime_type": "application/json"}
            )
            response = model.generate_content(prompt)
            return response.text
        else:
            raise NotImplementedError(f"Model '{self.model}' not supported yet.")

    def build_feedback(self, evaluator_results: dict, target_col: str, max_features_next_round: int = 15) -> dict:
        """
        Convert evaluator results into structured feedback via an LLM.
        """
        prompt = f"""
You are a feature feedback agent. We are predicting '{target_col}'.

Here are model evaluation results (feature importances are already normalized):
{json.dumps(evaluator_results, indent=2, default=str)}

Your task:
- Read the metrics (R², MAPE) and feature importances from each model.
- Identify which features should be KEPT (strong signal), which should be DROPPED (weak/noisy).
- Suggest transformation families that worked well and should be emphasized.
- Suggest which families to avoid.
- Summarize everything in STRICT JSON with the following keys:

{{
  "keep_features": [...],
  "drop_features": [...],
  "suggested_transforms": {{
    "emphasize": ["lag", "roll_std", ...],
    "deemphasize": ["rank", "interaction", ...]
  }},
  "notes": "any short free-text guidance"
}}

Limit new proposals in the next round to {max_features_next_round} features.
        """.strip()

        raw_output = self._call_model(prompt)

        try:
            feedback = json.loads(raw_output)
            return feedback
        except Exception as e:
            print("Could not parse feedback JSON:", e)
            print("Raw output:\n", raw_output)
            return {}
