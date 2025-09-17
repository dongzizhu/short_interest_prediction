from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import json
import google.generativeai as genai






@dataclass
class EvaluatorTool:
    target_col: str = "target"
    train_frac: float = 0.75
    rf_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 400,
        "random_state": 42,
        "n_jobs": -1,
        "min_samples_leaf": 2,
    })
    ridge_alpha: float = 1.0
    lasso_alpha: float = 0.001

    def chrono_split(self, X: pd.DataFrame, y: pd.Series):
        cutoff = int(len(X) * self.train_frac)
        return (X.iloc[:cutoff], X.iloc[cutoff:], y.iloc[:cutoff], y.iloc[cutoff:])

    def _eval_metrics(self, model, X_train, y_train, X_test, y_test):
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # avoid division by zero in MAPE
        mape = np.mean(
            np.abs((y_test - y_pred) / np.where(y_test == 0, 1e-8, y_test))
        ) * 100

        return {
            "rmse": np.sqrt(mse),
            "mae": mae,
            "r2": r2,
            "mape": mape,
            "dir_acc": np.mean(
                np.sign(y_pred - np.median(y_train)) ==
                np.sign(y_test - np.median(y_train))
            )
        }

    def evaluate(self, df_feats: pd.DataFrame, feature_names: List[str] = None) -> Dict[str, Any]:
        if feature_names is None:
            feature_names = [c for c in df_feats.columns if c != self.target_col]

        df_use = df_feats.dropna(subset=[self.target_col] + feature_names).copy()
        X = df_use[feature_names]
        y = df_use[self.target_col]

        X_train, X_test, y_train, y_test = self.chrono_split(X, y)
        results = {}

        # --- Linear Regression ---
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        results["linreg"] = {**self._eval_metrics(lr, X_train, y_train, X_test, y_test),
                             "n_features": len(feature_names)}

        # --- Ridge Regression ---
        ridge = Ridge(alpha=self.ridge_alpha)
        ridge.fit(X_train, y_train)
        results["ridge"] = {**self._eval_metrics(ridge, X_train, y_train, X_test, y_test),
                            "n_features": len(feature_names)}

        # --- Lasso Regression ---
        lasso = Lasso(alpha=self.lasso_alpha, max_iter=10000)
        lasso.fit(X_train, y_train)
        results["lasso"] = {**self._eval_metrics(lasso, X_train, y_train, X_test, y_test),
                            "n_features": len(feature_names),
                            "nonzero_coefs": int(np.sum(lasso.coef_ != 0))}

        # --- Random Forest ---
        rf = RandomForestRegressor(**self.rf_params)
        rf.fit(X_train, y_train)
        results["rf"] = {**self._eval_metrics(rf, X_train, y_train, X_test, y_test),
                         "n_features": len(feature_names),
                         "feature_importances": sorted(
                             list(zip(feature_names, rf.feature_importances_)),
                             key=lambda x: -x[1]
                         )[:10]}  # top 10 features

        return results



@dataclass
class EvaluatorAgent:
    model: str = "gemini-1.5-flash"

    def build_prompt(self, results: dict, feature_names: list[str]) -> str:
        return f"""
You are an evaluator agent in a feature-engineering loop for forecasting short interest.

Here are the evaluation results:

{json.dumps(results, indent=2, default=str)}

FEATURES TESTED:
{feature_names}

TASK:
- Identify which model(s) performed best and why.
- Highlight weak models and possible reasons.
- Point out strong feature groups (e.g., short lags, rolling means).
- Suggest which feature families (ratios, z-scores, etc.) may be hurting.
- Provide concise feedback for the Hypothesis Generator Agent to refine the feature set.

OUTPUT FORMAT:
Respond in JSON with the following fields:
{{
  "summary": "...short natural language summary...",
  "best_model": "name",
  "weak_models": ["list"],
  "suggest_keep": ["feature types to keep"],
  "suggest_drop": ["feature types to drop"],
  "next_steps": "...guidance for refining features..."
}}
"""

    def summarize(self, results: dict, feature_names: list[str]) -> dict:
        prompt = self.build_prompt(results, feature_names)
        model = genai.GenerativeModel(self.model)
        response = model.generate_content(prompt)

        try:
            return json.loads(response.text)
        except Exception:
            # fallback: return raw text if not JSON
            return {"summary_raw": response.text.strip()}
        
        