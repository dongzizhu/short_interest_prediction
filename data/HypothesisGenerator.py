from dataclasses import dataclass
import json
import pandas as pd
import numpy as np
import re
import google.generativeai as genai

@dataclass
class HypothesisGeneratorAgent:
    model: str = "gemini-1.5-flash"

    def propose(self, df, peer_cols, feedback=None, max_features=15):
        # Build preview
        preview = df[peer_cols + ["target"]].tail(5).reset_index()
        preview["index"] = preview["index"].astype(str)
        preview = preview.replace({np.nan: None}).astype(object).applymap(
            lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x
        )

        schema = {
            "target_col": "target",
            "peer_cols": peer_cols,
            "rows": int(df.shape[0]),
            "example_tail_rows": preview.to_dict(orient="records"),
        }

        base_prompt = f"""
You are a quantitative research assistant.
Your task is to propose engineered features to help predict 'target' (future short interest).

DATA SCHEMA:
{json.dumps(schema, indent=2)}

RULES:
- Use only the peer columns provided.
- Prefer features like:
  - lags (1, 2, 5, 10 periods)
  - rolling averages/volatilities (5, 10, 20 windows)
  - first differences
  - cross-sectional ranks/z-scores across peers
  - interactions between a few strong features
- Propose at most {max_features} features.
- Return STRICT JSON list only, e.g.:
[
  {{"name": "ABBV_lag1", "transformation": "lag of ABBV_lag0 by 1"}},
  {{"name": "MRK_roll10_std", "transformation": "10-day rolling std of MRK_lag0"}},
  {{"name": "peer_rank_JNJ", "transformation": "cross-sectional rank of JNJ_lag0 among peers"}}
]
"""

        if feedback:
            base_prompt += f"\nREFINEMENT FEEDBACK:\n{json.dumps(feedback, indent=2)}\n"

        # Call Gemini
        model = genai.GenerativeModel(
                "gemini-1.5-flash",
                generation_config={"response_mime_type": "application/json"}
                )
        response = model.generate_content(base_prompt)

        try:
            return json.loads(response.text)
        except Exception:
            print("⚠️ Could not parse JSON. Raw output:\n", response.text)
            return []
        



class FeatureExecutor:
    def __init__(self, target_col="target"):
        self.target_col = target_col

    def _resolve_col(self, col_str: str, df: pd.DataFrame) -> str:
        for c in df.columns:
            if c.lower() == col_str.lower():
                return c
        raise KeyError(f"Column {col_str} not found in dataframe")

    def execute(self, df: pd.DataFrame, features: list[dict]) -> pd.DataFrame:
        df = df.copy()
        peer_cols = [c for c in df.columns if c != self.target_col]

        for f in features:
            name = f["name"]
            desc = f["transformation"].lower().strip()

            try:
                # --- Lag ---
                if "lag" in desc and "period" in desc:
                    match = re.search(r"lag of (\w+) by (\d+)", desc)
                    if match:
                        col = self._resolve_col(match.group(1), df)
                        k = int(match.group(2))
                        df[name] = df[col].shift(k)
                        continue

                # --- Rolling mean/std ---
                if "rolling mean" in desc:
                    match = re.search(r"(\d+)-day rolling mean of (\w+)", desc)
                    if match:
                        w = int(match.group(1))
                        col = self._resolve_col(match.group(2), df)
                        df[name] = df[col].rolling(w, min_periods=1).mean()
                        continue

                if "rolling volatility" in desc or "rolling std" in desc:
                    match = re.search(r"(\d+)-day rolling (?:volatility|std) of (\w+)", desc)
                    if match:
                        w = int(match.group(1))
                        col = self._resolve_col(match.group(2), df)
                        df[name] = df[col].rolling(w, min_periods=1).std()
                        continue

                # --- First difference ---
                if "first difference" in desc:
                    match = re.search(r"first difference of (\w+)", desc)
                    if match:
                        col = self._resolve_col(match.group(1), df)
                        df[name] = df[col].diff(1)
                        continue

                # --- Cross-sectional rank ---
                if "cross-sectional rank" in desc:
                    match = re.search(r"cross-sectional rank of (\w+)", desc)
                    if match:
                        col = self._resolve_col(match.group(1), df)
                        df[name] = df[peer_cols].rank(axis=1, pct=True)[col]
                        continue

                # --- Cross-sectional z-score ---
                if "cross-sectional z-score" in desc:
                    match = re.search(r"cross-sectional z-score of (\w+)", desc)
                    if match:
                        col = self._resolve_col(match.group(1), df)
                        mu = df[peer_cols].mean(axis=1)
                        sigma = df[peer_cols].std(axis=1).replace(0, np.nan)
                        df[name] = (df[col] - mu) / sigma
                        continue

                # --- Ratio ---
                if "ratio of" in desc:
                    match = re.search(r"ratio of (\w+) to (\w+)", desc)
                    if match:
                        col1 = self._resolve_col(match.group(1), df)
                        col2 = self._resolve_col(match.group(2), df)
                        df[name] = df[col1] / df[col2]
                        continue

                # --- Interaction ---
                if "interaction" in desc:
                    match = re.search(r"interaction (?:of|between) (\w+) and (\w+)", desc)
                    if match:
                        col1 = self._resolve_col(match.group(1), df)
                        col2 = self._resolve_col(match.group(2), df)
                        df[name] = df[col1] * df[col2]
                        continue

                print(f"⚠️ Skipped unrecognized feature: {f}")

            except Exception as e:
                print(f"⚠️ Failed to compute feature {f}: {e}")

        return df