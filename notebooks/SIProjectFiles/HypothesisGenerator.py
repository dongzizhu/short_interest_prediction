
import re




from dataclasses import dataclass
import json
import numpy as np
import pandas as pd
import google.generativeai as genai

@dataclass
class HypothesisGeneratorAgent:
    model: str = "gemini-1.5-flash"

    def _call_model(self, prompt: str):
        """
        Internal method to call the configured LLM and return raw text.
        Currently supports Gemini. Extendable for other providers.
        """
        if self.model.startswith("gemini"):
            # Gemini example
            model = genai.GenerativeModel(
                self.model,
                generation_config={"response_mime_type": "application/json"}
            )
            response = model.generate_content(prompt)
            return response.text

        # Placeholder for future providers (OpenAI, Anthropic, etc.)
        raise NotImplementedError(f"Model '{self.model}' not supported yet.")

    def propose(self, df: pd.DataFrame, target_col: str, col_descriptions: dict,
                feedback: dict = None, max_features: int = 15):
        """
        Propose engineered features to help predict the target variable.

        Args:
            df (pd.DataFrame): Input dataframe with features + target.
            target_col (str): Column name of the target variable to predict.
            col_descriptions (dict): Dictionary mapping column names to descriptions.
            feedback (dict, optional): Refinement feedback from previous iteration.
            max_features (int, optional): Max number of features to propose.

        Returns:
            list[dict]: Proposed features, each with 'name' and 'transformation'.
        """

        # Identify feature columns (all columns except the target)
        feature_cols = [c for c in df.columns if c != target_col]

        # Quick safety check
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe.")
        if len(feature_cols) == 0:
            raise ValueError("No feature columns found — dataframe must contain at least one feature besides the target.")
        
        # Take a small preview (last 5 rows) for context
        preview = df[feature_cols + [target_col]].tail(5).reset_index()
        preview["index"] = preview["index"].astype(str)

        # Make sure preview is JSON-serializable (replace NaN, convert numerics to float)
        preview = preview.replace({np.nan: None}).astype(object).applymap(
            lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x
        )

        # Build schema
        schema = {
            "target_col": target_col,
            "feature_cols": feature_cols,
            "col_descriptions": col_descriptions,
            "rows": int(df.shape[0]),
            "example_tail_rows": preview.to_dict(orient="records"),
        }

        # Turn schema into JSON string
        schema_json = json.dumps(schema, indent=2)

        # Build inline descriptions string
        desc_lines = [f"- {col}: {desc}" for col, desc in col_descriptions.items()]
        descriptions_str = "\n".join(desc_lines)

        # Base prompt
        base_prompt = f"""
        You are a quantitative research assistant.
        Your task is to propose engineered features to help predict '{target_col}'.

        The target column represents: {col_descriptions.get(target_col, "No description provided")}.

        Feature columns available for use:
        {descriptions_str}

        DATA SCHEMA (for reference only):
        {schema_json}
        
        RULES:
        - Use only the feature columns provided.
        - Only propose features that can be computed with the following transformations:
        1. Lags (by 1, 2, 5, 10 periods)
        2. Rolling averages (5, 10, 20 windows)
        3. Rolling standard deviations (5, 10, 20 windows)
        4. First differences
        5. Percent changes
        6. Cross-sectional ranks among features
        7. Cross-sectional z-scores among features
        8. Interactions (elementwise product of 2 features)
        9. Averages across a specified list of features
        - Do not invent any transformation outside this list.
        - Propose at most {max_features} features.
        - Return STRICT JSON list only.
        - Each entry must contain exactly these keys:
        - "name": proposed feature name
        - "transformation": short text describing how the feature is derived, use the text provided above to describe the transformations



        Example output format:
        [
        {{"name": "ABBV_lag1", "transformation": "lag of ABBV_lag0 by 1"}},
        {{"name": "MRK_roll10_std", "transformation": "10-period rolling std of MRK_lag0"}},
        {{"name": "rank_JNJ", "transformation": "cross-sectional rank of JNJ_lag0 among features"}}
        ]
        """

        if feedback:
            base_prompt += f"\nREFINEMENT FEEDBACK:\n{json.dumps(feedback, indent=2)}\n"

        #Call model
        raw_output = self._call_model(base_prompt)

        #Parse JSON safely
        try:
            proposals = json.loads(raw_output)
            if isinstance(proposals, list) and all(
                isinstance(p, dict) and "name" in p and "transformation" in p
                for p in proposals
            ):
                return proposals
            else:
                print(" Output did not match expected format.")
                return []
        except Exception as e:
            print(" Could not parse JSON. Error:", e)
            print("Raw output:\n", raw_output)
            return []
        









from typing import List
import re

@dataclass
class ExecutorAgent:
    """
    Deterministic feature executor for the 9 supported transformation families.
    """


    def execute(self, df: pd.DataFrame, proposals: List[dict], feature_cols: List[str]) -> pd.DataFrame:
        # map lower -> actual column name (preserve exact df columns)
        df = df.copy()
        name_map = {c.lower(): c for c in df.columns}
        feature_cols_canon = [self._canon(c, name_map) for c in feature_cols]

        created_cols = []  # track successfully created features

        for prop in proposals:
            name = prop.get("name")
            trans = prop.get("transformation", "")
            try:
                series = self._dispatch(df, trans, feature_cols_canon, name_map)
                df[name] = series
                created_cols.append(name)
            except Exception as e:
                print(f" Skipping '{name}' ({trans}): {e}")

        # Always trim rows that have NaNs in the newly created features
        if created_cols:
            mask = df[created_cols].notna().all(axis=1)
            dropped = int((~mask).sum())
            df = df.loc[mask].copy()
            print(f"Trimmed {dropped} rows due to insufficient history in new features.")

        return df



    # ---------- router ----------
    def _dispatch(self, df: pd.DataFrame, transformation: str,
                  feature_cols: List[str], name_map: dict) -> pd.Series:
        tl = transformation.lower().strip()  # for routing only

        if "lag" in tl and "by" in tl:
            return self._lag(df, transformation, name_map)
        if "rolling" in tl and ("average" in tl or "mean" in tl):
            return self._rolling_mean(df, transformation, name_map)
        if "rolling" in tl and ("std" in tl or "standard deviation" in tl):
            return self._rolling_std(df, transformation, name_map)
        if "first difference" in tl:
            return self._first_diff(df, transformation, name_map)
        if "percent change" in tl:
            return self._pct_change(df, transformation, name_map)
        if "cross-sectional rank" in tl and "among" in tl:
            return self._xsec_rank(df, transformation, feature_cols, name_map)
        if "cross-sectional z-score" in tl and "among" in tl:
            return self._xsec_zscore(df, transformation, feature_cols, name_map)
        if "elementwise product" in tl and " and " in tl:
            return self._interaction(df, transformation, name_map)
        if ("average of" in tl) or ("mean of" in tl):
            return self._average_list(df, transformation, name_map)

        raise NotImplementedError(f"Unsupported transformation: {transformation}")

    # ---------- small helpers ----------
    @staticmethod
    def _first_int(text: str) -> int:
        m = re.search(r"\b(\d+)\b", text)
        if not m:
            raise ValueError(f"Could not find an integer in: '{text}'")
        return int(m.group(1))

    @staticmethod
    def _col_after_of(text: str) -> str:
        m = re.search(r"of\s+(.+?)(?:\s+by\b|\s+among\b|\s+and\b|$)", text, flags=re.IGNORECASE)
        if not m:
            raise ValueError(f"Could not parse column after 'of' in: '{text}'")
        return m.group(1).strip().strip(",")

    @staticmethod
    def _two_cols_after_of_and(text: str) -> tuple[str, str]:
        m = re.search(r"of\s+(.+?)\s+and\s+(.+)$", text, flags=re.IGNORECASE)
        if not m:
            raise ValueError(f"Could not parse 'of <A> and <B>' in: '{text}'")
        return m.group(1).strip().strip(","), m.group(2).strip().strip(",")

    @staticmethod
    def _list_after_of(text: str) -> List[str]:
        m = re.search(r"(?:average|mean)\s+of\s+(.+)$", text, flags=re.IGNORECASE)
        if not m:
            raise ValueError(f"Could not parse list after 'average/mean of' in: '{text}'")
        return [c.strip() for c in m.group(1).split(",") if c.strip()]

    @staticmethod
    def _canon(col: str, name_map: dict) -> str:
        """Map any-cased column name to the exact df column name."""
        key = col.strip().lower()
        if key not in name_map:
            raise KeyError(f"Column '{col}' not in dataframe.")
        return name_map[key]

    @staticmethod
    def _ensure(df: pd.DataFrame, cols: List[str]) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Columns not in dataframe: {missing}")

    # ---------- handlers ----------
    def _lag(self, df: pd.DataFrame, text: str, name_map: dict) -> pd.Series:
        lag = self._first_int(text)
        col = self._canon(self._col_after_of(text), name_map)
        self._ensure(df, [col])
        return df[col].shift(lag)

    def _rolling_mean(self, df: pd.DataFrame, text: str, name_map: dict) -> pd.Series:
        win = self._first_int(text)
        col = self._canon(self._col_after_of(text), name_map)
        self._ensure(df, [col])
        return df[col].rolling(win, min_periods=1).mean()

    def _rolling_std(self, df: pd.DataFrame, text: str, name_map: dict) -> pd.Series:
        win = self._first_int(text)
        col = self._canon(self._col_after_of(text), name_map)
        self._ensure(df, [col])
        return df[col].rolling(win, min_periods=2).std()

    def _first_diff(self, df: pd.DataFrame, text: str, name_map: dict) -> pd.Series:
        col = self._canon(self._col_after_of(text), name_map)
        self._ensure(df, [col])
        return df[col].diff()

    def _pct_change(self, df: pd.DataFrame, text: str, name_map: dict) -> pd.Series:
        col = self._canon(self._col_after_of(text), name_map)
        self._ensure(df, [col])
        return df[col].pct_change()

    def _xsec_rank(self, df: pd.DataFrame, text: str,
                   feature_cols: List[str], name_map: dict) -> pd.Series:
        col = self._canon(self._col_after_of(text), name_map)
        self._ensure(df, [col])
        ranks = df[feature_cols].rank(axis=1, method="average", na_option="keep")
        return ranks[col]

    def _xsec_zscore(self, df: pd.DataFrame, text: str,
                     feature_cols: List[str], name_map: dict) -> pd.Series:
        col = self._canon(self._col_after_of(text), name_map)
        self._ensure(df, [col])
        mu = df[feature_cols].mean(axis=1)
        sig = df[feature_cols].std(axis=1).replace({0.0: np.nan})
        return (df[col] - mu) / sig

    def _interaction(self, df: pd.DataFrame, text: str, name_map: dict) -> pd.Series:
        c1_raw, c2_raw = self._two_cols_after_of_and(text)
        c1, c2 = self._canon(c1_raw, name_map), self._canon(c2_raw, name_map)
        self._ensure(df, [c1, c2])
        return df[c1] * df[c2]

    def _average_list(self, df: pd.DataFrame, text: str, name_map: dict) -> pd.Series:
        cols_raw = self._list_after_of(text)
        cols = [self._canon(c, name_map) for c in cols_raw]
        self._ensure(df, cols)
        return df[cols].mean(axis=1)
