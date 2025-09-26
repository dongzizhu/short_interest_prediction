import pandas as pd
from HypothesisGenerator import HypothesisGeneratorAgent, ExecutorAgent
from SimpleEvaluator import evaluate_all, FeedbackAgent

class Orchestrator:
    def __init__(self,
                 target_col: str,
                 col_descriptions: dict,
                 max_features: int = 15,
                 max_iterations: int = 5,
                 stability_tol: int = 2):
        """
        Orchestrator that coordinates generator → executor → evaluator → feedback.
        """
        self.target_col = target_col
        self.col_descriptions = col_descriptions
        self.max_features = max_features
        self.max_iterations = max_iterations
        self.stability_tol = stability_tol

        # agents inside orchestrator
        self.generator = HypothesisGeneratorAgent()
        self.executor = ExecutorAgent()
        self.feedback_agent = FeedbackAgent()

    def run(self, main_df: pd.DataFrame):
        """
        Run the iterative loop until stability or max_iterations.
        main_df is always kept raw/unchanged.
        """
        feedback = None
        prev_keep = set()
        stable_counter = 0
        history = []

        for i in range(1, self.max_iterations + 1):
            print(f"\n🔄 Iteration {i}")

            # 1. Hypothesis generation
            proposals = self.generator.propose(
                df=main_df,
                target_col=self.target_col,
                col_descriptions=self.col_descriptions,
                feedback=feedback,
                max_features=self.max_features
            )
            print(f"  Proposed {len(proposals)} features.")

            # 2. Execute on a fresh copy of main_df
            df_sup = self.executor.execute(
                main_df.copy(),
                proposals,
                feature_cols=[c for c in main_df.columns if c != self.target_col]
            )
            print(f"  DataFrame now has {df_sup.shape[1]} columns.")

            # 3. Evaluate
            results = evaluate_all(df_sup, target_col=self.target_col)
            metrics = {m: r["performance"] for m, r in results.items()}
            print("  Metrics:", metrics)

            # 4. Feedback
            feedback = self.feedback_agent.build_feedback(results, target_col=self.target_col)
            keep_feats = feedback.get("keep_features", [])
            print(f"  Feedback keep_features: {keep_feats}")

            # log
            history.append({
                "iteration": i,
                "metrics": metrics,
                "keep_features": keep_feats,
                "drop_features": feedback.get("drop_features", [])
            })

            # 5. Stability check
            keep_now = set(keep_feats)
            if keep_now == prev_keep:
                stable_counter += 1
            else:
                stable_counter = 0
            prev_keep = keep_now

            if stable_counter >= self.stability_tol:
                print(f"\n✅ Features stabilized after {i} iterations.")
                break

        return df_sup, feedback, results, history
