from dataclasses import dataclass
from HypothesisGenerator import HypothesisGeneratorAgent, FeatureExecutor
from Evaluator import EvaluatorTool, EvaluatorAgent

@dataclass
class Orchestrator:
    generator: HypothesisGeneratorAgent
    executor: FeatureExecutor
    evaluator: EvaluatorTool
    evaluator_agent: EvaluatorAgent
    peer_cols: list
    target_col: str = "target"

    def run(self, df, max_rounds=3, max_features=15):
        history = []
        features_current = self.generator.propose(df, self.peer_cols, max_features=max_features)

        for round_num in range(max_rounds):
            print(f"\n=== Round {round_num+1} ===")

            # --- Feature execution
            df_feats = self.executor.execute(df, features_current)
            df_feats = df_feats.dropna(subset=[self.target_col] + [f for f in df_feats.columns])

            # --- Evaluation
            results = self.evaluator.evaluate(df_feats)

            # --- Feedback
            feature_names = [c for c in df_feats.columns if c != self.target_col]
            feedback = self.evaluator_agent.summarize(results, feature_names)

            history.append({
                "round": round_num+1,
                "features": features_current,
                "results": results,
                "feedback": feedback
            })

            print("Summary:", feedback.get("summary", feedback))

            # --- Refinement
            features_current = self.generator.propose(df, self.peer_cols,
                                                     feedback=feedback,
                                                     max_features=max_features)

        return history
