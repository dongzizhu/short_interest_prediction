import pandas as pd
import google.generativeai as genai
import os
import pprint


df_sup = pd.read_parquet("/Users/phaniayachitula/Documents/Term 3/short interest term project/Code/df_sup.parquet")


os.environ["GOOGLE_API_KEY"] = "AIzaSyBFMD2zMo3McDf9Y6NyLejLlP3CVbdIJag"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


# Columns: peer short interest + 'target'
feature_cols = [c for c in df_sup.columns if c != "target"]




from HypothesisGenerator import HypothesisGeneratorAgent, FeatureExecutor




# Initial round (no feedback)
generator = HypothesisGeneratorAgent()
features = generator.propose(df_sup, feature_cols, max_features=15)

executor = FeatureExecutor(target_col="target")
df_feats = executor.execute(df_sup, features)

df_feats


df_feats = df_feats.dropna(subset=[executor.target_col] + [f for f in df_feats.columns])


from Evaluator import EvaluatorTool, EvaluatorAgent
feature_names = [c for c in df_feats.columns if c != "target"]
evaluator = EvaluatorTool(target_col="target", train_frac=0.75)

results = evaluator.evaluate(df_feats, feature_names)
pprint.pprint(results)









evaluator_agent = EvaluatorAgent(model="gemini-1.5-flash")

feedback = evaluator_agent.summarize(results, feature_names)

print("\n=== Evaluator Agent Feedback ===")
pprint.pprint(feedback)





# Round 2: refinement using Evaluator Agent feedback
refined_features = generator.propose(df_sup, feature_cols, feedback=feedback, max_features=15)

df_feats_refined = executor.execute(df_sup, refined_features)
df_feats_refined = df_feats_refined.dropna(subset=["target"] + [f for f in df_feats_refined.columns])

results_refined = evaluator.evaluate(df_feats_refined)
pprint.pprint(results_refined)

feedback_refined = evaluator_agent.summarize(results_refined, [c for c in df_feats_refined.columns if c != "target"])
pprint.pprint(feedback_refined)


from Orchestrator import Orchestrator


orchestrator = Orchestrator(
    generator=generator,
    executor=executor,
    evaluator=evaluator,
    evaluator_agent=evaluator_agent,
    peer_cols=feature_cols
)

history = orchestrator.run(df_sup, max_rounds=3, max_features=15)

print("\n=== Final Results ===")
pprint.pprint(history[-1]["results"])