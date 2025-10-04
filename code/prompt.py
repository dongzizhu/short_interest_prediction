"""
Prompt templates for the iterative agent-based feature selection system.

This module contains all the prompt templates used for LLM-based feature engineering.
"""


def create_iterative_prompt_template(iteration_num: int, previous_results: list, 
                                   error_feedback: list = None) -> str:
    """Create a prompt that includes performance history and asks for improvements."""
    # Build performance history
    history_text = ""
    if previous_results:
        history_text = "\n\nPERFORMANCE HISTORY:\n"
        for i, result in enumerate(previous_results):
            history_text += f"Iteration {i}: {result['model_name']} - MAPE: {result['mape']:.2f}%"
            if i > 0:
                improvement = result['improvement']
                history_text += f" (Improvement Over Baseline: {improvement:+.1f}%)"
                history_text += f" (Improvement Over Last: {result.get('improvement_over_last', 0):+.1f}%)"
            history_text += f"\n  Features: {result['features_used']}\n"
            
            # Add DL-based feature importance analysis if available
            if 'feature_stats' in result and result['feature_stats']:
                history_text += f"  DL-Based Feature Importance Analysis:\n"
                
                # Add top 5 most important features
                sorted_features = sorted(result['feature_stats'].items(), 
                                       key=lambda x: x[1]['importance'], reverse=True)
                top_features = sorted_features[:5]
                if top_features:
                    feature_list = ', '.join([f'{name} (importance={stats["importance"]:.4f})' 
                                           for name, stats in top_features])
                    history_text += f"    • Top important features: {feature_list}\n"
            
            history_text += "\n"
    
    # Get the best performance so far
    best_mape = min([r['mape'] for r in previous_results]) if previous_results else 100.0
    
    # Get statistical insights from the best performing model
    best_result = min(previous_results, key=lambda x: x['mape']) if previous_results else None
    statistical_insights = ""
    
#     if best_result and 'feature_stats' in best_result:
#         statistical_insights = f"""
# DL-BASED FEATURE IMPORTANCE INSIGHTS FROM BEST MODEL (MAPE: {best_result['mape']:.2f}%):
# - Most important features: {', '.join([f"{name} (importance={stats['importance']:.4f})" for name, stats in sorted(best_result['feature_stats'].items(), key=lambda x: x[1]['importance'], reverse=True)[:5]])}
# - Least important features: {', '.join([f"{name} (importance={stats['importance']:.4f})" for name, stats in sorted(best_result['feature_stats'].items(), key=lambda x: x[1]['importance'])[:3]])}
# """
    
    # Add error feedback section if there are previous errors
    error_feedback_text = ""
    if error_feedback and len(error_feedback) > 0:
        error_feedback_text = "\n\nERROR FEEDBACK FROM PREVIOUS ATTEMPTS:\n"
        error_feedback_text += "The following errors occurred in previous attempts. Please analyze these errors and ensure your code avoids these issues:\n\n"
        
        for i, error_info in enumerate(error_feedback):
            error_feedback_text += f"Error {i+1}:\n"
            error_feedback_text += f"  • Error Type: {error_info.get('error_type', 'Unknown')}\n"
            error_feedback_text += f"  • Error Message: {error_info.get('error_message', 'No message')}\n"
            if 'sample_info' in error_info:
                error_feedback_text += f"  • Sample Info: {error_info['sample_info']}\n"
            if 'code_snippet' in error_info:
                error_feedback_text += f"  • Problematic Code: {error_info['code_snippet'][:200]}...\n"
            error_feedback_text += "\n"
        
        error_feedback_text += "IMPORTANT: Your new code must avoid these specific errors. Pay special attention to:\n"
        error_feedback_text += "- Array dimension mismatches and shape issues\n"
        error_feedback_text += "- Proper handling of edge cases and NaN values\n"
        error_feedback_text += "- Correct return value format (2D numpy array)\n"
        error_feedback_text += "- Robust error handling within the function\n"
    
    # Add previous code section if there are previous iterations
    previous_code_text = ""
    if previous_results and len(previous_results) > 0:
        # Get the most recent iteration's code
        last_result = previous_results[-1]
        if 'claude_code' in last_result and last_result['claude_code']:
            previous_code_text = "\n\nPREVIOUS ITERATION CODE:\n"
            previous_code_text += f"The following code was used in the most recent iteration (Iteration {last_result['iteration']}):\n\n"
            previous_code_text += "```python\n"
            previous_code_text += last_result['claude_code']
            previous_code_text += "\n```\n\n"
            previous_code_text += f"Performance of this code: MAPE = {last_result['mape']:.2f}%\n"
            if last_result.get('improvement', 0) > 0:
                previous_code_text += f"Improvement over previous: {last_result['improvement']:+.2f}%\n"
            else:
                previous_code_text += f"Change from previous: {last_result['improvement']:+.2f}%\n"
            
            # Add error information if available
            if last_result.get('errors_encountered', 0) > 0:
                previous_code_text += f"Errors encountered: {last_result['errors_encountered']}\n"
                if last_result.get('error_details'):
                    previous_code_text += "Sample errors:\n"
                    for i, error in enumerate(last_result['error_details'][:2]):
                        previous_code_text += f"  - {error.get('error_type', 'Unknown')}: {error.get('error_message', 'No message')}\n"
            
            # Add statistical significance information if available
            if last_result.get('feature_stats'):
                significant_count = len(last_result.get('significant_features', []))
                highly_significant_count = len(last_result.get('highly_significant_features', []))
                total_features = len(last_result['feature_stats'])
                previous_code_text += f"Statistical Analysis: {significant_count}/{total_features} features were significant (p < 0.05), {highly_significant_count} were highly significant (p < 0.01)\n"
            
            previous_code_text += "\nINSTRUCTIONS FOR NEW CODE:\n"
            previous_code_text += "- Analyze the previous code and understand what features it tried to create\n"
            previous_code_text += "- Identify what worked well and what didn't work based on performance and statistical significance\n"
            previous_code_text += "- If the previous code worked but had poor performance, try different feature engineering approaches\n"
            previous_code_text += "- Consider the statistical significance of features - focus on creating features that are likely to be statistically significant\n"
            previous_code_text += "- Your new code should be an improvement over the previous attempt\n"
            previous_code_text += "- Think about what additional financial insights or technical indicators could be valuable\n"
    
    prompt = f"""
You are a financial data scientist specializing in **feature engineering for short-interest prediction** on equity time series. 

## Data schema
- Input to your function: a **numpy array** `data` with shape **(lookback_window, 62)** for a *single* sample.
- Feature layout at each timestep `t`:
  - `data[t, 0]` → **short interest** at time *T* (reported every 15 days)
  - `data[t, 1]` → **average daily volume (past 15 days)**
  - `data[t, 2:62]` → **OHLC** over the past 15 days, flattened as **15 days × 4 columns** in order **[O, H, L, C]**  
    Use: `ohlc = data[t, 2:].reshape(15, 4)` → `open, high, low, close = ohlc[:,0], ohlc[:,1], ohlc[:,2], ohlc[:,3]`.

Total: 1 + 1 + 60 = 62 features(dimensions) per timestamp.

## Dataset constraints
- Only ~180 total samples available (very small).
- To avoid overfitting: generate no more than 80 features. The 80 features should be a mix of raw and constructed features.
- Avoid replacing or discarding the useful raw channels; focus on **augmenting**.

{history_text}

{statistical_insights}

{error_feedback_text}

{previous_code_text}

CURRENT TASK (Iteration {iteration_num}):
Your goal is to create an improved feature engineering function that will achieve better performance than the current best MAPE of {best_mape:.2f}%.

Based on the performance history, DL-based feature importance analysis, and previous code above, analyze what worked and what didn't, then create a new feature engineering approach that:
1. Learns from previous iterations' successes and failures
2. Analyzes the previous code to understand what features were attempted and their effectiveness
4. Considers financial domain knowledge, especially those related to short interest prediction
5. Maintains LSTM-compatible time series structure
6. Uses DL-based feature importance insights to prioritize feature construction
7. Improves upon the previous iteration's approach. If an area is high-importance, refine or extend it; if low-importance, reduce or transform it.

YOUR MAIN GOAL: Construct features that are most likely to be effective for short interest prediction. Do not make redundant features or less effective features. We have limited samples, so a large feature set might lead to overfitting.

Requirements:
1. Write a function called `construct_features` that takes a numpy array of shape (lookback_window, 62) and returns a numpy array of shape (lookback_window, constructed_features)
2. The function should process each timestamp independently but maintain the temporal structure
3. Focus on the most predictive features for each time step, using DL-based feature importance as guidance
4. Consider financial domain knowledge
5. The output should be a 2D numpy array with shape (lookback_window, constructed_features)
6. Include comments explaining your feature engineering choices and how they address previous performance issues and importance insights
7. Make sure the code is production-ready and handles edge cases
8. DO NOT include any import statements - the following libraries are already available in the execution environment:
   - numpy (as 'np')
   - pandas (as 'pd') 
   - math module (as 'math')
   - statistics module (as 'statistics')
   - scipy.stats (as 'stats')
   - sklearn.preprocessing.StandardScaler (as 'StandardScaler')
   - sklearn.preprocessing.MinMaxScaler (as 'MinMaxScaler')
   - sklearn.metrics.mean_squared_error (as 'mean_squared_error')
   - sklearn.metrics.mean_absolute_error (as 'mean_absolute_error')
   - datetime module (as 'datetime')
   - time module (as 'time')
   - All built-in Python functions
9. The function must return a 2D array where each row represents features for one time step
10. Use numpy nan_to_num to handle NaN values
11. Analyze the previous iteration's code and explain in comments how your approach differs and improves upon it
12. Your function should return both the raw features you want to keep and the newly constructed features

Please provide ONLY the Python function code, no explanations outside the code comments.
"""
    
    prompt = f"""
You are a financial data scientist specializing in **feature engineering for short-interest prediction** on equity time series.

## Data schema
- Input to your function: a **numpy array** `data` with shape **(lookback_window, 62)** for a *single* sample.
- Feature layout at each timestep `t`:
  - `data[t, 0]` → **short interest** at time *T* (reported every 15 days)
  - `data[t, 1]` → **average daily volume (past 15 days)**
  - `data[t, 2:62]` → **OHLC** over the past 15 days, flattened as **15 days × 4 columns** in order **[O, H, L, C]**  
    Use: `ohlc = data[t, 2:].reshape(15, 4)` → `open, high, low, close = ohlc[:,0], ohlc[:,1], ohlc[:,2], ohlc[:,3]`.

Total: 1 + 1 + 60 = 62 features per timestamp.

## Dataset constraints
- Only ~180 total samples available (very small).
- To reduce overfitting: **keep only the useful raw channels** and add **new features** so that **(kept raw + new) ≤ 85 total columns**.
- You **may drop** raw channels with consistently low importance or redundancy.
- Avoid redundant or near-duplicate engineered features. Prefer a small, diverse set.

{history_text}

{statistical_insights}

{error_feedback_text}

{previous_code_text}

CURRENT TASK (Iteration {iteration_num}):
Your goal is to create an improved feature engineering function that will achieve better performance than the current best MAPE of {best_mape:.2f}%.

### Strategy
- Learn from previous iterations: refine or extend **high-importance** areas, drop or transform **low-importance** ones.
- Learn from previous iterations: keep the features with high feature importance, drop the features with low feature importance.
- Use **financial domain knowledge**.
- Maintain **LSTM-compatible** time series structure.
- Keep the feature set **compact and non-redundant** due to the small sample size.

### HARD IMPLEMENTATION RULES (must follow to avoid index errors and ensure a stable shape)
- Define constants at the top of the function:
  - `RAW_DIM = 62`
  - `MAX_TOTAL = 80`
  - `MAX_NEW = MAX_TOTAL - 1`  # upper bound; actual new count is determined after raw selection, see below
- **Do NOT preallocate** a fixed-width array and write with a moving `idx`.  
  Instead, for each timestep `t`:
  1) Build two Python lists:
     - `raw_keep = []`  (subset of raw features you choose to keep at t, but your selection logic must be **the same for all timesteps** so the final width is constant)
     - `eng = []`       (engineered features you append one by one)
  2) Always include in `raw_keep`: short interest (index 0) and average volume (index 1).
     Prefer **compact OHLC summaries** over copying all 60 OHLC channels (e.g., last-bar O,H,L,C; mean/median close over last 5; normalized range).
  3) After `raw_keep` is formed, compute `MAX_NEW = MAX_TOTAL - len(raw_keep)`.  
     **Never exceed this cap** when appending to `eng`.
  4) For every engineered candidate, **append to `eng`**.  
     If you hit the cap (`len(eng) == MAX_NEW`), **stop adding** more features (no exceptions).
  5) **Never reference** engineered columns by hard-coded indices (e.g., `features[t, 62+7]` is forbidden).  
     If you need a previously computed engineered value, **reuse the local variable** (e.g., `rsi_val`), not a column number.
  6) Ensure the column count is **identical for all timesteps** (no branch-induced width changes).  
     If a feature cannot be computed (e.g., insufficient points), **append a 0 placeholder** for that slot so widths remain equal.
  7) Construct the row with concatenation:
     - `row = np.array(raw_keep + eng, dtype=np.float32)`
     - If `row.size < MAX_TOTAL`, **pad with zeros** to length `MAX_TOTAL`.
     - If `row.size > MAX_TOTAL`, **truncate the tail** to `MAX_TOTAL`.
- After looping over timesteps, stack rows into a 2D array with shape `(lookback_window, MAX_TOTAL)` and return it.
- The function must **never attempt to write past** column index `MAX_TOTAL - 1`.

### Requirements
1. Write a function called `construct_features` that takes a numpy array of shape (lookback_window, 62) and returns a numpy array of shape (lookback_window, constructed_features).
2. The function must:
   - **Select and preserve only the useful raw features** (you may drop low-importance raw channels).
   - Add **new, diverse features** while enforcing **(kept raw + new) ≤ 80**.
   - Avoid near-duplicates: do not include multiple horizons of the same measure unless clearly distinct.
   - Use **eps clamping** for all divisions: `den = max(abs(den), 1e-8)`.
   - Apply `np.nan_to_num(..., nan=0.0, posinf=0.0, neginf=0.0)` before return.
3. Process each timestep independently but maintain the temporal axis (lookback_window).
4. Focus on the **most predictive and stable** features using DL-based importance + domain knowledge.
5. Include **inline comments** explaining how you improved on previous attempts and why each new feature matters.
6. Code must be **production-ready**: numerically safe, vectorized where reasonable, no randomness or printing.
7. DO NOT include imports — these are already available: `np, pd, math, statistics, stats, StandardScaler, MinMaxScaler, mean_squared_error, mean_absolute_error, datetime, time`.
8. Return a **2D numpy array** `(lookback_window, constructed_features)` with dtype `float32`.

### Strong redundancy rules
- **One per family** unless clearly distinct (e.g., choose either SMA ratio or z-score, not both).
- Drop overlapping or affine equivalents (e.g., SMA ratio vs z-score with same window).
- Avoid fragile ops (`np.corrcoef`, polynomial fits, EMA on <3 points); prefer simple, stable ratios.

### Deliverable
Return **ONLY** the Python function code (no text outside the code).
"""
    return prompt

def create_universal_prompt_template(ticker_results: dict) -> str:
    """Create a prompt to generate universal feature engineering code from multiple ticker results."""
    
    # Build the prompt with all ticker results
    ticker_codes_section = ""
    performance_summary = ""
    
    for ticker, (best_result, iteration_codes) in ticker_results.items():
        ticker_codes_section += f"\n{'='*60}\n"
        ticker_codes_section += f"TICKER: {ticker}\n"
        ticker_codes_section += f"{'='*60}\n"
        ticker_codes_section += f"Best Performance: MAPE = {best_result['mape']:.2f}%\n"
        ticker_codes_section += f"Improvement over baseline: {best_result.get('improvement', 0):+.2f}%\n"
        ticker_codes_section += f"Feature count: {best_result.get('feature_count', 'Unknown')}\n"
        ticker_codes_section += f"Significant features: {len(best_result.get('significant_features', []))}\n"
        
        if best_result.get('claude_code'):
            ticker_codes_section += f"\nBEST FEATURE ENGINEERING CODE FOR {ticker}:\n"
            ticker_codes_section += "-" * 40 + "\n"
            ticker_codes_section += best_result['claude_code']
            ticker_codes_section += "\n"
        
        performance_summary += f"{ticker}: MAPE = {best_result['mape']:.2f}%, Features = {best_result.get('feature_count', 'Unknown')}\n"
    
    prompt = f"""
You are a financial data scientist expert in feature engineering for Short Interest prediction models.

I have run iterative feature engineering processes for multiple tickers and obtained their best-performing feature engineering codes. Now I need you to create a UNIVERSAL feature engineering function that combines the best insights from all tickers.

PERFORMANCE SUMMARY:
{performance_summary}

{ticker_codes_section}

TASK: Create a Universal Feature Engineering Function

Your goal is to analyze all the ticker-specific feature engineering codes above and create a single, universal `construct_features` function that:

1. **Combines the best practices** from all ticker-specific codes
2. **Identifies common patterns** that work well across different stocks
3. **Incorporates the most effective features** from each ticker's best code
4. **Leave the redunant features or less effective features** from each ticker's best code
5. **Maintains the same input/output format**: takes (lookback_window, 62) and returns (lookback_window, constructed_features)

ANALYSIS INSTRUCTIONS:
- Review each ticker's best code and identify the most effective feature engineering techniques
- Look for common patterns across tickers 
- Identify which features consistently perform well across different stocks
- Consider financial domain knowledge that applies universally
- Synthesize the best elements into a cohesive, universal approach

REQUIREMENTS:
1. Write a function called `construct_features` that takes a numpy array of shape (lookback_window, 62) and returns a numpy array of shape (lookback_window, constructed_features)
2. The function should process each timestamp independently but maintain the temporal structure
3. Focus on the most universally applicable features for short interest prediction
4. Include comprehensive comments explaining your feature engineering choices and how they synthesize insights from all tickers
5. The output should be a 2D numpy array with shape (lookback_window, constructed_features)
6. Make sure the code is production-ready and handles edge cases robustly
7. DO NOT include any import statements - the following libraries are already available in the execution environment:
   - numpy (as 'np')
   - pandas (as 'pd') 
   - math module (as 'math')
   - statistics module (as 'statistics')
   - scipy.stats (as 'stats')
   - sklearn.preprocessing.StandardScaler (as 'StandardScaler')
   - sklearn.preprocessing.MinMaxScaler (as 'MinMaxScaler')
   - sklearn.metrics.mean_squared_error (as 'mean_squared_error')
   - sklearn.metrics.mean_absolute_error (as 'mean_absolute_error')
   - datetime module (as 'datetime')
   - time module (as 'time')
   - All built-in Python functions
8. The function must return a 2D array where each row represents features for one time step
9. Use numpy nan_to_num to handle NaN values
10. Create features that are likely to be effective across different stocks and market conditions

Please provide ONLY the Python function code, no explanations outside the code comments.

The function should be a synthesis of the best practices from all the ticker-specific codes above.
"""
    
    prompt = f"""
You are a financial data scientist specializing in **feature engineering for short-interest prediction** on equity time series.

I ran iterative feature engineering for multiple tickers and captured their best-performing codes. Please synthesize a **UNIVERSAL** feature construction function that keeps the strongest, non-redundant ideas **without** inflating feature count.

## Inputs provided
PERFORMANCE SUMMARY:
{performance_summary}

BEST CODES (by ticker):
{ticker_codes_section}

## Data schema (single sample)
- Input: numpy array `data` with shape **(lookback_window, 62)**.
- At each timestep t:
  - `data[t, 0]` → short interest (SI_t) reported every 15 days
  - `data[t, 1]` → average daily volume (past 15 days)
  - `data[t, 2:62]` → OHLC over the past 15 days, flattened as **15 × 4** in order [O,H,L,C]
    Use: `ohlc = data[t, 2:].reshape(15, 4)` then `open_, high, low, close = ohlc[:,0], ohlc[:,1], ohlc[:,2], ohlc[:,3]`.
"""
    
    return prompt
