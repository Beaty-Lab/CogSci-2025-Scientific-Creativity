import os
import csv
import time
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from tqdm import tqdm
from anthropic import Anthropic
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from openai import OpenAI
from itertools import combinations
# TODO: You must create a keys.py and add your API keys to run this script
# alternatively, set the keys as environment variables and update lines 25-27
from keys import ANTHROPIC_API_KEY, OPENAI_API_KEY

# Model Configuration
CLAUDE_MODEL = "claude-3-5-haiku-20241022"
GPT4_MODEL = "gpt-4o-mini"

# Initialize API clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# DPT Scoring Configuration
RANDOM_SEED = 99  # Random seed for reproducibility
SCORING_SCALE = (1, 5)  # Min and max scores

CONDITION = "oracle"

# Rate Limiting and Retry Settings
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 2
MAX_RETRY_DELAY = 30

# Output File Configuration
timestamp = time.strftime("%Y%m%d_%H%M%S")
script_dir = os.path.dirname(os.path.abspath(__file__))
output_csv = os.path.join(script_dir, f"dpt_scores_{CONDITION}_{timestamp}.csv")
correlation_plot = os.path.join(
    script_dir, f"correlation_plot_{CONDITION}_{timestamp}.png"
)
agreement_plot = os.path.join(
    script_dir, f"model_agreement_plot_{CONDITION}_{timestamp}.png"
)
oracle_examples = pd.read_csv(
    "../Code/scientific-creativity/data_analysis/oracle_examples.csv"
)


def load_and_process_data(file_path: str, condition: str) -> pd.DataFrame:
    """
    Load DPT factor scores from a CSV file, filter by condition, and rescale originality factor.

    Args:
        file_path (str): The path to the CSV file containing DPT factor scores.
        condition (str): The condition to filter the data by.

    Returns:
        pd.DataFrame: A DataFrame containing the filtered and rescaled data.
    """
    df = pd.read_csv(file_path, index_col=0)
    filtered_df = df[df["condition"] == condition].copy()

    # Rescale originality_factor to 1-5 range
    min_factor = df["originality_rescaled_factor"].min()
    max_factor = df["originality_rescaled_factor"].max()
    filtered_df["scaled_score"] = (
        (filtered_df["originality_rescaled_factor"] - min_factor)
        / (max_factor - min_factor)
        * (SCORING_SCALE[1] - SCORING_SCALE[0])
        + SCORING_SCALE[0]
    ).round()

    print(f"\nData Statistics:")
    print("\nOriginality Score Distribution:")
    print(filtered_df["scaled_score"].value_counts().sort_index())
    print(len(filtered_df))

    return filtered_df


def select_few_shot_examples(row: pd.Series, condition: str) -> List[Tuple[str, float]]:
    """
    Select few-shot examples based on the given row and condition.

    Args:
        row (pd.Series): A row of data from the DataFrame.
        condition (str): The condition to filter the examples by.

    Returns:
        List[Tuple[str, float]]: A list of tuples containing the example text and its score.
    """
    if condition == "no_oracle":
        return []

    elif condition == "oracle":
        # given a problem id in the row, need to loc on the oracle exemplars to find the few-shot examples

        examples = oracle_examples.loc[oracle_examples["problem"] == row["problem"]]
        return list(zip(examples["response"], examples["originality_score"]))
    else:
        print("Not a valid condition!")
        exit(-1)


def get_scoring_prompt(problem: str) -> str:
    """
    Generates a system prompt for evaluating the originality of engineering solutions.

    Creates a detailed prompt that instructs an AI system how to evaluate solutions
    on multiple criteria including originality, uncommonness, remoteness and cleverness
    using 1-5 rating scales.

    Args:
        problem (str): The engineering design problem to be evaluated.

    Returns:
        str: A formatted prompt containing rating criteria and scales.
    """
    return f"""You are evaluating the originality of solutions to this engineering design problem: {problem}

You will rate solutions on multiple criteria using 1-5 scales and provide explanations. Your response must follow this exact format:

ORIGINALITY: [1-5 or NA]
UNCOMMON: [1-5]
REMOTE: [1-5]
CLEVER: [1-5]
EXPLANATION: [1-2 sentences explaining the originality rating from a STEM perspective]

Rating scales:
Originality:
1: Very Unoriginal
2: Unoriginal
3: Neutral
4: Original
5: Very Original
NA: For solutions that are too short or unclear

Uncommon:
1: Very common
2: Common
3: Neutral
4: Uncommon
5: Very uncommon

Remote:
1: Very unremote
2: Unremote
3: Neutral
4: Remote
5: Very remote

Clever:
1: Very unclever
2: Unclever
3: Neutral
4: Clever
5: Very clever

Evaluation criteria:
- Uncommon: Consider if the solution is rare. Common solutions given by many people score low. A unique solution may score high, unless it's just strange rather than original.
- Remote: Consider how far the solution is from everyday ideas. Non-obvious solutions score higher. Solutions similar to common approaches score lower.
- Clever: Consider the insight and wit shown. Clear, insightful solutions score higher. Unclear solutions score lower. Even common solutions can score high if presented cleverly.

Provide your STEM-based explanation focusing solely on the originality rating. Be specific about why the solution is or isn't original from an engineering perspective."""


def parse_model_response(response_text: str) -> dict:
    """
    Parses the LLM's response to extract originality ratings and explanation.

    Takes a model's response string containing ratings for originality, uncommonness,
    remoteness, cleverness and an explanation. Extracts and validates these values
    according to the expected format.

    Args:
        response (str): Raw response string from the language model containing
            ratings and explanation in the specified format.

    Returns:
        Dict[str, Union[int, str]]: Dictionary containing parsed values with keys:
            - 'originality': Integer rating 1-5 or 'NA'
            - 'uncommon': Integer rating 1-5
            - 'remote': Integer rating 1-5
            - 'clever': Integer rating 1-5
            - 'explanation': String containing the explanation text

    Raises:
        Error: If response cannot be parsed or contains invalid ratings.
    """
    try:
        lines = response_text.strip().split("\n")
        result = {}

        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()

                if key in ["originality", "uncommon", "remote", "clever"]:
                    if value.upper() == "NA":
                        result[key] = None
                    else:
                        result[key] = int(value)
                elif key == "explanation":
                    result[key] = value

        return result
    except Exception as e:
        print(f"Error parsing model response: {str(e)}")
        return None


def get_claude_score(
    claude_client: Anthropic,
    response: str,
    problem: str,
    few_shot_examples: List[Tuple[str, float]],
    max_retries: int = MAX_RETRIES,
) -> Optional[dict]:
    """
    Gets scores for a DPT solution using the Claude API.

    Makes an API call to Claude to evaluate the originality of a solution to an
    engineering design problem. Implements exponential backoff retry logic for
    rate limiting. Extracts scores for originality, uncommonness, remoteness,
    and cleverness along with an explanation.

    Args:
        problem (str): The engineering design problem being solved.
        response (str): The solution text to evaluate.

    Returns:
        Dict[str, Union[int, str]]: Dictionary containing:
            - 'originality': Integer rating 1-5 or 'NA'
            - 'uncommon': Integer rating 1-5
            - 'remote': Integer rating 1-5
            - 'clever': Integer rating 1-5
            - 'explanation': String containing the explanation text

    Raises:
        Exception: If API call fails after maximum retries or returns invalid response.
    """
    messages = []

    for example_response, example_rating in few_shot_examples:
        example_output = f"""ORIGINALITY: {int(example_rating)}
        UNCOMMON: {int(example_rating)}
        REMOTE: {int(example_rating)}
        CLEVER: {int(example_rating)}
        EXPLANATION: Example response with rating {int(example_rating)}."""

        messages.extend(
            [
                {"role": "user", "content": f"Rate this solution:\n{example_response}"},
                {"role": "assistant", "content": example_output},
            ]
        )

    messages.append({"role": "user", "content": f"Rate this solution:\n{response}"})

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = min(INITIAL_RETRY_DELAY * (2 ** (attempt - 1)), MAX_RETRY_DELAY)
                time.sleep(delay)

            response = claude_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=300,  # Increased for explanation
                temperature=0,
                system=get_scoring_prompt(problem),
                messages=messages,
            )

            result = parse_model_response(response.content[0].text)
            if result:
                return result

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error getting Claude score: {str(e)}")
                continue
            else:
                print(f"Final Claude error after {max_retries} attempts: {str(e)}")
                return None

    return None


def get_gpt4_score(
    response: str,
    problem: str,
    few_shot_examples: List[Tuple[str, float]],
    max_retries: int = MAX_RETRIES,
) -> Optional[dict]:
    """
    Gets originality scores for an engineering solution using the GPT-4 API.

    Makes an API call to GPT-4 to evaluate the originality of a solution to an
    engineering design problem. Implements exponential backoff retry logic for
    rate limiting. Extracts scores for originality, uncommonness, remoteness,
    and cleverness along with an explanation.

    Args:
        problem (str): The engineering design problem being solved.
        response (str): The solution text to evaluate.

    Returns:
        Dict[str, Union[int, str]]: Dictionary containing:
            - 'originality': Integer rating 1-5 or 'NA'
            - 'uncommon': Integer rating 1-5
            - 'remote': Integer rating 1-5
            - 'clever': Integer rating 1-5
            - 'explanation': String containing the explanation text

    Raises:
        Exception: If API call fails after maximum retries or returns invalid response.
    """
    messages = [{"role": "system", "content": get_scoring_prompt(problem)}]

    for example_response, example_rating in few_shot_examples:
        example_output = f"""ORIGINALITY: {int(example_rating)}
        UNCOMMON: {int(example_rating)}
        REMOTE: {int(example_rating)}
        CLEVER: {int(example_rating)}
        EXPLANATION: Example response with rating {int(example_rating)}."""

        messages.append(
            {"role": "user", "content": f"Rate this solution:\n{example_response}"}
        )
        messages.append({"role": "assistant", "content": example_output})

    messages.append({"role": "user", "content": f"Rate this solution:\n{response}"})

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = min(INITIAL_RETRY_DELAY * (2 ** (attempt - 1)), MAX_RETRY_DELAY)
                time.sleep(delay)

            response = openai_client.chat.completions.create(
                model=GPT4_MODEL,
                messages=messages,
                response_format={"type": "text"},
                max_tokens=300,
                temperature=0,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )

            result = parse_model_response(response.choices[0].message.content)
            if result:
                return result

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error getting GPT-4 score: {str(e)}")
                continue
            else:
                print(f"Final GPT-4 error after {max_retries} attempts: {str(e)}")
                return None

    return None



def compute_correlations_and_plot(df: pd.DataFrame):
    """
    Computes correlations between model ratings and creates visualization plots.

    Analyzes correlations between different model ratings (Claude, GPT4) and original scores
    across multiple criteria (Originality, Uncommon, Remote, Clever). Generates two types
    of visualizations:
    1. Correlation heatmaps showing relationships between all raters
    2. Pairwise agreement plots with regression lines for each pair of raters

    Args:
        df (pd.DataFrame): DataFrame containing rating columns for each model and criterion.
            Expected columns format: {ModelName}_{CriterionName} (e.g. 'Claude_Originality')
            Must include 'Original_Score' column.

    Returns:
        None: Saves visualization files to disk:
            - Correlation heatmaps: correlation_plot_{criterion}.png
            - Agreement plots: model_agreement_plot_{criterion}.png
    """
    criteria = ["Originality", "Uncommon", "Remote", "Clever"]
    models = ["Claude", "GPT4o"]

    for criterion in criteria:
        rating_cols = [f"{model}_{criterion}" for model in models] + ["Original_Score"]
        corr_matrix = df[rating_cols].corr()

        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="viridis", vmin=-1, vmax=1)
        plt.title(f"{criterion} Rating Correlations Between Models and Original Scores")
        plt.tight_layout()
        plt.savefig(
            correlation_plot.replace(".png", f"_{criterion.lower()}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Create pairwise agreement plots
        # Calculate number of combinations for subplot layout
        num_combinations = len(list(combinations(rating_cols, 2)))
        num_rows = (num_combinations + 2) // 3  # Ceiling division to get number of rows
        plt.figure(figsize=(15, 5 * num_rows))

        for i, (col1, col2) in enumerate(combinations(rating_cols, 2)):
            plt.subplot(num_rows, 3, i + 1)
            sns.regplot(data=df, x=col1, y=col2, scatter_kws={"alpha": 0.5})
            plt.title(f'{col1.split("_")[0]} vs {col2.split("_")[0]}')
        plt.tight_layout()
        plt.savefig(
            agreement_plot.replace(".png", f"_{criterion.lower()}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Print correlation statistics
        print(f"\nCorrelation Statistics for {criterion}:")
        for col1, col2 in combinations(rating_cols, 2):
            r, p = stats.pearsonr(df[col1], df[col2])
            print(
                f"{col1.split('_')[0]} vs {col2.split('_')[0]}: r = {r:.3f}, p = {p:.4f}"
            )

        # Print model agreement statistics
        model_cols = [f"{model}_{criterion}" for model in models]
        print(f"\nModel Agreement Statistics for {criterion}:")
        for col1, col2 in combinations(model_cols, 2):
            agreement = (df[col1] == df[col2]).mean() * 100
            print(
                f"{col1.split('_')[0]} vs {col2.split('_')[0]}: {agreement:.1f}% exact agreement"
            )


def main():
    print("Starting DPT multi-model scoring script...")

    # Initialize Anthropic client
    claude_client = Anthropic(api_key=ANTHROPIC_API_KEY)

    # Load and process data
    print("Loading and processing DPT data...")
    df = load_and_process_data(
        "../Code/scientific-creativity/data_analysis/cleaned_data_explanations_gold.csv",
        CONDITION,
    )

    all_scores = []

    # Write CSV headers
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Response",
                "Claude_Originality",
                "Claude_Uncommon",
                "Claude_Remote",
                "Claude_Clever",
                "Claude_Explanation",
                "GPT4o_Originality",
                "GPT4o_Uncommon",
                "GPT4o_Remote",
                "GPT4o_Clever",
                "GPT4o_Explanation"
                "Average_Originality",
                "Average_Uncommon",
                "Average_Remote",
                "Average_Clever",
                "Original_Score",
            ]
        )

    # Score responses
    with tqdm(total=len(df), desc="Scoring responses") as pbar:
        for _, row in df.iterrows():
            response = row["response"]
            original_score = row["scaled_score"]
            problem_statement = row["problem"]
            few_shot_examples = select_few_shot_examples(row, CONDITION)

            try:
                # Get scores from all models
                claude_results = get_claude_score(
                    claude_client, response, problem_statement, few_shot_examples
                )

                gpt4_results = get_gpt4_score(
                    response, problem_statement, few_shot_examples
                )

                # Calculate average scores if all ratings were successful
                if all(
                    results is not None for results in [claude_results, gpt4_results]
                ):
                    # Calculate averages for each criterion
                    avg_originality = round(
                        sum(
                            r["originality"]
                            for r in [claude_results, gpt4_results]
                            if r["originality"] is not None
                        )
                        / 3,
                        2,
                    )
                    avg_uncommon = round(
                        sum(
                            r["uncommon"]
                            for r in [claude_results, gpt4_results]
                            if r["uncommon"] is not None
                        )
                        / 3,
                        2,
                    )
                    avg_remote = round(
                        sum(
                            r["remote"]
                            for r in [claude_results, gpt4_results]
                            if r["remote"] is not None
                        )
                        / 3,
                        2,
                    )
                    avg_clever = round(
                        sum(
                            r["clever"]
                            for r in [claude_results, gpt4_results]
                            if r["clever"] is not None
                        )
                        / 3,
                        2,
                    )

                    # Save to CSV
                    with open(output_csv, "a", newline="", encoding="utf-8") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(
                            [
                                response,
                                claude_results["originality"],
                                claude_results["uncommon"],
                                claude_results["remote"],
                                claude_results["clever"],
                                claude_results["explanation"],
                                gpt4_results["originality"],
                                gpt4_results["uncommon"],
                                gpt4_results["remote"],
                                gpt4_results["clever"],
                                gpt4_results["explanation"],
                                avg_originality,
                                avg_uncommon,
                                avg_remote,
                                avg_clever,
                                original_score,
                            ]
                        )

                    all_scores.append(
                        {
                            "Response": response,
                            "Claude_Originality": claude_results["originality"],
                            "Claude_Uncommon": claude_results["uncommon"],
                            "Claude_Remote": claude_results["remote"],
                            "Claude_Clever": claude_results["clever"],
                            "GPT4o_Originality": gpt4_results["originality"],
                            "GPT4o_Uncommon": gpt4_results["uncommon"],
                            "GPT4o_Remote": gpt4_results["remote"],
                            "GPT4o_Clever": gpt4_results["clever"],
                            "Average_Originality": avg_originality,
                            "Average_Uncommon": avg_uncommon,
                            "Average_Remote": avg_remote,
                            "Average_Clever": avg_clever,
                            "Original_Score": original_score,
                        }
                    )

            except Exception as e:
                print(f"\nError processing response: {str(e)}")

            pbar.update(1)

    # Create correlation plots if we have enough data
    if len(all_scores) > 1:
        results_df = pd.DataFrame(all_scores)
        compute_correlations_and_plot(results_df)

        # Print summary statistics
        print("\nSummary Statistics:")
        print("\nAverage Ratings by Model:")
        for col in ["Claude_Originality", "GPT4o_Originality"]:
            print(f"\n{col}:")
            print(results_df[col].describe())

        print("\nCorrelation with Original Scores:")
        for col in ["Claude_Originality", "GPT4o_Originality"]:
            r, p = stats.pearsonr(results_df[col], results_df["Original_Score"])
            print(f"{col}: r = {r:.3f}, p = {p:.4f}")
    else:
        print("\nNot enough data for correlation analysis")

    print(f"\nResults have been saved to:")
    print(f"Scores and responses: {output_csv}")
    print(f"Correlation plot: {correlation_plot}")
    print(f"Agreement plot: {agreement_plot}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        import traceback

        print("\nFull error traceback:")
        print(traceback.format_exc())
