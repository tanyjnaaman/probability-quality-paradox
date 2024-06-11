from typing_extensions import Literal
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns  # type: ignore
import matplotlib.colors as mcolors
import pathlib
import matplotlib


from tabulate import tabulate  # type: ignore
from numba import prange  # type: ignore
from scipy.stats import pearsonr, spearmanr, chi2_contingency  # type: ignore
from pydantic_argparse import ArgumentParser
from pydantic import BaseModel, Field
from typing import List
from tqdm import tqdm

sns.set_theme(font="serif")


class ScriptArguments(BaseModel):
    input_data_dir: str = Field(
        title="Input Data Directory",
        description="The directory containing the input data",
    )
    cache_dir: str = Field(
        title="Output Directory",
        description=(
            "The directory to save the generated figures and intermediate outputs"
        ),
    )
    overwrite_cache: bool = Field(
        False,
        title="Overwrite Cache",
        description="Whether to overwrite the cache",
    )
    use_secret_reward: bool = Field(
        False,
        title="Use Secret Reward",
        description="Whether to use the secret reward function",
    )
    figure_name: str = Field(
        title="Figure Name",
        description="The name of the figure to save",
    )


class SamplingTypeMetadata(BaseModel):
    # metadata
    filename: str
    sampling_type: Literal[
        "top_p095",
        "top_p090",
        "top_k50",
        "top_k30",
        "ancestral_strict",
        "ancestral",
        "typical_p090",
        "eta_n00009",
    ]
    temperature: float

    # intermediate output filename
    mean_corpuses_df_cache_filename: str

    # style
    color: str
    pretty_name: str
    alpha: float


def main():
    # 0. Parse arguments
    parser = ArgumentParser(model=ScriptArguments)
    args = parser.parse_typed_args()

    # 1. Load the data
    data_files = os.listdir(args.input_data_dir)
    reward_df_files = sorted(
        [f for f in data_files if "scoredreward_humanassistant_includeprompt" in f],
        key=lambda x: x.replace("_ancestral_", "_top_k50_") if "strict" not in x else x,
    )
    nll_df_files = sorted(
        [f for f in data_files if "scorednll" in f and "includeprompt" not in f],
        key=lambda x: x.replace("_ancestral_", "_top_k50_") if "strict" not in x else x,
    )
    correction_files = sorted(
        [f for f in data_files if "scoredcorrectionnll" in f],
        key=lambda x: x.replace("_ancestral_", "_top_k50_") if "strict" not in x else x,
    )

    triplet_files = [
        (r, n, c)
        for r, n, c in zip(reward_df_files, nll_df_files, correction_files)
        if "ancestral_strict" not in r or "t1.0_" in r
    ]
    assert len(reward_df_files) == len(nll_df_files) == len(correction_files)
    for r, n, c in triplet_files:
        print()
        print(r)
        print(n)
        print(c)

    # 2. params
    title_size = 21
    legend_size = 21
    tick_label_size = 21
    marker = "o"
    legend_kwargs = {
        "loc": "lower center",
        "bbox_to_anchor": (0.5, -0.4),
        "ncol": 2,
        "markerscale": 2,
        "fontsize": legend_size,
    }

    num_corpora = 2000
    num_samples_per_corpus = 200_000

    # 3. Plotting
    fig, axs = plt.subplots(1, 2, figsize=(20, 10), dpi=300)
    total_mean_corpuses = None
    total_corpus = None
    for reward_df_file, nll_df_file, correction_df_file in tqdm(triplet_files):
        # 3.0 load and merge
        sampling_type_metadata = get_metadata_for_sampling_type(
            reward_df_file, args.cache_dir
        )
        reward_df = pd.read_csv(f"{args.input_data_dir}/{reward_df_file}")
        nll_df = pd.read_csv(f"{args.input_data_dir}/{nll_df_file}")
        correction_df = pd.read_csv(f"{args.input_data_dir}/{correction_df_file}")
        df = reward_df.merge(
            nll_df, on=["prompt", "generated_text"], how="inner"
        ).merge(correction_df, on=["prompt", "generated_text"], how="inner")

        # 3.1 populate
        df["log_probability"] = -df["negative_log_probability"]
        df["secret_reward"] = (
            df["original_negative_log_probability"] + df["log_probability"]
        )
        if args.use_secret_reward:
            print("Using secret reward.")
            df["score"] = df["secret_reward"]

        # 3.2 clean
        df = df[df.apply(lambda row: len(row["generated_text"]) > 0, axis=1)]
        df = df[~df["negative_log_probability"].isin([-np.inf, np.inf])]
        df = df[~df["negative_log_probability"].isnull()]
        df = df[~df["score"].isin([-np.inf, np.inf])]
        df = df[~df["score"].isnull()]
        df = df.drop_duplicates()
        assert len(df) > 0
        print(f"Length: {len(df)}")
        print(tabulate(df.head(), headers="keys", tablefmt="psql"))
        print(
            tabulate(
                df[["score", "log_probability"]].describe(),
                headers="keys",
                tablefmt="psql",
            )
        )

        # 3.3 plot string level
        spearman = spearmanr(df["score"], df["log_probability"])
        pearson = pearsonr(df["score"], df["log_probability"])
        print(f"Spearman: {spearman}, Pearson: {pearson}")
        df.plot.scatter(
            y="score",
            x="log_probability",
            title="String-level correlations",
            ax=axs[0],
            c=sampling_type_metadata.color,
            alpha=sampling_type_metadata.alpha,
            marker=marker,
            label=(
                f"{sampling_type_metadata.pretty_name}"
                if sampling_type_metadata.temperature == 1.5
                or sampling_type_metadata.sampling_type == "ancestral_strict"
                else None
            ),
        )
        sns.regplot(
            y="score",
            x="log_probability",
            data=df,
            ax=axs[0],
            scatter=False,
            color=sampling_type_metadata.color,
        )
        if total_corpus is None:
            total_corpus = df
        else:
            total_corpus = pd.concat([total_corpus, df])

        # 3.3 plot corpus level
        mean_corpuses_df = do_imh_bootstrap(
            df,
            num_corpora=num_corpora,
            num_samples_per_corpus=num_samples_per_corpus,
            cache_save_path=sampling_type_metadata.mean_corpuses_df_cache_filename,
            overwrite_cache=args.overwrite_cache,
        )
        spearman = spearmanr(
            mean_corpuses_df["score"],
            mean_corpuses_df["log_probability"],
        )
        pearson = pearsonr(
            mean_corpuses_df["score"],
            mean_corpuses_df["log_probability"],
        )
        print(f"Spearman: {spearman}, Pearson: {pearson}")
        mean_corpuses_df.plot.scatter(
            y="score",
            x="log_probability",
            title="Corpus-level correlations",
            ax=axs[1],
            c=sampling_type_metadata.color,
            marker=marker,
            alpha=sampling_type_metadata.alpha,
            label=(
                f"{sampling_type_metadata.pretty_name}"
                if sampling_type_metadata.temperature == 1.5
                or sampling_type_metadata.sampling_type == "ancestral_strict"
                else None
            ),
        )
        sns.regplot(
            y="score",
            x="log_probability",
            data=mean_corpuses_df,
            ax=axs[1],
            scatter=False,
            color=sampling_type_metadata.color,
        )
        if total_mean_corpuses is None:
            total_mean_corpuses = mean_corpuses_df
        else:
            total_mean_corpuses = pd.concat([total_mean_corpuses, mean_corpuses_df])

    # 4. plot total
    # 4.1 total string level
    print("===== Total corpus level statistics =====")
    spearman = spearmanr(total_corpus["score"], total_corpus["log_probability"])
    pearson = pearsonr(total_corpus["score"], total_corpus["log_probability"])
    print(f"Spearman: {spearman}, Pearson: {pearson}")
    total_corpus.plot.scatter(
        y="score",
        x="log_probability",
        alpha=0.0,
        ax=axs[0],
    )
    sns.regplot(
        y="score",
        x="log_probability",
        data=total_corpus,
        scatter=False,
        color="black",
        ax=axs[0],
        label=f"Total: ρ={spearman[0]:.2f}, r={pearson[0]:.2f}",
    )
    axs[0].set_ylabel("Reward", size=title_size)
    axs[0].set_xlabel("Log Probability", size=title_size)
    axs[0].set_title("String-level Correlations", size=title_size)
    axs[0].tick_params(axis="both", which="major", labelsize=tick_label_size)
    axs[0].legend(**legend_kwargs)

    # 4.2 total corpus level
    print("===== Total metropolis corpus mean statistics =====")
    spearman = spearmanr(
        total_mean_corpuses["score"],
        total_mean_corpuses["log_probability"],
    )
    pearson = pearsonr(
        total_mean_corpuses["score"],
        total_mean_corpuses["log_probability"],
    )
    print(f"Spearman: {spearman}, Pearson: {pearson}")
    total_mean_corpuses.plot.scatter(
        y="score",
        x="log_probability",
        alpha=0.0,
        ax=axs[1],
    )
    sns.regplot(
        y="score",
        x="log_probability",
        data=total_mean_corpuses,
        scatter=False,
        color="black",
        ax=axs[1],
        label=f"Total: ρ={spearman[0]:.2f}, r={pearson[0]:.2f}",
    )
    axs[1].set_ylabel("Average Reward", size=title_size)
    axs[1].set_xlabel("Average Log Probability", size=title_size)
    axs[1].set_title("Corpus-level Correlation", size=title_size)
    axs[1].tick_params(axis="both", which="major", labelsize=tick_label_size)
    axs[1].legend(**legend_kwargs)

    plt.legend(**legend_kwargs)
    plt.savefig(
        f"{args.figure_name}.png",
        bbox_inches="tight",
        format="png",
    )
    plt.close()


def do_imh_bootstrap(
    df: pd.DataFrame,
    num_corpora: int,
    num_samples_per_corpus: int,
    cache_save_path: str,
    overwrite_cache: bool = False,
) -> pd.DataFrame:
    # 0. check cache
    if not overwrite_cache and os.path.exists(cache_save_path):
        print(f"Loading from cache: {cache_save_path}")
        return pd.read_csv(cache_save_path)

    # 1. imh
    # see https://www.statlect.com/fundamentals-of-statistics/Metropolis-Hastings-algorithm
    # see https://projecteuclid.org/journals/annals-of-statistics/volume-24/issue-1/Rates-of-convergence-of-the-Hastings-and-Metropolis-algorithms/10.1214/aos/1033066201.full
    metropolis_df = None
    shuffled_df = df.sample(n=num_samples_per_corpus, random_state=0, replace=True)
    acceptance_thresholds = np.random.uniform(0.0, 1.0, size=len(shuffled_df))
    acceptance_count = 0
    start = shuffled_df.iloc[0]
    prev = start
    for step in prange(1, len(shuffled_df)):
        sample = shuffled_df.iloc[step]
        acceptance = (
            min(
                1.0,
                (
                    np.exp(
                        np.array(
                            -sample["original_negative_log_probability"]
                            - prev["samplingbiased_negative_log_probability"]
                            + prev["original_negative_log_probability"]
                            + sample["samplingbiased_negative_log_probability"]
                        ).astype(np.float128)
                    ).item()
                ),
            )
            > acceptance_thresholds[step],
        )
        if acceptance:
            acceptance_count += 1
            prev = sample
            to_add = df[
                (df["prompt"] == sample["prompt"])
                & (df["generated_text"] == sample["generated_text"])
            ]
        else:
            to_add = df[
                (df["prompt"] == prev["prompt"])
                & (df["generated_text"] == prev["generated_text"])
            ]
        if metropolis_df is None:
            metropolis_df = to_add
        else:
            metropolis_df = pd.concat([metropolis_df, to_add])
    algorithm_acceptance_rate = acceptance_count / len(shuffled_df)
    print(f"Algorithm acceptance rate: {algorithm_acceptance_rate}")

    # 2. compute autocorrelation
    # cramer's v: https://stats.stackexchange.com/questions/438279/autocorrelation-for-a-categorical-time-series
    # see transition matrix based test of convergence in https://arxiv.org/abs/1706.04919
    # 2.1 create an adjacency matrix by assigning every prompt + text a unique state
    states, promptandtexts = pd.factorize(
        shuffled_df.apply(
            lambda row: f"{row['prompt']};;;{row['generated_text']}", axis=1
        )
    )
    promptandtext2state = {
        promptandtext: state
        for promptandtext, state in zip(
            set(promptandtexts.to_numpy().tolist()), set(states.tolist())
        )
    }
    unique_states = list(set(states))
    transition_matrix = np.zeros((len(unique_states), len(unique_states)))

    # 2.2 track transition sequence
    transition_sequence = (
        pd.concat([start.to_frame().T, metropolis_df]).apply(
            lambda row: promptandtext2state[
                f"{row['prompt']};;;{row['generated_text']}"
            ],
            axis=1,
        )
    ).to_numpy()
    for i in range(1, len(transition_sequence)):
        transition_matrix[transition_sequence[i - 1], transition_sequence[i]] += 1

    # 2.3 remove rows and cols with only zero
    transition_matrix = transition_matrix[~np.all(transition_matrix == 0, axis=1)]
    transition_matrix = transition_matrix[:, ~np.all(transition_matrix == 0, axis=0)]

    # 2.4 cramer's v
    X2 = chi2_contingency(transition_matrix, correction=False)[0]
    N = np.sum(transition_matrix)
    min_dim = min(transition_matrix.shape) - 1
    cramer = np.sqrt(X2 / (N * min_dim))
    print(f"Cramer's V: {cramer}")

    # 3. resample corpora
    assert type(metropolis_df) is pd.DataFrame
    corpuses = {
        corpus_seed: metropolis_df.sample(
            num_samples_per_corpus, random_state=corpus_seed, replace=True
        )
        for corpus_seed in range(num_corpora)
    }
    mean_corpuses_df = pd.DataFrame.from_dict({
        corpus_seed: corpus[["score", "log_probability"]].mean(axis=0)
        for corpus_seed, corpus in corpuses.items()
    }).T

    # 4. save
    mean_corpuses_df.to_csv(cache_save_path, index=False)
    print(f"Saved to cache: {cache_save_path}")
    return mean_corpuses_df


def get_metadata_for_sampling_type(
    filename: str, output_dir: str
) -> SamplingTypeMetadata:
    # 1. parse sampling type and temperature
    sampling_type_keywords = {
        "top_p095",
        "top_p090",
        "top_k50",
        "top_k30",
        "ancestral_strict",
        "ancestral",
        "typical_p090",
        "eta_n00009",
    }
    sampling_keyword_to_sampling_type = {
        "_" + k + "_t": k for k in sampling_type_keywords
    }
    sampling_keyword_candidates = [
        k for k in sampling_keyword_to_sampling_type.keys() if k in filename
    ]
    assert len(sampling_keyword_candidates) == 1, f"Found {sampling_keyword_candidates}"
    sampling_keyword = sampling_keyword_candidates[0]
    sampling_type = sampling_keyword_to_sampling_type[sampling_keyword]
    temperature = float(filename.split(sampling_keyword)[1].split("_")[0])

    # assign colours by temperature and type
    colors = get_matplotlib_colours()
    sampling_type_to_colour = {
        k: colors[i] for i, k in enumerate(sampling_type_keywords)
    }
    alpha = max((temperature - 0.5) / (1.5 - 0.5), 0.1)
    color = color_mixer(
        color_mixer("#ffffff", sampling_type_to_colour[sampling_type], 0.25),
        sampling_type_to_colour[sampling_type],
        alpha,
    )

    # assign names
    sampling_type_to_pretty_name = {
        "top_p095": "Nucleus (π=0.95)",
        "top_p090": "Nucleus (π=0.90)",
        "top_k50": "Top-k (k=50)",
        "top_k30": "Top-k (k=30)",
        "ancestral_strict": "Ancestral",
        "ancestral": "Top-k (k=50)",
        "typical_p090": "Locally Typical",
        "eta_n00009": "η",
    }

    # intermediate files
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    mean_corpuses_df_cache_filename = str(
        pathlib.Path(output_dir, f"{sampling_type}_t{temperature}_mean_corpuses_df.csv")
    )

    return SamplingTypeMetadata(
        filename=filename,
        sampling_type=sampling_type,  # type: ignore
        temperature=temperature,
        color=color,
        pretty_name=sampling_type_to_pretty_name[sampling_type],
        alpha=alpha,
        mean_corpuses_df_cache_filename=mean_corpuses_df_cache_filename,
    )


def get_matplotlib_colours() -> List[str]:
    rejected_colour_filters = [
        lambda x: x != "w" and x != "k",  # no black or white in cmyk
        # no whites, grays or blacks in css colours
        lambda x: x
        not in {
            "snow",
            "ivory",
            "black",
            "seashell",
            "azure",
            "aliceblue",
            "honeydew",
            "oldlace",
            "bisque",
            "blanchedalmond",
            "papayawhip",
            "beige",
            "lemonchiffon",
            "linen",
            "moccasin",
            "wheat",
            "peachpuff",
        },
        lambda x: "gray" not in x
        and "grey" not in x
        and "white" not in x
        and "light" not in x
        and "misty" not in x
        and "silk" not in x
        and "mint" not in x
        and "lavendar" not in x
        and "pale" not in x,
    ]

    return [
        c
        for c in (
            list(mcolors.BASE_COLORS.keys())
            + list(mcolors.TABLEAU_COLORS.keys())
            + list(mcolors.CSS4_COLORS.keys())
        )
        if all(f(c) for f in rejected_colour_filters)
    ]


def color_mixer(
    c1: str, c2: str, mix: float = 0
):  # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    assert 0 <= mix <= 1
    return matplotlib.colors.to_hex(
        (1 - mix) * np.array(matplotlib.colors.to_rgb(c1))  # type: ignore
        + mix * np.array(matplotlib.colors.to_rgb(c2))
    )


if __name__ == "__main__":
    main()
