import itertools
import json
import os

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator

import envs

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def figsize_third(scale, height_ratio=1.0):
    fig_width_pt = 156  # Get this from LaTeX using \the\columnwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width * golden_mean * height_ratio  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


def figsize_column(scale, height_ratio=1.0):
    fig_width_pt = 234  # Get this from LaTeX using \the\columnwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width * golden_mean * height_ratio  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


def figsize_text(scale, height_ratio=1.0):
    fig_width_pt = 468  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width * golden_mean * height_ratio  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


pgf_with_latex = {  # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",  # change this if using xetex or luatex
    "text.usetex": True,  # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 8,
    "font.size": 8,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize_column(1.0),
    "legend.framealpha": 1.0,
    "text.latex.preamble": [
        r"\usepackage[utf8x]{inputenc}",  # use utf8 fonts because your computer can handle it :)
        r"\usepackage[T1]{fontenc}",  # plots will be generated using this preamble
    ],
}
matplotlib.rcParams.update(pgf_with_latex)

BASICNAMES = {
    "blur": "Blur",
    "fliplr": "Flip L/R",
    "flipud": "Flip U/D",
    "grayscale": "Grayscale",
    "invert": "Invert",
    "rotation": "Rotation",
    "shear": "Shear",
}


def read_files(environment, scenario, dataset, agent, logbasedir):
    summary = {}
    stats = {}
    # for ag in [agent]:
    expdir = "exp_{environment}_{scenario}_{dataset}_{agent}".format(
        environment=environment, scenario=scenario, dataset=dataset, agent=agent
    )
    logdir = os.path.join(logbasedir, expdir)
    df_summary, df_stats = group_logs(logdir)
    df_summary[["rel_failure", "original_accuracy", "modified_accuracy"]] *= 100
    summary[agent] = df_summary
    stats[agent] = df_stats

    bl = get_baseline(environment, scenario, dataset)
    return summary, stats, bl


def evaluate_classification(scenario, dataset, agent, logbasedir, outdir):
    SUM_NEW_COLS = {
        "rel_failure": "Failure Rate",
        "original_accuracy": "Accuracy (Orig.)",
        "modified_accuracy": "Accuracy (MR)",
        "failure_baseline": "Failure Rate (Baseline)",
        "modified_accuracy_baseline": "Accuracy (MR, Baseline)",
    }

    plot_charts(
        "classification", scenario, dataset, agent, logbasedir, outdir, SUM_NEW_COLS
    )


def evaluate_detection(scenario, dataset, agent, logbasedir, outdir):
    SUM_NEW_COLS = {
        "rel_failure": "Failure Rate",
        "original_accuracy": "mAP (Orig.)",
        "modified_accuracy": "mAP (MR)",
        "failure_baseline": "Failure Rate (Baseline)",
        "modified_accuracy_baseline": "mAP (MR, Baseline)",
    }

    plot_charts("detection", scenario, dataset, agent, logbasedir, outdir, SUM_NEW_COLS)


def plot_charts(
    environment, scenario, dataset, agent, logbasedir, outdir, summary_columns
):
    summary, stats, bl = read_files(environment, scenario, dataset, agent, logbasedir)

    print(environment, scenario, dataset, agent)
    print("avg. duration per iteration: ", summary[agent].duration.mean())
    #print("rel_failure: ", summary[agent].rel_failure.last())
    print("orig. accuracy: ", summary[agent].original_accuracy.mean())
    print("mod. accuracy: ", summary[agent].modified_accuracy.mean())

    plot_progress(
        environment, dataset, scenario, agent, summary[agent], bl, outdir, summary_columns
    )

    if scenario == "basic":
        plot_basic_action_distribution(
            environment, dataset, scenario, agent, stats[agent], bl, outdir
        )
    elif scenario == "rotation":
        plot_parametrized_action_distribution(
            environment, dataset, scenario, "rotation", agent, stats[agent], bl, outdir
        )
    elif scenario == "shear":
        plot_parametrized_action_distribution(
            environment, dataset, scenario, "shear", agent, stats[agent], bl, outdir
        )
    else:
        rot_actions = stats[agent].action.str.match(r"rot[-\d]+")
        shear_actions = stats[agent].action.str.match(r"shear[-\d]+")
        plot_basic_action_distribution(
            environment,
            dataset,
            scenario,
            agent,
            stats[agent][(~rot_actions) & (~shear_actions)],
            bl,
            outdir,
        )
        plot_parametrized_action_distribution(
            environment,
            dataset,
            scenario,
            "rotation",
            agent,
            stats[agent][rot_actions],
            bl,
            outdir,
        )
        plot_parametrized_action_distribution(
            environment,
            dataset,
            scenario,
            "shear",
            agent,
            stats[agent][shear_actions],
            bl,
            outdir,
        )


def plot_progress(environment, dataset, scenario, agent, df_summary, bl, outdir, colnames):
    df_summary["failure_baseline"] = bl["failure"].mean() * 100
    df_summary["modified_accuracy_baseline"] = bl["modified_accuracy"].mean() * 100

    #colors = sns.color_palette("colorblind", n_colors=3)
    colors = ["#9b59b6", "#3498db", "#e74c3c"]
    c1 = colors[0]
    c2 = colors[1]
    c3 = colors[2]
    ax = df_summary.rename(columns=colnames).plot(
        x="iteration",
        y=[
            colnames["original_accuracy"],
            colnames["modified_accuracy"],
            colnames["rel_failure"],
            colnames["modified_accuracy_baseline"],
            colnames["failure_baseline"],
        ],
        style=["-", "-", "-", "--", "--"],
        color=[c1, c2, c3, c2, c3],
        figsize=figsize_column(1.1, height_ratio=0.7),
    )
    ax.grid()
    ax.set_xlabel("Iteration")
    ax.set_xlim([0, df_summary.iteration.max()])
    ax.set_ylim([0, 100])

    # if environment == 'classification':
    #    ax.set_ylim([25, 100])

    plt.locator_params(axis="y", nbins=7)

    hand, labl = ax.get_legend_handles_labels()
    ax.legend(
        hand[:3], labl[:3], ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.35), fancybox=False
    )

    sns.despine()
    plt.savefig(
        os.path.join(
            outdir, "{}-{}-{}-{}-process.pgf".format(environment, dataset, scenario, agent)
        ),
        dpi=500,
        bbox_inches="tight",
        pad_inches=0,
    )


def plot_basic_action_distribution(
    environment, dataset, scenario, agent, df_stats, bl, outdir
):
    bl["action"].replace(BASICNAMES, inplace=True)
    df_stats["action"].replace(BASICNAMES, inplace=True)

    df_final = df_stats.loc[
        df_stats.iteration == df_stats.iteration.max(), ["action", "rel_failure"]
    ]
    df_final.set_index("action", inplace=True)
    combdf = df_final.join(bl[["action", "failure"]].groupby("action").mean())
    combdf.rename(
        columns={"failure": "Baseline", "rel_failure": "Tetraband"}, inplace=True
    )
    combdf *= 100
    combdf.sort_values(["Tetraband"], inplace=True)

    palette = sns.color_palette("colorblind", 4)

    if environment == "detection":
        palette = palette[2:]

    sns.set_palette(palette)

    combdf.round(2).to_latex(
        open(
            os.path.join(
                outdir,
                "{}-{}-{}-main-{}-actions.tex".format(
                    environment, dataset, scenario, agent
                ),
            ),
            "w",
        )
    )

    ax = combdf.plot.bar(
        y=["Tetraband", "Baseline"], figsize=figsize_text(1.1, height_ratio=0.3),
        width=0.85,
        edgecolor='white'
    )

    bars = ax.patches
    hatches = ''.join(h * len(combdf) for h in '/.')

    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    labels = ax.get_xticklabels()
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("Failure Rate (in %)")
    ax.legend(loc="upper left")
    plt.locator_params(axis="y", nbins=7)

    sns.despine()
    plt.savefig(
        "{}-{}-{}-main-{}-actions.pdf".format(environment, dataset, scenario, agent),
        dpi=500,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.savefig(
        os.path.join(
            outdir,
            "{}-{}-{}-main-{}-actions.pgf".format(
                environment, dataset, scenario, agent
            ),
        ),
        dpi=500,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


def plot_parametrized_action_distribution(
    environment, dataset, scenario, action_name, agent, df_stats, bl, outdir
):
    df_final = df_stats.loc[
        df_stats.iteration == df_stats.iteration.max(), ["action", "rel_failure"]
    ]
    df_final.set_index("action", inplace=True)
    combdf = df_final.join(bl[["parameter", "failure"]].groupby("parameter").mean())
    combdf.rename(
        columns={"failure": "Baseline", "rel_failure": "Tetraband"}, inplace=True
    )
    combdf *= 100
    combdf["deg"] = combdf.index.str.replace(r"[a-z]+", "").map(int)
    combdf = combdf.append(pd.Series({"deg": 0, "Baseline": 0, "Tetraband": 0}), ignore_index=True)
    combdf.sort_values(["deg"], inplace=True)
    combdf.set_index("deg", inplace=True)

    combdf.round(2).to_latex(
        open(
            os.path.join(
                outdir,
                "{}-{}-{}-{}-{}-actions.tex".format(
                    environment, dataset, scenario, action_name, agent
                ),
            ),
            "w",
        )
    )

    palette = sns.color_palette("colorblind", 4)
    # palette = sns.color_palette("Paired", 4)

    if environment == "classification":
        palette = palette[:2]
    else:
        palette = palette[2:]

    sns.set_palette(palette)

    fig, ax = plt.subplots(figsize=figsize_column(1.1, height_ratio=0.75))
    combdf["Tetraband"].plot.line(linestyle='-', ax=ax)
    combdf["Baseline"].plot.line(linestyle='--', ax=ax)
    # ax = combdf.plot.bar(
    #     y=["Tetraband", "Baseline"],
    #     # edgecolor='white',
    #     #width=0.9 if action_name == "rotation" else 0.8,
    #     figsize=figsize_text(1.1, height_ratio=0.3),
    # )

    if action_name == "rotation":
        ax.set_ylabel("Failure Rate (in \%)")
        ax.xaxis.set_major_locator(MultipleLocator(20))
    else:
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.legend(fancybox=False, loc="upper center", ncol=1)

    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.axvline(0, linewidth=1, linestyle="--", c="k")
    ax.grid(axis="y", which='both')
    ax.set_xlabel("")
    ax.yaxis.set_minor_locator(MultipleLocator(10))

    if environment == "classification":
        ax.set_ylim([0, 80])
    else:
        ax.set_ylim([50, 100])
        plt.locator_params(axis="y", nbins=5)

    sns.despine()
    plt.savefig(
        "{}-{}-{}-{}-{}-actions-wide.pdf".format(
            environment, dataset, scenario, action_name, agent
        ),
        dpi=500,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.savefig(
        os.path.join(
            outdir,
            "{}-{}-{}-{}-{}-actions-wide.pgf".format(
                environment, dataset, scenario, action_name, agent
            ),
        ),
        dpi=500,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


def get_action_names(basename, scenario, dataset):
    env_name = "{basename}-{scenario}-{dataset}-v0".format(
        basename=basename, scenario=scenario, dataset=dataset
    )
    env = gym.make(env_name)
    action_names = env.action_names()
    return action_names


def get_baseline(envname, scenario, dataset):
    filename = "baseline_{}_{}_{}.csv".format(envname, scenario, dataset)
    df = pd.read_csv(os.path.join("logs", filename), sep=";")

    if envname == "classification":
        df["original_accuracy"] = df["original"] == df["label"]
        df["modified_accuracy"] = df["prediction"] == df["label"]
        df["failure"] = df["original"] != df["prediction"]
    else:
        df["original_accuracy"] = df["original_score"]
        df["modified_accuracy"] = df["modified_score"]
        # Success was defined as a passing test case, but in the evaluation we see it differently
        df["failure"] = ~df["success"]

    if scenario in ("rotation", "shear"):
        df["parameter"] = df["action"]
        df["action"] = scenario

    df["action_orig"] = df["action"]

    for actidx, actname in enumerate(
        get_action_names(envs.BASENAMES[envname], scenario, dataset)
    ):
        df.loc[df.action == actidx, "action"] = actname

    return df


def load_log(logfile):
    summary = []
    action_stats = []

    for l in open(logfile, "r"):
        rowdict = json.loads(l)
        # rowdict = yaml.safe_load(l)
        summary.append({k: v for k, v in rowdict.items() if k != "statistics"})

        for act in rowdict["statistics"].values():
            act["iteration"] = rowdict["iteration"]
            action_stats.append(act)

    # print(rowdict)
    df_summary = pd.DataFrame.from_records(summary)
    df_summary.rename(columns={"success": "failure"}, inplace=True)
    df_summary["rel_failure"] = df_summary["failure"] / df_summary["iteration"]

    df_stats = pd.DataFrame.from_records(action_stats)
    df_stats.rename(columns={"success": "failure"}, inplace=True)
    df_stats["rel_failure"] = df_stats["failure"] / df_stats["count"]
    return df_summary, df_stats


def group_logs(logdir):
    stats = []
    summary = []

    for f in os.listdir(logdir):
        df_summary, df_stats = load_log(os.path.join(logdir, f))

        summary.append(df_summary)
        stats.append(df_stats)

    df_summary = pd.concat(summary)
    df_summary = df_summary.groupby(
        ["agent", "env", "iteration"], as_index=False
    ).mean()
    df_summary["rel_failure"] = df_summary["failure"] / df_summary["iteration"]

    df_stats = pd.concat(stats)
    df_stats = df_stats.groupby(["action", "iteration"], as_index=False).mean()
    df_stats["rel_failure"] = df_stats["failure"] / df_stats["count"]
    return df_summary, df_stats


def effects_table(logfile, outdir):
    df = pd.read_csv(logfile, sep=";")
    df["has_effect"] = df["original"] != df["prediction"]
    df["original_name"] = df["original"].apply(
        lambda x: CIFAR10_CLASSES[x].capitalize()
    )
    df["prediction_name"] = df["prediction"].apply(
        lambda x: CIFAR10_CLASSES[x].capitalize()
    )
    df["label_name"] = df["label"].apply(lambda x: CIFAR10_CLASSES[x].capitalize())

    action_names = [
        a[0]
        for a in envs.classification.ImageClassificationEnv(
            "basic", "difference"
        ).action_names()
    ]
    df["action_name"] = df["action"].apply(lambda x: BASICNAMES[x])

    # MR effects per action and class
    epac = (
        df[["action_name", "label_name", "has_effect"]]
        .groupby(["action_name", "label_name"], as_index=False)
        .mean()
    )
    epac["has_effect"] *= 100
    epac_pivot = epac.pivot_table(
        index="action_name",
        columns="label_name",
        values="has_effect",
        margins=True,
        margins_name="All",
    ).round(1)
    print(epac_pivot)
    epac_pivot.to_latex(
        buf=open(os.path.join(outdir, "effects_per_action.tex"), "w"), index_names=False
    )

    # MR effects: class -> class
    df["dummy"] = 1
    mrcc = (
        df[["label_name", "prediction_name", "dummy"]]
        .groupby(["label_name", "prediction_name"], as_index=False)
        .count()
    )
    mrcc["dummy"] = mrcc["dummy"] / 9000 * 100  # scale by number of images per class
    mrcc_pivot = mrcc.pivot_table(
        index="label_name", columns="prediction_name", values="dummy", margins=True
    ).round(2)
    mrcc_pivot.drop(["All"], inplace=True, axis=1)
    print(mrcc_pivot)
    mrcc_pivot.to_latex(
        buf=open(os.path.join(outdir, "effects_class2class.tex"), "w"),
        index_names=False,
    )


def effect_examples(outdir, image_idx=4800):
    env = envs.classification.ImageClassificationEnv(
        "basic", "difference", dataset="imagenet"
    )

    original_image, label = env._get_image(image_idx)
    print(image_idx, label)
    examples = [("original", original_image)]

    for action_idx in range(len(env.actions)):
        modified_image = env.get_action(action_idx)(image=original_image)
        examples.append((env.get_action_name(action_idx, None)[0], modified_image))

    for action_name, image in examples:
        image.save(os.path.join(outdir, "{}_{}.pdf".format(image_idx, action_name)))


if __name__ == "__main__":
    outdir = "/home/helge/Dropbox/Apps/ShareLaTeX/MR Selection/figures/"
    effects_table('logs/baseline_classification_hierarchical_cifar10.csv', outdir)
    # effect_examples(outdir)

    # evaluate_classification("hierarchical", "cifar10", "bandit", "logs/", outdir)
    # evaluate_classification("rotation", "cifar10", "bandit", "logs/", outdir)
    # evaluate_classification("basic", "cifar10", "bandit", "logs/", outdir)
    # evaluate_classification("shear", "cifar10", "bandit", "logs/", outdir)
    # #
    # # evaluate_classification('basic', 'cifar10', 'random', 'logs/', outdir)
    # # evaluate_classification('hierarchical', 'cifar10', 'random', 'logs/', outdir)
    # # evaluate_classification('rotation', 'cifar10', 'random', 'logs/', outdir)
    # # evaluate_classification('shear', 'cifar10', 'random', 'logs/', outdir)
    #
    # evaluate_classification("basic", "imagenet", "bandit", "logs/", outdir)
    # evaluate_classification("hierarchical", "imagenet", "bandit", "logs/", outdir)
    # # evaluate_classification('rotation', 'imagenet', 'bandit', 'logs/', outdir)
    # # evaluate_classification('shear', 'imagenet', 'bandit', 'logs/', outdir)
    #
    # evaluate_classification('basic', 'cifar10', 'random', 'logs/', outdir)
    evaluate_classification('hierarchical', 'cifar10', 'random', 'logs/', outdir)
    # evaluate_classification('rotation', 'cifar10', 'random', 'logs/', outdir)
    # evaluate_classification('shear', 'cifar10', 'random', 'logs/', outdir)

    #evaluate_classification("basic", "imagenet", "bandit", "logs/", outdir)
    #evaluate_classification("hierarchical", "imagenet", "bandit", "logs/", outdir)
    # evaluate_classification('rotation', 'imagenet', 'bandit', 'logs/', outdir)
    # evaluate_classification('shear', 'imagenet', 'bandit', 'logs/', outdir)

    #evaluate_detection('basic', 'coco', 'bandit', 'logs/', outdir)
    evaluate_detection('hierarchical', 'coco', 'bandit', 'logs/', outdir)
    # evaluate_detection('rotation', 'coco', 'bandit', 'logs/', outdir)
    # evaluate_detection('shear', 'coco', 'bandit', 'logs/', outdir)
    #
    # evaluate_detection('basic', 'coco', 'random', 'logs/', outdir)
    evaluate_detection('hierarchical', 'coco', 'random', 'logs/', outdir)
    # evaluate_detection('rotation', 'coco', 'random', 'logs/', outdir)
    # evaluate_detection('shear', 'coco', 'random', 'logs/', outdir)
