import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def plot_target_data(ser, figsize=(7, 5), x_ticklabels=None):
    if x_ticklabels is None:
        x_ticklabels = ['On Time', 'Payment Difficulty']
    print(f"There are {ser.value_counts()[0]} applicants who could pay on time.")
    print(f"There are {ser.value_counts()[1]} applicants with payment difficulties.")

    fig, ax = plt.subplots(figsize=figsize)
    ser.value_counts(normalize=True).plot(kind='bar', ax=ax)
    sns.despine(left=True, bottom=True)
    for bar in ax.patches:
        ax.annotate(format(bar.get_height(), '.2f'),
                    (bar.get_x() + bar.get_width() / 2,
                     bar.get_height()), ha='center', va='center',
                    size=14, xytext=(0, 8),
                    textcoords='offset points')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(x_ticklabels, rotation=0)

    ax.axes.get_yaxis().set_visible(False)
    ax.set_title('Proportion of Target Class')
    ax.set_ylabel('Proportion')
    ax.set_xlabel("")
    plt.show()
    return fig, ax


def plot_correlation_matrix(corr_matrix, mask='upper', threshold=None, figsize=(14, 10),
                            title='Phik Correlation Heatmap', xlabel="", ylabel=""):
    """
    Plots a heatmap of a correlation matrix with options to mask upper/lower triangles and apply thresholds.

    Parameters:
        corr_matrix (pd.DataFrame): The correlation matrix to plot.
        mask (str or None): Mask option. 'upper' for upper triangle, 'lower' for lower triangle, or None for no mask.
        threshold (float or None): If set, only show correlations above this threshold.
        figsize (tuple): The size of the figure (default is (14, 10)).
        title (str): The title of the plot (default is 'Phik Correlation Heatmap').
        xlabel (str): The label for the x-axis (default is an empty string).
        ylabel (str): The label for the y-axis (default is an empty string).

    Returns:
        ax: The matplotlib axis with the plot.
    """
    if mask == 'upper':
        mask_array = np.triu(np.ones_like(corr_matrix, dtype=bool))
    elif mask == 'lower':
        mask_array = np.tril(np.ones_like(corr_matrix, dtype=bool))
    else:
        mask_array = None

    if threshold is not None:
        below_threshold = np.abs(corr_matrix) < threshold
        mask_array = np.logical_or(mask_array, below_threshold) if mask_array is not None else below_threshold

    plt.figure(figsize=figsize)
    ax = sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap="coolwarm", center=0,
                     mask=mask_array, annot_kws={"size": 10},
                     cbar_kws={"shrink": .8, "label": 'Correlation', "ticks": [-1, 0, 1]},
                     linewidths=.5, linecolor='gray')

    plt.xticks(fontsize=12, rotation=45, ha='right')
    plt.yticks(fontsize=12)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    plt.show()

    return ax


def subplot_sns_histogram(
    df,
    list_of_features,
    title,
    nrows,
    ncols,
    bins,
    figsize=(10, 10),
    hue=None,
    kde=True,
):
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axs = axs.flatten()

    for i, feature in enumerate(list_of_features):
        sns.histplot(data=df, x=feature, hue=hue, kde=kde, ax=axs[i], bins=bins)
        axs[i].set_title(f"Distribution of {feature}")

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle(title)
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.tight_layout()
    plt.show()
    return fig, axs


def subplot_sns_boxplot(
    df,
    list_of_features,
    nrows,
    ncols,
    hue=None,
    figsize=(10, 10),
    title="",
    legend_loc="upper right",
):
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axs = axs.flatten()

    for i, feature in enumerate(list_of_features):
        sns.boxplot(data=df, x=feature, hue=hue, ax=axs[i])
        axs[i].set_title(f"Distribution of {feature}")

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    if hue:
        handles, labels = axs[0].get_legend_handles_labels()

        fig.legend(handles, labels, loc=legend_loc, title=hue)

        for ax in axs:
            if ax.get_legend() is not None:
                ax.legend_.remove()

    fig.suptitle(title)
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.tight_layout()
    plt.show()

    return fig, axs


def horizontal_bar_value_counts_subplots(
    df, list_of_features, nrows, ncols, figsize=(10, 10)
):
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    axs = axs.flatten()

    for i, feature in enumerate(list_of_features):
        if i >= len(axs):
            break

        df[feature].value_counts().plot.barh(ax=axs[i])
        axs[i].set_xlabel("Count")
        axs[i].set_ylabel(f"{feature}")

        for p in axs[i].patches:
            axs[i].annotate(
                f"{p.get_width():.0f}",
                (p.get_width(), p.get_y() + p.get_height() / 2),
                ha="left",
                va="center",
                fontsize=10,
            )

    plt.tight_layout()
    plt.show()

    return fig, axs


def plot_feature_qqplot_subplot(df, list_of_features, nrows, ncols, figsize, title):
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axs = axs.flatten()

    for i, feat in enumerate(list_of_features):
        df_no_nan = df[~df[feat].isna()]
        stats.probplot(df_no_nan[feat], dist="norm", plot=axs[i])

        axs[i].set_title(feat)
        axs[i].set_xlabel("Theoretical Quantiles")
        axs[i].set_ylabel("Ordered Values")

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

    return fig, axs


def plot_normalized_counts_by_target(
    df, list_of_features, target, nrows, ncols, figsize=(16, 16)
):

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df must be a pandas DataFrame.")
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in DataFrame columns.")

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    axs = axs.flatten()

    for i, feature in enumerate(list_of_features):
        if feature not in df.columns:
            print(f"Warning: Feature '{feature}' not found in DataFrame. Skipping.")
            continue

        temp_counts = df.groupby([feature, target], observed=False).size()
        temp_counts_normalized = (
            temp_counts.groupby(level=0, observed=False)
            .transform(lambda x: x / x.sum())
            .unstack()
        )

        temp_counts_normalized.plot.barh(stacked=True, ax=axs[i])
        axs[i].set_title(f"Normalized {feature} Distribution by Target")
        axs[i].set_xlabel("Proportion")
        axs[i].set_ylabel(feature)

        for container in axs[i].containers:
            axs[i].bar_label(container, fmt="%.2f", label_type="center", fontsize=9)

    plt.tight_layout()
    plt.show()

    return fig, axs


def plot_missing_values_by_target_class(ext_1_na_df, ext_2_na_df, ext_3_na_df, ext_mean_na_df, ext_std_na_df):
    colors = ["#4878CF", "#EE854A"]

    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    fig.suptitle("Missing Values by Target Class")

    def add_labels(ax):
        for p in ax.patches:
            value = f"{p.get_height():.2f}"
            ax.text(
                p.get_x() + p.get_width() / 2,
                p.get_height() + 0.01,
                value,
                ha="center",
                fontsize=10,
            )

    for ax, dataframe, title in zip(
        axs.flatten()[:5],
        [ext_1_na_df, ext_2_na_df, ext_3_na_df, ext_mean_na_df, ext_std_na_df],
        [
            "Ext Source 1",
            "Ext Source 2",
            "Ext Source 3",
            "Ext Source MEAN",
            "Ext Source STD",
        ],
    ):
        dataframe["TARGET"].value_counts(normalize=True).plot.bar(
            ax=ax, title=title, xlabel="", color=colors
        )
        add_labels(ax)
        ax.set_yticks([])

        ax.spines["left"].set_visible(False)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axs[2, 1].axis("off")

    plt.tight_layout()
    plt.show()
    return fig, axs


def plot_subplots_correlation_matrix(
    corr_mat,
    list_of_features,
    n_features,
    nrows,
    ncols,
    figsize=(16, 16),
    title="Phi K Correlation Matrices for External Sources",
):
    vmin = 0
    vmax = 1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axs = axs.flatten()
    list_of_feat_correlation_dfs = []
    fig.suptitle(title)
    for i, feat in enumerate(list_of_features):
        feat_corr = pd.DataFrame(
            corr_mat[feat].sort_values(ascending=False)[: n_features + 1]
        )
        sns.heatmap(
            feat_corr,
            annot=True,
            fmt=".3f",
            cmap="Blues_r",
            vmin=vmin,
            vmax=vmax,
            ax=axs[i],
        )
        list_of_feat_correlation_dfs.append(feat_corr)
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])
    plt.tight_layout()
    plt.show()
    return fig, axs, list_of_feat_correlation_dfs


def horizontal_bar_value_counts_by_class_subplots(
    df, list_of_features, target, nrows, ncols, figsize=(10, 10)
):
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    axs = axs.flatten()

    for i, feature in enumerate(list_of_features):
        if i >= len(axs):
            break

        # Group the data by feature and target class, and count the occurrences
        feature_counts = (
            df.groupby([feature, target], observed=False).size().unstack(fill_value=0)
        )

        # Plot the counts as horizontal bars
        feature_counts.plot(
            kind="barh", stacked=False, ax=axs[i], color=["#1f77b4", "#ff7f0e"]
        )  # Adjust colors as needed
        axs[i].set_xlabel("Count")
        axs[i].set_ylabel(f"{feature}")
        axs[i].set_title(f"{feature} by {target}")

        # Add labels to each bar
        for p in axs[i].patches:
            axs[i].annotate(
                f"{p.get_width():.0f}",
                (p.get_width(), p.get_y() + p.get_height() / 2),
                ha="left",
                va="center",
                fontsize=10,
            )

    plt.tight_layout()
    plt.show()

    return fig, axs


def conf_mat_plot(y, predictions, disp_labels=None, figsize=(8, 6), title="Confusion matrix"):
    if disp_labels is None:
        disp_labels = ["On Time", "Payment Difficulty"]
    conf_matrix = confusion_matrix(y, predictions)
    cm_display = ConfusionMatrixDisplay(
        conf_matrix, display_labels=disp_labels
    )
    fig, ax = plt.subplots(figsize=figsize)
    cm_display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.grid(False)
    ax.set_title(title, fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    for text in ax.texts:
        text.set_color("black")
        text.set_fontsize(14)

    plt.tight_layout()
    plt.show()

    return fig, ax


def plot_pr_curve(y, pred_probs, title="Precision-Recall Curve"):
    precision, recall, thresholds = precision_recall_curve(y, pred_probs)
    auc_score = auc(recall, precision)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"Precision-Recall Curve (AUC = {auc_score:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.show()
    return precision, recall, thresholds, auc_score


def plot_roc_curve(y, pred_probs, title="ROC Curve"):
    fpr, tpr, thresh = roc_curve(y, pred_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.show()

    return fpr, tpr, thresh, roc_auc
