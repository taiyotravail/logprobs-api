"""Visualisations matplotlib pour les logprobs."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt


def plot_token_probabilities(token_data) -> plt.Figure:
    """Scatter plot des top-3 probabilités pour chaque token généré.

    Les alternatives sont grisées, le token effectivement choisi est mis en
    évidence en bleu.
    """
    alt_x, alt_y = [], []
    chosen_x, chosen_y = [], []
    x_labels = []

    for idx, item in enumerate(token_data, start=1):
        x_labels.append(f"T{idx}\n{repr(item.token)}")
        chosen_x.append(idx)
        chosen_y.append(math.exp(item.logprob) * 100)
        if item.top_logprobs:
            for alt in item.top_logprobs:
                alt_x.append(idx)
                alt_y.append(math.exp(alt.logprob) * 100)

    fig, ax = plt.subplots(figsize=(max(8, len(token_data) * 0.5), 5))
    ax.scatter(
        alt_x,
        alt_y,
        s=40,
        alpha=0.5,
        color="#94a3b8",
        edgecolors="black",
        label="Alternatives (top 3)",
        zorder=2,
    )
    ax.scatter(
        chosen_x,
        chosen_y,
        s=80,
        color="#2563eb",
        edgecolors="black",
        label="Token choisi",
        zorder=3,
    )
    ax.set_ylim(0, 105)
    ax.set_ylabel("Probabilité (%)")
    ax.set_xticks(range(1, len(x_labels) + 1))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig
