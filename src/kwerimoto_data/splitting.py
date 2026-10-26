"""Step 4b: assign whole groups to train / validation / test.

Groups are placed one at a time, largest first, each into the split that keeps
every class closest to its target share (70/15/15 by default). Groups of the
same size are shuffled with the seed, so the result is random but repeatable.
A second pass then moves single groups to another split while that brings the
classes closer to their targets; this fixes cases where classes share many
groups and the first pass gets stuck.
"""
import numpy as np
import pandas as pd

FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def split_groups(df, fractions=FRACTIONS, seed=42, by="label"):
    """Return one split name per row of `df` (needs a `group` column and the `by` column).

    Each value of `by` is kept close to `fractions`; pass a finer column (for
    example label plus source) to balance real and synthetic clips separately.
    """
    names = list(fractions)
    labels = sorted(df[by].unique())
    counts = pd.crosstab(df["group"], df[by]).reindex(columns=labels, fill_value=0)
    totals = counts.sum().to_numpy(dtype=float)
    targets = np.outer([fractions[n] for n in names], totals)

    rng = np.random.default_rng(seed)
    order = counts.index[rng.permutation(len(counts))]
    sizes = counts.loc[order].sum(axis=1)
    order = sizes.sort_values(ascending=False, kind="stable").index

    def cost(assigned):
        # How far each class sits from its target, as a share of that class.
        return (((assigned - targets) / totals) ** 2).sum()

    def costs_if_placed(assigned, add):
        """Cost of putting a group with class counts `add` into each split."""
        options = []
        for s in range(len(names)):
            trial = assigned.copy()
            trial[s] += add
            options.append(cost(trial))
        return options

    assigned = np.zeros_like(targets)
    group_split = {}
    for group in order:
        add = counts.loc[group].to_numpy(dtype=float)
        options = costs_if_placed(assigned, add)
        best = int(np.argmin(options))
        assigned[best] += add
        group_split[group] = best

    for _ in range(100):  # second pass: move groups while it helps
        improved = False
        for group in order:
            add = counts.loc[group].to_numpy(dtype=float)
            current = group_split[group]
            without = assigned.copy()
            without[current] -= add
            options = costs_if_placed(without, add)
            best = int(np.argmin(options))
            if options[best] < options[current] - 1e-12:
                assigned = without
                assigned[best] += add
                group_split[group] = best
                improved = True
        if not improved:
            break
    return df["group"].map({g: names[s] for g, s in group_split.items()}).rename("split")
