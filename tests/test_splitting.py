import numpy as np
import pandas as pd

from kwerimoto_data.splitting import split_groups


def realistic_frame():
    """Roughly our real mix: many single-clip groups, a few large chains."""
    rng = np.random.default_rng(0)
    rows = []
    n = 0

    def add(group, label):
        nonlocal n
        rows.append({"record_id": f"r{n}", "group": group, "label": label})
        n += 1

    for i in range(26):
        add("chain26", "engine_knock" if i < 23 else "healthy")
    for i in range(20):
        add("chain20", "engine_knock" if i < 15 else "healthy")
    for g in range(300):
        size = int(rng.choice([1, 1, 1, 2, 3]))
        label = str(rng.choice(["healthy", "engine_knock"]))
        for _ in range(size):
            add(f"revix{g}", label)
    for i in range(83):
        add(f"hf_h{i}", "healthy")
    for i in range(111):
        add(f"hf_m{i}", "misfire")
    return pd.DataFrame(rows)


def test_every_clip_gets_a_split():
    df = realistic_frame()
    splits = split_groups(df, seed=42)
    assert len(splits) == len(df)
    assert set(splits) == {"train", "validation", "test"}


def test_a_group_never_spans_two_splits():
    df = realistic_frame()
    df["split"] = split_groups(df, seed=42)
    assert (df.groupby("group")["split"].nunique() == 1).all()


def test_each_class_roughly_follows_70_15_15():
    df = realistic_frame()
    df["split"] = split_groups(df, seed=42)
    shares = pd.crosstab(df["label"], df["split"], normalize="index")
    for label in shares.index:
        assert abs(shares.loc[label, "train"] - 0.70) < 0.03, label
        assert abs(shares.loc[label, "validation"] - 0.15) < 0.03, label
        assert abs(shares.loc[label, "test"] - 0.15) < 0.03, label


def test_can_balance_by_another_column():
    # Real and synthetic knock should both follow 70/15/15, not just knock overall.
    df = realistic_frame()
    df["stratum"] = df["label"]
    real = df["group"].str.startswith("revix") & (df["label"] == "engine_knock")
    df.loc[real.to_numpy().nonzero()[0][::2], "stratum"] = "engine_knock/real"
    df["split"] = split_groups(df, seed=42, by="stratum")
    shares = pd.crosstab(df["stratum"], df["split"], normalize="index")
    for stratum in shares.index:
        assert abs(shares.loc[stratum, "train"] - 0.70) < 0.03, stratum


# Group shapes from our first real build: (healthy/hf, healthy/revix, knock/real,
# knock/revix, misfire/hf) -> how many groups look like that. Real knock clips
# share groups with Revix mixes (and often a healthy Revix clip), and few Revix
# mixes are on their own, so a split that never revisits a choice gets stuck.
REAL_GROUP_SHAPES = {
    (0, 1, 0, 0, 0): 209, (0, 0, 1, 0, 0): 113, (0, 0, 0, 0, 1): 108, (1, 0, 0, 0, 0): 80,
    (0, 0, 1, 1, 0): 43, (0, 0, 0, 1, 0): 16, (0, 1, 1, 1, 0): 14, (0, 0, 1, 2, 0): 12,
    (0, 1, 1, 2, 0): 10, (0, 1, 1, 3, 0): 8, (0, 1, 0, 1, 0): 6, (0, 0, 1, 3, 0): 5, (0, 0, 2, 2, 0): 5,
    (0, 0, 0, 2, 0): 4, (0, 1, 2, 2, 0): 3, (0, 0, 1, 4, 0): 2, (0, 0, 3, 5, 0): 2, (0, 0, 2, 5, 0): 2,
    (0, 2, 1, 3, 0): 2, (0, 1, 2, 3, 0): 2, (0, 2, 0, 2, 0): 1, (0, 3, 8, 17, 0): 1, (0, 0, 0, 7, 0): 1,
    (0, 0, 0, 4, 0): 1, (0, 1, 1, 4, 0): 1, (0, 2, 0, 4, 0): 1, (0, 1, 1, 6, 0): 1, (0, 0, 0, 3, 0): 1,
    (0, 2, 1, 2, 0): 1, (0, 1, 3, 6, 0): 1, (0, 1, 1, 0, 0): 1, (0, 2, 0, 0, 0): 1, (0, 0, 0, 5, 0): 1,
}
STRATA = ["healthy/hf", "healthy/revix", "knock/real", "knock/revix", "misfire/hf"]


def test_balances_strata_that_share_groups():
    rows = []
    for shape, n_groups in REAL_GROUP_SHAPES.items():
        for g in range(n_groups):
            group = f"{shape}#{g}"
            for stratum, count in zip(STRATA, shape):
                rows += [{"record_id": f"{group}/{stratum}/{i}", "group": group,
                          "stratum": stratum, "label": stratum} for i in range(count)]
    df = pd.DataFrame(rows)

    df["split"] = split_groups(df, seed=42, by="stratum")

    shares = pd.crosstab(df["stratum"], df["split"], normalize="index")
    for stratum in shares.index:
        for split, target in [("train", 0.70), ("validation", 0.15), ("test", 0.15)]:
            assert abs(shares.loc[stratum, split] - target) < 0.03, (stratum, split)
    assert (df.groupby("group")["split"].nunique() == 1).all()


def test_same_seed_gives_same_split():
    df = realistic_frame()
    assert split_groups(df, seed=7).equals(split_groups(df, seed=7))


def test_different_seed_gives_different_split():
    df = realistic_frame()
    assert not split_groups(df, seed=1).equals(split_groups(df, seed=2))


def test_result_lines_up_with_input_rows():
    df = realistic_frame().sample(frac=1, random_state=3)
    splits = split_groups(df, seed=42)
    assert splits.index.equals(df.index)
