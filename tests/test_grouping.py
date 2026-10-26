import pandas as pd

from kwerimoto_data.grouping import assign_groups, video_id


def frame(rows):
    return pd.DataFrame(rows, columns=["record_id", "source_dataset", "source_recording", "knock_source"])


def test_video_id_drops_audioset_start_time():
    assert video_id("DqOGNij7qIE_30000.wav") == "DqOGNij7qIE"
    assert video_id("-B1uzsLG0Dk_30000") == "-B1uzsLG0Dk"


def test_unrelated_clips_get_their_own_groups():
    df = frame([
        ("a", "revix", "AAA_0.wav", ""),
        ("b", "revix", "BBB_0.wav", ""),
    ])
    groups = assign_groups(df)
    assert groups.nunique() == 2


def test_clean_clip_and_knock_mix_of_same_recording_share_a_group():
    df = frame([
        ("clean", "revix", "AAA_30000.wav", ""),
        ("knock", "revix", "AAA_30000.wav", "KKK_1000.wav"),
    ])
    groups = assign_groups(df)
    assert groups.nunique() == 1


def test_knock_mixes_reusing_one_knock_sound_share_a_group():
    df = frame([
        ("k1", "revix", "AAA_0.wav", "KKK_0.wav"),
        ("k2", "revix", "BBB_0.wav", "KKK_0.wav"),
    ])
    assert assign_groups(df).nunique() == 1


def test_links_chain_together():
    # k1 shares a knock sound with k2; k2 shares its car recording with clean.
    df = frame([
        ("k1", "revix", "AAA_0.wav", "KKK_0.wav"),
        ("k2", "revix", "BBB_0.wav", "KKK_0.wav"),
        ("clean", "revix", "BBB_0.wav", ""),
        ("alone", "revix", "CCC_0.wav", ""),
    ])
    groups = assign_groups(df).set_axis(df["record_id"])
    assert groups["k1"] == groups["k2"] == groups["clean"]
    assert groups["alone"] != groups["k1"]


def test_different_pieces_of_one_video_share_a_group():
    # A video used once as a car recording and once as a knock sound.
    df = frame([
        ("clean", "revix", "VVV_10000.wav", ""),
        ("knock", "revix", "AAA_0.wav", "VVV_40000.wav"),
    ])
    assert assign_groups(df).nunique() == 1


def test_same_name_in_different_datasets_is_not_linked():
    df = frame([
        ("r", "revix", "healthy_001", ""),
        ("h", "huggingface", "healthy_001", ""),
    ])
    assert assign_groups(df).nunique() == 2


def test_real_audioset_clip_joins_the_revix_mixes_that_used_its_video():
    # Revix mixed a piece of video KKK onto a car; we also use KKK as a real knock clip.
    df = frame([
        ("mix", "revix", "AAA_0.wav", "KKK_27000.wav"),
        ("real", "audioset", "KKK_27000.wav", ""),
        ("other_real", "audioset", "ZZZ_0.wav", ""),
    ])
    groups = assign_groups(df).set_axis(df["record_id"])
    assert groups["mix"] == groups["real"]
    assert groups["other_real"] != groups["real"]


def test_group_names_are_readable_and_stable():
    df = frame([
        ("k", "revix", "BBB_0.wav", "AAA_5000.wav"),
        ("h", "huggingface", "misfire_007", ""),
    ])
    groups = assign_groups(df).set_axis(df["record_id"])
    assert groups["k"] == "youtube:AAA"
    assert groups["h"] == "huggingface:misfire_007"
