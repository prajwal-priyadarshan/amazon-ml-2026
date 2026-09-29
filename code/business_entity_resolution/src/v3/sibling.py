"""L4 sibling graph: evidence borrowed from records that describe the same place.

The measured motivation: 1.2-1.7% of true pairs have *no* usable pair-level evidence (the
name is a random string and the address is dead or empty), and 3-5% of S2/S3 records have
no address at all. Neither is reachable by any pairwise model. They are reachable through a
mate: the noisy copies in S2/S3 mirror each other, so a record with nothing to say often
sits in a group whose other members say it clearly.

A group is "same address modulo word order" -- the exact set of normalised address tokens.
That is a conservative key on purpose. Different businesses genuinely share an address
(one of the hard-negative classes the judge is trained on), so a group is treated as
*evidence*, never as a decision: the model sees what the siblings scored and weighs it.

Groups are built inside one (country, source) block, so a record's mirror in the other
source is not its sibling here. Cross-source groups would mean holding both source shards
of a country at once, and the per-S1 cross-source aggregate they would mainly buy is
already available at the resolve stage, where both sources are in memory as lean frames.

Everything below is vectorised. A groupby-apply over millions of (S1, group) keys is not
a small constant factor slower, it is a different order of magnitude.
"""
import numpy as np
import pandas as pd


def group_ids(d, max_size=12):
    """Group id per row of a record shard; -1 for rows with no group.

    Rows with an empty address, or in an address group larger than ``max_size`` (a
    city-only address shared by thousands of records), are left ungrouped.
    """
    addr = d["addr_norm"].fillna("").astype(str).to_numpy()
    keys = pd.Series([" ".join(sorted(set(a.split()))) if a else "" for a in addr])
    counts = keys.map(keys.value_counts())
    ok = ((keys != "") & (counts >= 2) & (counts <= max_size)).to_numpy()
    gid = np.full(len(d), -1, np.int32)
    if ok.any():
        gid[ok] = pd.factorize(keys[ok])[0].astype(np.int32)
    return gid


def add_features(cand, gid, prob_col="prob1"):
    """sib_max / sib_mean / sib_cnt: the other members' scores for the *same* S1 entity."""
    n = len(cand)
    for c in ("sib_max", "sib_mean", "sib_cnt"):
        cand[c] = np.zeros(n, np.float32)
    g = gid[cand["d_pos"].to_numpy()]
    has = g >= 0
    if not has.any():
        return cand

    sub = pd.DataFrame({
        "row": np.flatnonzero(has).astype(np.int64),
        "s1_pos": cand["s1_pos"].to_numpy()[has],
        "gid": g[has],
        "p": cand[prob_col].to_numpy(np.float32)[has],
    }).sort_values(["s1_pos", "gid", "p"], ascending=[True, True, False])

    grp = sub.groupby(["s1_pos", "gid"], sort=False)
    size = grp["p"].transform("size").to_numpy(np.float32)
    total = grp["p"].transform("sum").to_numpy(np.float32)
    rank = grp.cumcount().to_numpy()
    # Rows are sorted descending inside each group, so rank 0 is the maximum and rank 1 the
    # runner-up. A row that is itself the maximum must read the runner-up, or every top row
    # would see its own score as its sibling evidence.
    first = np.where(rank == 0, sub["p"].to_numpy(np.float32), np.nan)
    second = np.where(rank == 1, sub["p"].to_numpy(np.float32), np.nan)
    gfirst = pd.Series(first).groupby([sub["s1_pos"].to_numpy(), sub["gid"].to_numpy()]).transform("max")
    gsecond = pd.Series(second).groupby([sub["s1_pos"].to_numpy(), sub["gid"].to_numpy()]).transform("max")
    gfirst = gfirst.to_numpy(np.float32)
    gsecond = np.nan_to_num(gsecond.to_numpy(np.float32), nan=0.0)

    p = sub["p"].to_numpy(np.float32)
    rows = sub["row"].to_numpy()
    cand.loc[cand.index[rows], "sib_cnt"] = size - 1.0
    cand.loc[cand.index[rows], "sib_mean"] = np.where(size > 1, (total - p) / np.maximum(size - 1.0, 1.0), 0.0)
    cand.loc[cand.index[rows], "sib_max"] = np.where(rank == 0, gsecond, gfirst)
    for c in ("sib_max", "sib_mean", "sib_cnt"):
        cand[c] = cand[c].astype(np.float32)
    return cand


def expand(cand, gid, prob_col="prob1", thr=0.9, per_s1=6):
    """View F: a confident hit drags its group mates in as candidates.

    This targets exactly the records whose own text is too poor for any view to retrieve.
    Capping at ``per_s1`` keeps a shared office block from becoming a candidate for
    everyone in it.
    """
    empty = pd.DataFrame({"s1_pos": np.array([], np.int32), "d_pos": np.array([], np.int32)})
    strong = cand.loc[cand[prob_col].to_numpy() >= thr, ["s1_pos", "d_pos"]]
    if strong.empty:
        return empty
    sg = gid[strong["d_pos"].to_numpy()]
    strong = strong.loc[sg >= 0].copy()
    if strong.empty:
        return empty
    strong["gid"] = gid[strong["d_pos"].to_numpy()]

    members = pd.DataFrame({"gid": gid, "d_pos": np.arange(len(gid), dtype=np.int32)})
    members = members[members["gid"] >= 0]
    new = strong[["s1_pos", "gid"]].drop_duplicates().merge(members, on="gid")[["s1_pos", "d_pos"]]

    # Anti-join against what is already a candidate (a merge, not a Python set lookup).
    have = cand[["s1_pos", "d_pos"]].drop_duplicates()
    have["_seen"] = np.int8(1)
    new = new.merge(have, on=["s1_pos", "d_pos"], how="left")
    new = new[new["_seen"].isna()].drop(columns=["_seen"])
    if new.empty:
        return empty
    new = new.astype({"s1_pos": np.int32, "d_pos": np.int32})
    return new.groupby("s1_pos", sort=False).head(per_s1).reset_index(drop=True)
