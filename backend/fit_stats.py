"""
Computes fit outcome stats for MynFit, at each level of the fallback ladder
used by recommend.py:

  1. cluster + brand + category  (most specific - this exact body type, this
     exact brand, this exact item type)
  2. cluster + brand             (this body type, this brand, any item type)
  3. brand only                  (just this brand's overall behavior)
  4. category only               (true last resort, brand-agnostic)

Note: gender is already baked into `cluster` (e.g. "Female_Cluster_2" from
clustering.py), so it's not a separate grouping key here - that would be
redundant and would just re-fragment the data for no benefit.
"""

import os
import sys

import pandas as pd

# ensures clustering.py is importable regardless of the working directory
# app.py / recommend.py is launched from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clustering import get_cluster_label

KEPT_OUTCOMES = {"kept"}
# A recommendation should be backed by a meaningful number of shoppers in the
# *same size*, not merely by a large parent group.  This prevents, for example,
# three XS orders with a 100% keep rate from beating thirty M orders.
MIN_SIZE_SAMPLE = 10
FULL_CONFIDENCE_SIZE_SAMPLE = 30

DEFAULT_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "clean_fit_data.csv")


def load_clean(csv_path: str = None) -> pd.DataFrame:
    return pd.read_csv(csv_path or DEFAULT_CSV_PATH)


def _size_breakdown(group: pd.DataFrame) -> dict:
    """Dynamically computes, per size actually ordered in this group, its own
    real keep rate - e.g. {'S': {'n': 40, 'kept_rate': 0.55}, 'M': {'n': 90, 'kept_rate': 0.85}}.
    This is what lets the system pick the size with the best real fit outcome,
    not just the size most people happened to order."""
    breakdown = {}
    if "size_bought" not in group.columns or len(group) == 0:
        return breakdown
    for size, size_group in group.groupby("size_bought"):
        n_size = len(size_group)
        if n_size < MIN_SIZE_SAMPLE:
            continue
        kept_n = size_group["outcome"].isin(KEPT_OUTCOMES).sum()
        kept_rate = kept_n / n_size
        # Discount a rate until it is supported by 30 shoppers.  The score is
        # deliberately returned so the selection rule is transparent to the
        # API/UI, while kept_rate remains the shopper-facing fit statistic.
        support_weight = min(1.0, n_size / FULL_CONFIDENCE_SIZE_SAMPLE)
        breakdown[size] = {
            "n": int(n_size),
            "kept_rate": round(kept_rate, 2),
            "score": round(kept_rate * support_weight, 3),
        }
    return breakdown


def _summarize(group: pd.DataFrame, cluster: str = None) -> dict:
    n = len(group)
    kept = group["outcome"].isin(KEPT_OUTCOMES).sum()

    top_reason = None
    reasons = group.loc[group["outcome"] == "returned", "reason"].dropna()
    if not reasons.empty:
        top_reason = reasons.mode().iloc[0]

    top_brands = group["brand"].value_counts().head(2).index.tolist() if n else []

    size_breakdown = _size_breakdown(group)
    recommended_size = None

    if size_breakdown:
        ranked = sorted(
            size_breakdown,
            key=lambda s: (
                size_breakdown[s]["score"],
                size_breakdown[s]["n"],
                size_breakdown[s]["kept_rate"],
            ),
            reverse=True,
        )
        top = ranked[0]

        MIN_SCORE_MARGIN = 0.05
        if len(ranked) > 1 and (
            size_breakdown[top]["score"]
            - size_breakdown[ranked[1]]["score"]
        ) < MIN_SCORE_MARGIN:
            recommended_size = None
        else:
            recommended_size = top
    return {
        "n": n,
        "kept_rate": round(kept / n, 2) if n else 0.0,
        "recommended_size": recommended_size,
        "size_breakdown": size_breakdown,
        "common_return_reason": top_reason,
        "sample_brands": top_brands,
        "cluster_label": get_cluster_label(cluster) if cluster else None,
    }


def compute_cluster_brand_category_stats(df: pd.DataFrame, cluster: str, brand: str, category: str) -> dict:
    subset = df[(df["cluster"] == cluster) & (df["brand"] == brand) & (df["category"] == category)]
    return _summarize(subset, cluster=cluster)


def compute_cluster_brand_stats(df: pd.DataFrame, cluster: str, brand: str) -> dict:
    subset = df[(df["cluster"] == cluster) & (df["brand"] == brand)]
    return _summarize(subset, cluster=cluster)


def compute_cluster_category_stats(df: pd.DataFrame, cluster: str, category: str) -> dict:
    subset = df[(df["cluster"] == cluster) & (df["category"] == category)]
    return _summarize(subset, cluster=cluster)


def compute_cluster_stats(df: pd.DataFrame, cluster: str) -> dict:
    subset = df[df["cluster"] == cluster]
    return _summarize(subset, cluster=cluster)


def compute_brand_category_stats(df: pd.DataFrame, brand: str, category: str) -> dict:
    subset = df[(df["brand"] == brand) & (df["category"] == category)]
    return _summarize(subset)


def compute_brand_stats(df: pd.DataFrame, brand: str) -> dict:
    subset = df[df["brand"] == brand]
    return _summarize(subset)


def compute_category_stats(df: pd.DataFrame, category: str) -> dict:
    subset = df[df["category"] == category]
    return _summarize(subset)


def compute_global_stats(df: pd.DataFrame) -> dict:
    return _summarize(df)


def get_similar_reviews(
    df: pd.DataFrame,
    cluster: str,
    brand: str,
    category: str,
    size: str,
    height_cm: float,
    weight_kg: float,
    top_k: int = 5,
) -> list:
    """Returns review evidence from the same predicted body cluster and size."""
    if not size:
        return []

    subset = df[
        (df["cluster"] == cluster)
        & (df["brand"] == brand)
        & (df["category"] == category)
        & (df["size_bought"] == size)
    ].copy()

    if subset.empty:
        return []

    subset["similarity_distance"] = (
        abs(subset["height_cm"] - height_cm)
        + abs(subset["weight_kg"] - weight_kg)
    )

    subset = subset.sort_values(
        ["similarity_distance", "rating"],
        ascending=[True, False],
    )

    # height_cm/weight_kg are used above only to rank which reviewers are most
    # similar to this shopper - they are another buyer's personal data and must
    # never be exposed in the API response, so they're deliberately left out
    # of the dicts returned below.
    return [
        {
            "size_bought": row["size_bought"],
            "rating": row["rating"],
            "fit_feedback": row["fit_feedback"],
            "review": row["review"],
            "outcome": row["outcome"],
            "same_body_cluster": True,
        }
        for _, row in subset.head(top_k).iterrows()
    ]


if __name__ == "__main__":
    df = load_clean()

    sample_cluster = df["cluster"].iloc[0]
    sample_brand = df["brand"].iloc[0]
    sample_category = df["category"].iloc[0]

    print("cluster+brand+category:", compute_cluster_brand_category_stats(df, sample_cluster, sample_brand, sample_category))
    print("cluster+brand:", compute_cluster_brand_stats(df, sample_cluster, sample_brand))
    print("brand only:", compute_brand_stats(df, sample_brand))
    print("category only:", compute_category_stats(df, sample_category))
