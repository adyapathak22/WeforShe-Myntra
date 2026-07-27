"""
Body clustering for MynFit.

Groups shoppers into body-type clusters using KMeans on (height_cm, weight_kg),
fit separately per gender since height/weight distributions differ meaningfully
between them and pooling would distort cluster boundaries.

Design choices, and why:
- Separate scaler + KMeans model per gender, so one gender's spread doesn't
  skew the other's cluster boundaries.
- StandardScaler before KMeans, since KMeans is distance-based and height (cm)
  and weight (kg) are on very different scales.
- Each cluster also gets a human-readable label (e.g. "Average height, Athletic
  build") derived from its centroid, computed relative to that gender's own
  height/weight tertiles. Raw labels like "Female_Cluster_2" mean nothing in a
  demo or pitch - this makes the fallback ladder's output explainable.
- A gender-agnostic "All" model is fit as a genuine safety net. If gender is
  missing/unrecognized, we fall back to this pooled model instead of returning
  a fake cluster ID with zero data behind it anywhere in the system.
- The fitted model is persisted to disk (pickle), so data_prep.py fits it once
  and recommend.py can load + predict for a live request without refitting.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

MODEL_PATH = os.path.join(os.path.dirname(__file__), "cluster_model.pkl")
GENDERS = ["Female", "Male"]
N_CLUSTERS = 5

HEIGHT_LABELS = ["Petite", "Average height", "Tall"]
WEIGHT_LABELS = ["Slim", "Athletic build", "Broad build"]


class BodyClusterer:
    def __init__(self, n_clusters: int = N_CLUSTERS):
        self.n_clusters = n_clusters
        self.scalers = {g: StandardScaler() for g in GENDERS}
        self.models = {g: KMeans(n_clusters=n_clusters, random_state=42, n_init=10) for g in GENDERS}
        # pooled fallback model, ignores gender - used only if gender is missing/unrecognized
        self.fallback_scaler = StandardScaler()
        self.fallback_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = {}  # e.g. "Female_Cluster_2" -> "Average height, Athletic build"
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        for gender in GENDERS:
            gender_df = df[df["gender"] == gender]
            if len(gender_df) == 0:
                continue
            X = gender_df[["height_cm", "weight_kg"]].values
            X_scaled = self.scalers[gender].fit_transform(X)
            X_scaled[:, 0] *= 0.3
            self.models[gender].fit(X_scaled)
            self._label_clusters(gender, gender_df)

        # pooled fallback, fit on everyone regardless of gender
        X_all = df[["height_cm", "weight_kg"]].values
        X_all_scaled = self.fallback_scaler.fit_transform(X_all)
        self.fallback_model.fit(X_all_scaled)

        self.is_fitted = True
        return self

    def _label_clusters(self, gender: str, gender_df: pd.DataFrame):
        """Builds a human-readable label for each cluster from its centroid,
        relative to that gender's own height/weight tertiles."""
        h_cuts = gender_df["height_cm"].quantile([0.33, 0.66]).values
        w_cuts = gender_df["weight_kg"].quantile([0.33, 0.66]).values

        centroids_scaled = self.models[gender].cluster_centers_
        centroids = self.scalers[gender].inverse_transform(centroids_scaled)

        for idx, (h, w) in enumerate(centroids):
            h_label = HEIGHT_LABELS[0] if h < h_cuts[0] else HEIGHT_LABELS[1] if h < h_cuts[1] else HEIGHT_LABELS[2]
            w_label = WEIGHT_LABELS[0] if w < w_cuts[0] else WEIGHT_LABELS[1] if w < w_cuts[1] else WEIGHT_LABELS[2]
            cluster_id = f"{gender}_Cluster_{idx}"
            self.cluster_labels[cluster_id] = f"{h_label}, {w_label}"

    def predict_cluster(self, gender: str, height_cm: float, weight_kg: float) -> str:
        if not self.is_fitted:
            raise ValueError("The clusterer has not been fitted yet - run fit() or load a saved model first.")

        X_new = np.array([[height_cm, weight_kg]])

        if gender in GENDERS:
            X_scaled = self.scalers[gender].transform(X_new)
            X_scaled[:, 0] *= 0.3
            idx = self.models[gender].predict(X_scaled)[0]
            return f"{gender}_Cluster_{idx}"

        # safe fallback: unrecognized/missing gender uses the pooled model,
        # which has real data behind every cluster - never returns an empty bucket
        X_scaled = self.fallback_scaler.transform(X_new)
        idx = self.fallback_model.predict(X_scaled)[0]
        return f"All_Cluster_{idx}"

    def get_cluster_label(self, cluster_id: str) -> str:
        """Human-readable description for a cluster ID, for display/explanations."""
        return self.cluster_labels.get(cluster_id, "your body type")

    def assign_clusters_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["cluster"] = None
        for gender in GENDERS:
            mask = df["gender"] == gender
            gender_df = df[mask]
            if len(gender_df) == 0:
                continue
            X = gender_df[["height_cm", "weight_kg"]].values
            X_scaled = self.scalers[gender].transform(X)
            X_scaled[:, 0] *= 0.3
            clusters = self.models[gender].predict(X_scaled)
            df.loc[mask, "cluster"] = [f"{gender}_Cluster_{c}" for c in clusters]
        return df

    def save(self, path: str = MODEL_PATH):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str = MODEL_PATH) -> "BodyClusterer":
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No saved cluster model at {path}. Run data_prep.py first to fit and save one."
            )
        with open(path, "rb") as f:
            return pickle.load(f)


# --- module-level convenience functions, so recommend.py / data_prep.py can use
# the same simple calling style as before without holding onto a class instance ---

def fit_and_save(df: pd.DataFrame, path: str = MODEL_PATH) -> BodyClusterer:
    clusterer = BodyClusterer(n_clusters=N_CLUSTERS).fit(df)
    clusterer.save(path)
    return clusterer


def assign_cluster(gender: str, height_cm: float, weight_kg: float, path: str = MODEL_PATH) -> str:
    clusterer = BodyClusterer.load(path)
    return clusterer.predict_cluster(gender, height_cm, weight_kg)


def get_cluster_label(cluster_id: str, path: str = MODEL_PATH) -> str:
    clusterer = BodyClusterer.load(path)
    return clusterer.get_cluster_label(cluster_id)


if __name__ == "__main__":
    # quick sanity check (requires cluster_model.pkl already generated by data_prep.py)
    print(assign_cluster("Female", 164, 59))
    print(get_cluster_label(assign_cluster("Female", 164, 59)))
    print(assign_cluster("Male", 180, 85))
    print(assign_cluster("Other", 170, 70))  # exercises the safe fallback path
