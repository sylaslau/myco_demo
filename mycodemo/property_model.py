"""M3 property model: recipe -> density -> stress at 10% strain.

Stress is modelled on a log10 scale, because values span 0.003-0.3 MPa.
Every model is tested by leaving one lab (dataset) out at a time.
"""

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

RECIPE_FEATURES = ["species_name", "substrate_group"]
# How the substrate was prepared: particle size class (1 dust, 2 chopped,
# 3 loose fibres or particles, 4 long tow), pressing while growing (d005) and
# compaction after growth (d023)
PROCESS_FEATURES = ["particle_size_class", "pre_compressed", "compacted_after_growth"]
YES_NO_FEATURES = ["pre_compressed", "compacted_after_growth"]


def training_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Uncoated samples with known pressing; chitosan-coated rows are kept aside as a check."""
    return table[~table["coated"] & table["pre_compressed"].notna()].copy()


def recipe_model(features: list[str] = RECIPE_FEATURES) -> object:
    """Ridge regression on one-hot recipe columns (plus 0/1 pressing columns if given)."""
    categorical = [f for f in features if f not in YES_NO_FEATURES]
    yes_no = [f for f in features if f in YES_NO_FEATURES]
    parts = [(OneHotEncoder(handle_unknown="ignore"), categorical)]
    if yes_no:
        parts.append(("passthrough", yes_no))
    return make_pipeline(make_column_transformer(*parts), Ridge(alpha=1.0))


class TwoStepModel:
    """Step 1: recipe -> log density. Step 2: log density -> log stress."""

    def fit(self, df: pd.DataFrame) -> "TwoStepModel":
        d = df[df["density_kg_m3"].notna()]
        self.density = recipe_model().fit(
            d[RECIPE_FEATURES], np.log10(d["density_kg_m3"]),
            ridge__sample_weight=d["n_specimens"],
        )
        b = df[df["density_kg_m3"].notna() & df["stress_mpa"].notna()]
        self.stress = LinearRegression().fit(
            np.log10(b[["density_kg_m3"]]), np.log10(b["stress_mpa"]),
            sample_weight=b["n_specimens"],
        )
        return self

    def predict_density(self, df: pd.DataFrame) -> np.ndarray:
        return 10 ** self.density.predict(df[RECIPE_FEATURES])

    def predict_stress(self, df: pd.DataFrame) -> np.ndarray:
        log_density = pd.DataFrame({"density_kg_m3": np.log10(self.predict_density(df))})
        return 10 ** self.stress.predict(log_density)


def _predictions(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Predict stress for one held-out lab with every model and baseline."""
    s = train[train["stress_mpa"].notna()]
    w = s["n_specimens"]
    out = pd.DataFrame(index=test.index)

    # Baseline 1: the average (geometric mean) stress of the other labs
    out["mean_baseline"] = 10 ** np.average(np.log10(s["stress_mpa"]), weights=w)

    # Recipe straight to stress, no density step
    direct = recipe_model().fit(
        s[RECIPE_FEATURES], np.log10(s["stress_mpa"]), ridge__sample_weight=w
    )
    out["recipe_direct"] = 10 ** direct.predict(test[RECIPE_FEATURES])

    # The two-step model (what the planner can use before growing)
    two = TwoStepModel().fit(train)
    out["two_step"] = two.predict_stress(test)

    # Reference: stress from the MEASURED density (only known after harvest)
    has_d = test["density_kg_m3"].notna()
    out["measured_density"] = np.nan
    if has_d.any():
        log_d = np.log10(test.loc[has_d, ["density_kg_m3"]])
        out.loc[has_d, "measured_density"] = 10 ** two.stress.predict(log_d)
    return out


def leave_one_lab_out(table: pd.DataFrame) -> pd.DataFrame:
    """Return one row per sample with the true stress and each prediction."""
    df = training_rows(table)
    rows = []
    for lab in sorted(df["dataset_id"].unique()):
        test = df[(df["dataset_id"] == lab) & df["stress_mpa"].notna()]
        train = df[df["dataset_id"] != lab]
        if test.empty:
            continue
        pred = _predictions(train, test)
        rows.append(test[["dataset_id", "sample_id", "n_specimens", "stress_mpa"]].join(pred))
    return pd.concat(rows)


def within_lab_check(table: pd.DataFrame, lab: str = "d005") -> pd.DataFrame:
    """Leave one sample out inside one lab: recipe alone vs recipe plus processing.

    Other labs keep processing constant, so the processing effect can only be
    tested inside a lab that varied it (d005: loose, chopped, tow, pre-compressed).
    """
    df = training_rows(table)
    df = df[(df["dataset_id"] == lab) & df["stress_mpa"].notna()].copy()
    df["particle_size_class"] = df["particle_size_class"].astype(str)
    y = np.log10(df["stress_mpa"])
    feature_sets = {
        "recipe_only": RECIPE_FEATURES,
        "recipe_plus_processing": RECIPE_FEATURES + PROCESS_FEATURES,
    }
    out = df[["dataset_id", "sample_id", "n_specimens", "stress_mpa"]].copy()
    out["mean_baseline"] = np.nan
    for name in feature_sets:
        out[name] = np.nan
    for i in df.index:
        train = df.drop(index=i)
        out.loc[i, "mean_baseline"] = 10 ** np.log10(train["stress_mpa"]).mean()
        for name, features in feature_sets.items():
            model = recipe_model(features).fit(train[features], y.drop(index=i))
            out.loc[i, name] = 10 ** model.predict(df.loc[[i], features])[0]
    return out


def score(pred: pd.DataFrame) -> pd.DataFrame:
    """Fold error per lab and model: 2.0 means typically off by a factor of 2."""
    models = [c for c in ["mean_baseline", "recipe_direct", "two_step", "measured_density",
                          "recipe_only", "recipe_plus_processing"] if c in pred]
    records = []
    for lab, g in pred.groupby("dataset_id"):
        for m in models:
            ok = g[m].notna()
            if not ok.any():
                continue
            err = np.abs(np.log10(g.loc[ok, m]) - np.log10(g.loc[ok, "stress_mpa"]))
            records.append({
                "held_out_lab": lab, "model": m, "n_rows": int(ok.sum()),
                "fold_error": round(float(10 ** np.average(err, weights=g.loc[ok, "n_specimens"])), 2),
            })
    return pd.DataFrame(records)
