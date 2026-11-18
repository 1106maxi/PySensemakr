"""
Computes eta2 for the cluster adjustment of the sensitivity analysis, plus helper functions.

When the treatment is assigned at the level of clusters, an omitted variable can
only confound through its variation between clusters, because its variation within clusters is uncorrelated with
the treatment. Such a confounder can explain at most the share eta2 of the residual variance of the outcome, which
is the share that the cluster indicators can explain. If `cluster` or `eta2` is passed to a function of the
package, partial R2 values with the outcome refer to the between-cluster variation in the outcome instead of all
of its variation. Partial R2 values with the treatment stay the same. The adjustment only
concerns the point estimate, so the functions for standard errors, t-values and confidence intervals do not take
these arguments.

Reference:
------------
Strezhnev, A. (2026), "Omitted variable bias sensitivity analysis with clustered treatment assignment."

Example:
------------
See specific functions below.

Functions
------------
"""

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from statsmodels.regression.linear_model import RegressionResultsWrapper


def cluster_partial_r2(model: RegressionResultsWrapper, cluster: str) -> float:
    """
    Compute eta2, the partial R2 of the cluster indicators with the outcome.

    eta2 is the share of the residual variance of the outcome that the cluster indicators can explain,
    after partialing out the treatment and the covariates. When the treatment is assigned at the cluster level,
    it is the largest partial R2 with the outcome that an omitted confounder can have. See details in
    Strezhnev (2026).

    Parameters
    ----------
    model : statsmodels OLSResults object
        a statsmodels OLSResults object containing the restricted regression.
    cluster : string
        a string with the name of the column of the model's data that identifies the clusters.

    Returns
    -------
    float
        a float with eta2, greater than 0 and at most 1.
    """
    if np.any(np.asarray(model.model.weights) != 1):
        raise ValueError("The cluster adjustment requires an unweighted OLS model. Weighted models are not supported yet.")
    cluster_vec = get_cluster(model, cluster)
    y = np.asarray(model.model.endog, dtype=float)
    x = np.asarray(model.model.exog, dtype=float)
    # Partial out the cluster indicators
    y_within = y - cluster_mean(y, cluster_vec)
    x_within = x - cluster_mean(x, cluster_vec)
    coef = np.linalg.lstsq(x_within, y_within, rcond=None)[0]
    ssr_with_indicators = np.sum((y_within - x_within @ coef) ** 2)
    ssr_without_indicators = np.sum(np.asarray(model.resid, dtype=float) ** 2)
    return float(1 - ssr_with_indicators / ssr_without_indicators)


def resolve_eta2(
    model: RegressionResultsWrapper | None = None,
    cluster: str | None = None,
    eta2: float | None = None,
    variables: str | list[str] | None = None,
) -> float | None:
    """Check the arguments and return eta2 from cluster or eta2, or None if neither is given.

    With cluster, the named variables must be constant within clusters.
    """
    if cluster is not None and eta2 is not None:
        raise TypeError("Provide either cluster or eta2, not both.")
    if eta2 is not None:
        check_eta2(eta2)
        return eta2
    if cluster is None:
        return None
    if model is None:
        raise TypeError("cluster requires a statsmodels OLSResults object. With summary statistics, pass eta2 instead.")
    if variables is not None:
        check_cluster_level(model, variables, cluster)
    return cluster_partial_r2(model, cluster)


def get_cluster(model: RegressionResultsWrapper, cluster: str) -> np.ndarray:
    """Cluster labels of the observations used in the model, as a numpy array."""
    if not isinstance(cluster, str):
        raise TypeError(f"cluster must be the name of a column of the model's data. cluster was: {cluster!r}")
    data = model.model.data
    if cluster not in data.frame.columns:
        raise ValueError(f"Cluster column {cluster!r} not found in the model's data.")
    # keep only the rows the model used, so that rows dropped for missing values are dropped here too
    cluster_vec = data.frame[cluster].loc[data.row_labels].to_numpy()
    if cluster_vec.shape != (int(model.nobs),):
        raise ValueError("Could not align the cluster labels with the model. The index of the model's data must be unique.")
    if pd.isnull(cluster_vec).any():
        raise ValueError("Cluster labels are missing for some observations used in the model.")
    return cluster_vec


def cluster_mean(values: ArrayLike, cluster_vec: np.ndarray) -> np.ndarray:
    """Cluster means of values (one column or several), expanded back to the rows."""
    values = np.asarray(values, dtype=float)
    matrix = values.reshape(len(cluster_vec), -1)
    means = pd.DataFrame(matrix).groupby(cluster_vec).transform("mean").to_numpy()
    return means.reshape(values.shape)


def check_eta2(eta2: Any) -> None:
    """Ensure that eta2 is a single number greater than 0 and at most 1."""
    if isinstance(eta2, bool) or not isinstance(eta2, (int, float, np.integer, np.floating)):
        raise TypeError(f"eta2 must be a single number. eta2 was: {eta2!r}")
    if not 0 < eta2 <= 1:
        raise ValueError(f"eta2 must be greater than 0 and at most 1. eta2 was: {eta2}")


def check_cluster_level(model: RegressionResultsWrapper, variables: str | list[str], cluster: str) -> None:
    """Raise an error if one of the named variables of the model takes more than one value within a cluster."""
    cluster_vec = get_cluster(model, cluster)
    names = list(model.model.exog_names)
    for name in [variables] if isinstance(variables, str) else variables:
        if name not in names:
            raise ValueError(f"Variable {name!r} not found in the model.")
        values = pd.Series(model.model.exog[:, names.index(name)])
        if (values.groupby(cluster_vec).nunique() > 1).any():
            raise ValueError(
                f"Variable {name!r} varies within clusters. The cluster adjustment applies only to variables "
                "that are constant within clusters, such as a treatment assigned at the cluster level."
            )
