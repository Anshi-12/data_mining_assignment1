"""Regression candidate estimators."""
from collections import OrderedDict
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from crispdm_studio.modeling.baseline import regression_baseline


def regression_estimators():
    return OrderedDict([
        ("Dummy baseline", regression_baseline()),
        ("Ridge", Ridge(alpha=1.0)),
        ("Random Forest", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
        ("HistGradientBoosting", HistGradientBoostingRegressor(random_state=42)),
    ])
