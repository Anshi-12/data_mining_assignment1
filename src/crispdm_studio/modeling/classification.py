"""Classification candidate estimators."""
from collections import OrderedDict
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from crispdm_studio.modeling.baseline import classification_baseline


def classification_estimators():
    return OrderedDict([
        ("Dummy baseline", classification_baseline()),
        ("Logistic Regression", LogisticRegression(max_iter=2000, random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced")),
        ("HistGradientBoosting", HistGradientBoostingClassifier(random_state=42)),
    ])
