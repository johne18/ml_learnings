from enum import Enum, member
from functools import partial

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import classification_report


CLASS_WEIGHT_MODELS = {
    "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
    "LogisticRegression": LogisticRegression,
    "LinearSVC": LinearSVC,
}


def run_random_forest_model(
        x_train,
        y_train,
        x_val,
        y_val,
        total_trees: int = 100,
        depth: int = 5,
        class_weight=None,
):
    acc_scores = []

    for num_trees in range(50, total_trees + 1, 50):
        model = RandomForestClassifier(
            n_estimators=num_trees,
            max_depth=depth,
            random_state=18,
            class_weight=class_weight,
        )
        model.fit(x_train, y_train)

        predictions = model.predict(x_val)

        scores = classification_report(y_val, predictions, output_dict=True)

        acc_scores.append({
            "num_trees": num_trees,
            "classification_scores": scores
        })

    return acc_scores
    

def run_class_weight_models(
        x_train,
        y_train,
        x_val,
        y_val,
        class_weight=None,
):
    acc_scores = []

    for name, ModelClass in CLASS_WEIGHT_MODELS.items():
        model = ModelClass(class_weight=class_weight)
        model.fit(x_train, y_train)
        predictions = model.predict(x_val)

        scores = classification_report(y_val, predictions, output_dict=True)
        acc_scores.append({
            "model_name":name, 
            "classification_scores": scores
        })

    return acc_scores
