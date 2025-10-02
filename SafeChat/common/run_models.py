from typing import Optional, Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from sklearn.metrics import classification_report, roc_auc_score


CLASS_WEIGHT_MODELS = {
    "gradient_classifier": HistGradientBoostingClassifier,
    "logistic_regression": LogisticRegression,
    "linear_svc": LinearSVC,
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

        scores = classification_report(
            y_val, 
            predictions, 
            output_dict=True,
            labels=[0, 1],
            target_names=["non_toxic", "toxic"],
        )
        rocauc_score = roc_auc_score(
            y_val,
            predictions,
        )

        acc_scores.append({
            "num_trees": num_trees,
            "classification_scores": scores,
            "roc_auc_score": rocauc_score,
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
        model = ModelClass(class_weight=class_weight, random_state=18)
        model.fit(x_train, y_train)
        predictions = model.predict(x_val)

        scores = classification_report(
            y_val, 
            predictions, 
            output_dict=True,
            labels=[0, 1],
            target_names=["non_toxic", "toxic"],
        )
        rocauc_score = roc_auc_score(
            y_val,
            predictions,
        )

        acc_scores.append({
            "model_name":name, 
            "classification_scores": scores,
            "roc_auc_score": rocauc_score,
        })

    return acc_scores


def get_pipelines(model_name, sampling_strategy=1.0, model_params: Optional[Dict[str, Any]] = None):
    vectorizer = TfidfVectorizer(
            ngram_range=(2,3),
            max_features=5000,
            max_df=0.7,
            stop_words='english'
        )
    steps = [
        ("tfidf", vectorizer),
        ("under_sampler", RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=18)),
    ]

    if model_name not in CLASS_WEIGHT_MODELS:
        steps.append(("rf_model", RandomForestClassifier(**model_params)))
        return Pipeline(steps)
    else:
        steps.append(("toarray", FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)))

        model_class = CLASS_WEIGHT_MODELS[model_name]
        model = model_class(**model_params)

        steps.append((model_name, model))

        return Pipeline(steps)
