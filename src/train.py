import os
from time import time

import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


CATEGORIES = [
    "alt.atheism",
    "talk.religion.misc",
    "comp.graphics",
    "sci.space",
]


def build_pipeline():
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("clf", LogisticRegression(max_iter=5000, random_state=42)),
        ]
    )


def get_param_grid():
    return [
        {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__max_df": [0.5, 0.75, 0.85],
            "tfidf__min_df": [1, 2],
            "tfidf__sublinear_tf": [True, False],
            "clf": [
                LogisticRegression(max_iter=5000, solver="liblinear", random_state=42)
            ],
            "clf__C": [0.5, 1.0, 2.0],
        },
        {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__max_df": [0.5, 0.75, 0.85],
            "tfidf__min_df": [1, 2],
            "tfidf__sublinear_tf": [True, False],
            "clf": [LinearSVC(random_state=42)],
            "clf__C": [0.1, 1.0, 2.0],
        },
    ]


def main():
    train_data = fetch_20newsgroups(
        subset="train",
        categories=CATEGORIES,
        shuffle=True,
        random_state=42,
        remove=("headers", "footers", "quotes"),
    )

    test_data = fetch_20newsgroups(
        subset="test",
        categories=CATEGORIES,
        shuffle=True,
        random_state=42,
        remove=("headers", "footers", "quotes"),
    )

    pipeline = build_pipeline()
    param_grid = get_param_grid()

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1,
        scoring="accuracy",
    )

    print("Starting grid search for best model and vectorizer settings...")
    t0 = time()
    grid_search.fit(train_data.data, train_data.target)
    print(f"Grid search done in {time() - t0:.3f}s")

    best_model = grid_search.best_estimator_
    print("Best parameters:", grid_search.best_params_)

    t0 = time()
    y_pred = best_model.predict(test_data.data)
    print(f"Prediction done in {time() - t0:.3f}s")

    accuracy = accuracy_score(test_data.target, y_pred)
    print("Accuracy:", accuracy)
    print(
        classification_report(
            test_data.target,
            y_pred,
            target_names=train_data.target_names,
        )
    )

    os.makedirs("outputs", exist_ok=True)

    ConfusionMatrixDisplay.from_predictions(
        test_data.target,
        y_pred,
        display_labels=train_data.target_names,
        xticks_rotation=45,
        cmap=plt.cm.Blues,
    )
    plt.title("20 Newsgroups (4-class) Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", bbox_inches="tight")
    plt.close()

    joblib.dump(best_model, "outputs/newsgroups_text_model.joblib")
    print("Saved confusion matrix to outputs/confusion_matrix.png")
    print("Saved model to outputs/newsgroups_text_model.joblib")


if __name__ == "__main__":
    main()


