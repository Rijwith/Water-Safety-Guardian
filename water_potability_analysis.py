from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


DATA_PATH = Path("water_potability.csv")
OUTPUT_DIR = Path("outputs")
PLOTS_DIR = OUTPUT_DIR / "plots"


def optimize_threshold(y_true: pd.Series, y_proba: pd.Series) -> tuple[float, float]:
    thresholds = [step / 100 for step in range(30, 71)]
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold, best_f1


def ensure_directories() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    print("Step 1: Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")
    print("\nMissing values per column:")
    print(df.isna().sum())
    return df


def save_class_balance_plot(df: pd.DataFrame) -> None:
    print("Step 2: Creating class balance plot...")
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=df, x="Potability", hue="Potability", palette="Set2", legend=False)
    ax.set_title("Potability Class Balance")
    ax.set_xlabel("Potability Class")
    ax.set_ylabel("Count")

    total = len(df)
    for patch in ax.patches:
        count = int(patch.get_height())
        pct = (count / total) * 100
        ax.annotate(
            f"{count}\n({pct:.1f}%)",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "potability_class_balance.png", dpi=300)
    plt.close()


def save_feature_distribution_plot(df: pd.DataFrame, feature_columns: list[str]) -> None:
    print("Step 3: Creating feature distribution plots...")
    n_cols = 3
    n_rows = (len(feature_columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    axes = axes.flatten()

    for idx, column in enumerate(feature_columns):
        sns.histplot(
            data=df,
            x=column,
            hue="Potability",
            kde=True,
            bins=30,
            element="step",
            stat="density",
            common_norm=False,
            palette="Set1",
            ax=axes[idx],
        )
        axes[idx].set_title(f"Distribution of {column}")

    for idx in range(len(feature_columns), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_distributions.png", dpi=300)
    plt.close()


def save_correlation_heatmap(df: pd.DataFrame) -> None:
    print("Step 4: Creating correlation heatmap...")
    plt.figure(figsize=(11, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_correlation_heatmap.png", dpi=300)
    plt.close()


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("num", numeric_pipeline, feature_columns)],
        remainder="drop",
    )


def train_models(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    print("Step 5: Training machine learning models...")
    feature_columns = [column for column in df.columns if column != "Potability"]
    target_column = "Potability"

    X = df[feature_columns]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    preprocessor = build_preprocessor(feature_columns)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "decision_tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=1
        ),
        "svm": SVC(kernel="rbf", probability=True, random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=7),
    }

    results = []
    detailed_reports = {}

    for model_name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "model": model_name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        results.append(metrics)

        detailed_reports[model_name] = {
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

    ann_search = RandomizedSearchCV(
        estimator=Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    MLPClassifier(
                        early_stopping=True,
                        max_iter=800,
                        n_iter_no_change=30,
                        random_state=42,
                    ),
                ),
            ]
        ),
        param_distributions={
            "model__hidden_layer_sizes": [
                (64, 32),
                (96, 48),
                (128, 64),
                (128, 64, 32),
                (256, 128, 64),
            ],
            "model__activation": ["relu", "tanh"],
            "model__solver": ["adam"],
            "model__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
            "model__learning_rate_init": [0.0003, 0.0005, 0.001, 0.003],
            "model__batch_size": [32, 64, 128],
        },
        n_iter=16,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=42),
        n_jobs=1,
        random_state=42,
        refit=True,
    )

    print("Tuning ANN with randomized search...")
    ann_search.fit(X_train, y_train)
    ann_pipeline = ann_search.best_estimator_

    ann_train_proba = ann_pipeline.predict_proba(X_train)[:, 1]
    ann_threshold, ann_train_f1 = optimize_threshold(y_train, ann_train_proba)

    ann_test_proba = ann_pipeline.predict_proba(X_test)[:, 1]
    ann_test_pred = (ann_test_proba >= ann_threshold).astype(int)
    ann_model = ann_pipeline.named_steps["model"]

    ann_metrics = {
        "model": "ann_tuned",
        "accuracy": accuracy_score(y_test, ann_test_pred),
        "precision": precision_score(y_test, ann_test_pred, zero_division=0),
        "recall": recall_score(y_test, ann_test_pred, zero_division=0),
        "f1_score": f1_score(y_test, ann_test_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, ann_test_proba),
    }
    results.append(ann_metrics)

    detailed_reports["ann_tuned"] = {
        "best_params": ann_search.best_params_,
        "best_cv_score": ann_search.best_score_,
        "decision_threshold": ann_threshold,
        "train_f1_at_threshold": ann_train_f1,
        "classification_report": classification_report(
            y_test, ann_test_pred, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, ann_test_pred).tolist(),
    }
    save_training_loss_plot(ann_model.loss_curve_, "tuned_ann_loss_trajectory.png", "Training Loss Trajectory for Tuned ANN")

    results_df = pd.DataFrame(results).sort_values(
        by=["f1_score", "roc_auc", "accuracy"], ascending=False
    )
    results_df.to_csv(OUTPUT_DIR / "model_results.csv", index=False)

    with open(OUTPUT_DIR / "detailed_model_reports.json", "w", encoding="utf-8") as file:
        json.dump(detailed_reports, file, indent=2)

    return results_df, detailed_reports


def save_training_loss_plot(
    loss_curve: list[float],
    file_name: str = "training_loss_trajectory.png",
    title: str = "Training Loss Trajectory for MLP Classifier",
) -> None:
    print("Step 6: Creating training loss trajectory plot...")
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_curve) + 1), loss_curve, marker="o", linewidth=2)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / file_name, dpi=300)
    plt.close()


def save_summary(df: pd.DataFrame, results_df: pd.DataFrame) -> None:
    print("Step 7: Writing summary file...")
    class_counts = df["Potability"].value_counts().sort_index()
    missing_values = df.isna().sum().sort_values(ascending=False)

    summary_lines = [
        "Water Potability Analysis Summary",
        "=" * 34,
        "",
        f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns",
        "",
        "Class balance:",
        class_counts.to_string(),
        "",
        "Missing values:",
        missing_values.to_string(),
        "",
        "Model leaderboard:",
        results_df.to_string(index=False),
        "",
        "Saved plots:",
        "- outputs/plots/potability_class_balance.png",
        "- outputs/plots/feature_distributions.png",
        "- outputs/plots/feature_correlation_heatmap.png",
        "- outputs/plots/tuned_ann_loss_trajectory.png",
        "",
        "Saved model artifacts:",
        "- outputs/model_results.csv",
        "- outputs/detailed_model_reports.json",
    ]

    (OUTPUT_DIR / "analysis_summary.txt").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )


def main() -> None:
    sns.set_theme(style="whitegrid")
    ensure_directories()
    df = load_dataset()
    feature_columns = [column for column in df.columns if column != "Potability"]

    save_class_balance_plot(df)
    save_feature_distribution_plot(df, feature_columns)
    save_correlation_heatmap(df)
    results_df, _ = train_models(df)
    save_summary(df, results_df)

    print("\nAnalysis complete.")
    print(f"Plots saved to: {PLOTS_DIR.resolve()}")
    print(f"Model results saved to: {(OUTPUT_DIR / 'model_results.csv').resolve()}")


if __name__ == "__main__":
    main()
