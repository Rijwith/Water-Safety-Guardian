from pathlib import Path
import json
import pickle

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


DATA_PATH = Path("water_potability.csv")
MODEL_PATH = Path("mymodel.pkl")
OUTPUT_DIR = Path("outputs_xgb")
PLOTS_DIR = OUTPUT_DIR / "plots"
RESULTS_PATH = OUTPUT_DIR / "xgb_results.json"
XGB_THRESHOLD = 0.48


def ensure_directories() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    print("Dataset shape:", df.shape)
    print("\nMissing values:")
    print(df.isna().sum())
    return df


def save_class_balance_plot(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=df, x="Potability", hue="Potability", palette="Set2", legend=False)
    ax.set_title("Water Potability Class Balance")
    ax.set_xlabel("Potability")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "class_balance.png", dpi=300)
    plt.close()


def save_model_accuracy_plot(models: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5))
    sns.barplot(data=models, x="Accuracy_score", y="Model", palette="viridis")
    plt.title("Model Accuracy Comparison")
    plt.xlabel("Accuracy")
    plt.ylabel("Model")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_accuracy_comparison.png", dpi=300)
    plt.close()


def save_confusion_matrix_plot(cm: np.ndarray, model_name: str, file_name: str) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / file_name, dpi=300)
    plt.close()


def save_correlation_heatmap(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_correlation_heatmap.png", dpi=300)
    plt.close()


def save_roc_curve_plot(y_true: pd.Series, y_score: np.ndarray) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"XGBoost ROC-AUC = {roc_auc:.3f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("XGBoost ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "xgboost_roc_curve.png", dpi=300)
    plt.close()


def save_feature_importance_plot(model: XGBClassifier, feature_names: list[str]) -> None:
    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }
    ).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(9, 6))
    sns.barplot(data=importance_df, x="Importance", y="Feature", palette="magma")
    plt.title("XGBoost Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "xgboost_feature_importance.png", dpi=300)
    plt.close()


def classwise_mean_imputation(df: pd.DataFrame) -> pd.DataFrame:
    sulfate_0 = df[df["Potability"] == 0]["Sulfate"].mean(skipna=True)
    sulfate_1 = df[df["Potability"] == 1]["Sulfate"].mean(skipna=True)

    ph_0 = df[df["Potability"] == 0]["ph"].mean(skipna=True)
    ph_1 = df[df["Potability"] == 1]["ph"].mean(skipna=True)

    trihalo_0 = df[df["Potability"] == 0]["Trihalomethanes"].mean(skipna=True)
    trihalo_1 = df[df["Potability"] == 1]["Trihalomethanes"].mean(skipna=True)

    df.loc[(df["Potability"] == 0) & (df["Sulfate"].isna()), "Sulfate"] = sulfate_0
    df.loc[(df["Potability"] == 1) & (df["Sulfate"].isna()), "Sulfate"] = sulfate_1

    df.loc[(df["Potability"] == 0) & (df["ph"].isna()), "ph"] = ph_0
    df.loc[(df["Potability"] == 1) & (df["ph"].isna()), "ph"] = ph_1

    df.loc[(df["Potability"] == 0) & (df["Trihalomethanes"].isna()), "Trihalomethanes"] = trihalo_0
    df.loc[(df["Potability"] == 1) & (df["Trihalomethanes"].isna()), "Trihalomethanes"] = trihalo_1

    return df


def prepare_scaled_data(df: pd.DataFrame):
    X = df.drop("Potability", axis=1)
    y = df["Potability"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.33,
        random_state=42,
    )

    return X_train, X_test, y_train, y_test, scaler


def evaluate_model(
    model_name: str,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    threshold: float = 0.5,
) -> tuple[float, np.ndarray, np.ndarray]:
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
    else:
        probabilities = None
        predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    print(f"{model_name} Accuracy: {accuracy:.6f}")
    print(f"{model_name} Recall for class 1: {recall_score(y_test, predictions, zero_division=0):.6f}")
    print(f"{model_name} F1-score: {f1_score(y_test, predictions, zero_division=0):.6f}")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix:")
    print(cm)

    return accuracy, cm, probabilities


def main() -> None:
    ensure_directories()
    sns.set_theme(style="whitegrid")
    df = load_dataset()
    save_class_balance_plot(df)
    df = classwise_mean_imputation(df)
    save_correlation_heatmap(df)

    print("\nMissing values after imputation:")
    print(df.isna().sum())

    feature_names = df.drop("Potability", axis=1).columns.tolist()
    X_train, X_test, y_train, y_test, scaler = prepare_scaled_data(df)

    model_lr = LogisticRegression(max_iter=120, random_state=0, n_jobs=1)
    model_dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    model_rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=0.16, random_state=42)
    model_xgb = XGBClassifier(
        max_depth=7,
        n_estimators=500,
        random_state=0,
        learning_rate=0.02,
        min_child_weight=2,
        gamma=0,
        scale_pos_weight=1.15,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=1,
    )
    model_kn = KNeighborsClassifier(n_neighbors=9, leaf_size=20)
    model_svm = SVC(kernel="rbf", random_state=42)

    lr, _, _ = evaluate_model("Logistic Regression", model_lr, X_train, X_test, y_train, y_test)
    dt, _, _ = evaluate_model("Decision Tree", model_dt, X_train, X_test, y_train, y_test)
    rf, _, _ = evaluate_model("Random Forest", model_rf, X_train, X_test, y_train, y_test)
    xgb, xgb_cm, xgb_prob = evaluate_model(
        "XGBoost",
        model_xgb,
        X_train,
        X_test,
        y_train,
        y_test,
        threshold=XGB_THRESHOLD,
    )
    kn, _, _ = evaluate_model("KNeighbours", model_kn, X_train, X_test, y_train, y_test)
    svm, _, _ = evaluate_model("SVM", model_svm, X_train, X_test, y_train, y_test)

    models = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "XGBoost",
                "KNeighbours",
                "SVM",
            ],
            "Accuracy_score": [lr, dt, rf, xgb, kn, svm],
        }
    ).sort_values(by="Accuracy_score", ascending=False)

    print("\nModel leaderboard:")
    print(models.to_string(index=False))
    print(f"\nXGBoost threshold used: {XGB_THRESHOLD:.2f}")
    save_model_accuracy_plot(models)
    save_confusion_matrix_plot(xgb_cm, "XGBoost", "xgboost_confusion_matrix.png")
    save_roc_curve_plot(y_test, xgb_prob)
    save_feature_importance_plot(model_xgb, feature_names)

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(
            {
                "model": model_xgb,
                "scaler": scaler,
                "threshold": XGB_THRESHOLD,
                "features": feature_names,
            },
            file,
        )

    with open(MODEL_PATH, "rb") as file:
        saved_artifacts = pickle.load(file)

    loaded_model = saved_artifacts["model"]
    loaded_scaler = saved_artifacts["scaler"]
    loaded_threshold = saved_artifacts["threshold"]
    loaded_features = saved_artifacts["features"]

    xgb_pred = (loaded_model.predict_proba(X_test)[:, 1] >= loaded_threshold).astype(int)
    acc = accuracy_score(y_test, xgb_pred)
    print(f"\nReloaded XGBoost accuracy: {acc:.6f}")
    print("Reloaded feature order:")
    print(loaded_features)

    fpr, tpr, _ = roc_curve(y_test, xgb_prob)
    xgb_results = {
        "model": "XGBoost",
        "threshold": loaded_threshold,
        "accuracy": acc,
        "precision": precision_score(y_test, xgb_pred, zero_division=0),
        "recall": recall_score(y_test, xgb_pred, zero_division=0),
        "f1": f1_score(y_test, xgb_pred, zero_division=0),
        "roc_auc": auc(fpr, tpr),
        "confusion_matrix": xgb_cm.tolist(),
        "features": loaded_features,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(xgb_results, file, indent=2)

    input_data = (
        8.197353369384867,
        203.1050914346406,
        27701.794054691156,
        6.472914285587643,
        328.88683761881884,
        444.612723622325,
        14.25087508151961,
        62.90620518305302,
        3.3618333238544555,
    )

    if len(input_data) != len(loaded_features):
        raise ValueError(
            f"Expected {len(loaded_features)} features, but received {len(input_data)} values."
        )

    input_as_np = np.array(input_data, dtype=np.float64).reshape(1, -1)
    input_scaled = loaded_scaler.transform(input_as_np)
    prediction_xgb = (loaded_model.predict_proba(input_scaled)[:, 1] >= loaded_threshold).astype(int)

    print("\nSample input:")
    print(input_data)
    print("Prediction:", prediction_xgb)
    if prediction_xgb[0] == 0:
        print("The water is not potable")
    else:
        print("The water is potable")

    print(f"\nPlots saved to: {PLOTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
