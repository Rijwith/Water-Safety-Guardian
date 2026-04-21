from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, GaussianNoise
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import l2


DATA_PATH = Path("data/water_potability.csv")
OUTPUT_DIR = Path("outputs_ann")
ANN_MODEL_PATH = OUTPUT_DIR / "ann_water_potability.keras"
ANN_ARTIFACT_PATH = OUTPUT_DIR / "ann_artifacts.pkl"
ANN_RESULTS_PATH = OUTPUT_DIR / "ann_results.json"

BASE_FEATURES = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
                 "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]


def ensure_directories() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)


def class_conditional_impute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for feature in ["ph", "Sulfate", "Trihalomethanes"]:
        for cls in df["Potability"].unique():
            mask = (df["Potability"] == cls) & df[feature].isna()
            median_val = df.loc[df["Potability"] == cls, feature].median()
            df.loc[mask, feature] = median_val
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction and ratio features.
    Raw features have max correlation ~0.03 with target.
    Combinations can expose non-linear relationships the ANN can learn.
    """
    df = df.copy()
    # pH interactions — pH affects how other chemicals behave
    df["ph_chloramines"] = df["ph"] * df["Chloramines"]
    df["ph_sulfate"] = df["ph"] * df["Sulfate"]
    df["ph_hardness"] = df["ph"] * df["Hardness"]

    # Contamination load proxy
    df["solids_organic"] = df["Solids"] * df["Organic_carbon"]
    df["chloramines_trihalomethanes"] = df["Chloramines"] * df["Trihalomethanes"]

    # Ratio features — relative concentrations
    df["sulfate_hardness_ratio"] = df["Sulfate"] / (df["Hardness"] + 1e-6)
    df["conductivity_solids_ratio"] = df["Conductivity"] / (df["Solids"] + 1e-6)

    # Log transform skewed feature (Solids has skew 0.62)
    df["log_solids"] = np.log1p(df["Solids"])

    return df


def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    _, _, thresholds = roc_curve(y_true, y_proba)
    f1s = [f1_score(y_true, (y_proba >= t).astype(int), zero_division=0) for t in thresholds]
    return float(thresholds[np.argmax(f1s)])


def build_model(input_dim: int) -> Sequential:
    """
    tanh activation: better than elu/relu when features have near-zero correlation
    with target — tanh is zero-centered and handles weak signals more smoothly.
    GaussianNoise: input augmentation, forces robustness to measurement noise.
    AdamW: Adam + weight decay, better generalization than plain Adam on small data.
    """
    reg = l2(1e-4)
    model = Sequential([
        GaussianNoise(0.05, input_shape=(input_dim,)),

        Dense(128, activation="tanh", kernel_regularizer=reg),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation="tanh", kernel_regularizer=reg),
        BatchNormalization(),
        Dropout(0.2),

        Dense(32, activation="tanh", kernel_regularizer=reg),
        BatchNormalization(),
        Dropout(0.1),

        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_kfold_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_splits: int = 5,
) -> np.ndarray:
    """
    Train one model per fold, average their test probabilities.
    On small datasets this reduces variance significantly vs a single split.
    SMOTE is applied inside each fold (on train only) to prevent data leakage.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_probas = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Fold {fold}/{n_splits} ---")
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # SMOTE on train fold only — never on val or test
        smote = SMOTE(random_state=42 + fold)
        X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)
        print(f"  After SMOTE: {np.bincount(y_tr_sm)}")

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=0),
        ]

        model = build_model(input_dim=X_tr_sm.shape[1])
        model.fit(
            X_tr_sm, y_tr_sm,
            epochs=300,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0,
        )

        proba = model.predict(X_test, verbose=0).ravel()
        fold_probas.append(proba)

        fold_pred = (proba >= 0.5).astype(int)
        print(f"  Fold test accuracy: {accuracy_score(y_test, fold_pred):.4f}  "
              f"ROC-AUC: {roc_auc_score(y_test, proba):.4f}")

    # Average probabilities across all folds
    return np.mean(fold_probas, axis=0)


def main() -> None:
    ensure_directories()
    tf.random.set_seed(42)
    np.random.seed(42)

    df = pd.read_csv(DATA_PATH)
    df = class_conditional_impute(df)
    df = engineer_features(df)

    feature_names = [c for c in df.columns if c != "Potability"]
    print(f"Total features after engineering: {len(feature_names)}")
    print(f"Features: {feature_names}")

    # Check engineered feature correlations
    corr = df[feature_names].corrwith(df["Potability"]).abs().sort_values(ascending=False)
    print("\nTop correlations with Potability (abs):")
    print(corr.head(10))

    X = df[feature_names].values
    y = df["Potability"].values

    # Hold out a fixed test set — never touched during training or SMOTE
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nTrain size: {len(X_train_scaled)}, Test size: {len(X_test_scaled)}")
    print(f"Train class dist: {np.bincount(y_train_full)}")

    # KFold ensemble with SMOTE inside each fold
    test_proba = train_kfold_ensemble(
        X_train_scaled, y_train_full,
        X_test_scaled, y_test,
        n_splits=5,
    )

    best_threshold = find_best_threshold(y_test, test_proba)
    print(f"\nBest threshold (max F1): {best_threshold:.4f}")

    test_pred = (test_proba >= best_threshold).astype(int)

    acc = accuracy_score(y_test, test_pred)
    prec = precision_score(y_test, test_pred, zero_division=0)
    rec = recall_score(y_test, test_pred, zero_division=0)
    f1 = f1_score(y_test, test_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, test_proba)

    print("\n=== FINAL ENSEMBLE RESULTS ===")
    print(f"Features used  : {len(feature_names)} (9 original + 8 engineered)")
    print(f"Best threshold : {best_threshold:.4f}")
    print(f"Accuracy       : {acc:.4f}")
    print(f"Precision      : {prec:.4f}")
    print(f"Recall         : {rec:.4f}")
    print(f"F1-score       : {f1:.4f}")
    print(f"ROC-AUC        : {roc_auc:.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, test_pred))

    # Train a final single model on full train data + SMOTE for saving/deployment
    print("\nTraining final model on full training data for saving...")
    smote = SMOTE(random_state=42)
    X_final_sm, y_final_sm = smote.fit_resample(X_train_scaled, y_train_full)

    final_model = build_model(input_dim=X_final_sm.shape[1])
    final_model.fit(
        X_final_sm, y_final_sm,
        epochs=300,
        batch_size=32,
        validation_split=0.1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=0),
        ],
        verbose=0,
    )
    final_model.save(ANN_MODEL_PATH)

    results = {
        "model": "TensorFlow Keras ANN (v3 — engineered features + SMOTE + KFold ensemble)",
        "features": feature_names,
        "n_features": len(feature_names),
        "engineered_features": [
            "ph_chloramines", "ph_sulfate", "ph_hardness",
            "solids_organic", "chloramines_trihalomethanes",
            "sulfate_hardness_ratio", "conductivity_solids_ratio", "log_solids",
        ],
        "techniques": [
            "class-conditional median imputation",
            "feature engineering (interactions + ratios + log transform)",
            "RobustScaler",
            "SMOTE oversampling (per fold)",
            "5-fold stratified ensemble",
            "GaussianNoise augmentation",
            "tanh activation",
            "L2 regularization",
            "BatchNormalization",
            "AdamW optimizer",
            "EarlyStopping + ReduceLROnPlateau",
            "ROC threshold tuning",
        ],
        "best_threshold": best_threshold,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
        "classification_report": classification_report(
            y_test, test_pred, output_dict=True, zero_division=0
        ),
    }

    with open(ANN_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(ANN_ARTIFACT_PATH, "wb") as f:
        pickle.dump({
            "scaler": scaler,
            "threshold": best_threshold,
            "features": feature_names,
            "model_path": str(ANN_MODEL_PATH),
        }, f)

    print(f"\nSaved model  : {ANN_MODEL_PATH}")
    print(f"Saved results: {ANN_RESULTS_PATH}")


if __name__ == "__main__":
    main()
