# Water Potability Prediction Project Analysis

## Project Title

Water Potability Prediction Using Artificial Neural Networks

## Project Objective

The objective of this project is to predict whether water is potable or not based on different water quality parameters.

The target column is:

```text
Potability
```

Target meaning:

- `0` means water is not potable
- `1` means water is potable

## Dataset

Dataset file:

```text
water_potability.csv
```

Dataset shape:

```text
3276 rows x 10 columns
```

The dataset contains 9 input features and 1 target column.

## Features Used

The following 9 features are used for prediction:

1. `ph`
2. `Hardness`
3. `Solids`
4. `Chloramines`
5. `Sulfate`
6. `Conductivity`
7. `Organic_carbon`
8. `Trihalomethanes`
9. `Turbidity`

The feature order is important during deployment. The model input must follow the same order used during training.

## Data Preprocessing

The dataset contains missing values in:

- `ph`
- `Sulfate`
- `Trihalomethanes`

For the ANN model:

- Missing values are handled using mean imputation.
- Features are scaled using `StandardScaler`.

Scaling is important because neural networks are sensitive to the range of input values.

## Deep Learning Model: ANN

The main deep learning model used in this project is an Artificial Neural Network built using TensorFlow/Keras.

The ANN architecture is:

```text
Dense(128, activation='relu', input_dim=9)
BatchNormalization()
Dropout(0.3)

Dense(64, activation='relu')
BatchNormalization()
Dropout(0.3)

Dense(32, activation='relu')

Dense(1, activation='sigmoid')
```

The model has:

```text
4 Dense layers total
3 hidden layers
1 output layer
```

The output layer uses sigmoid activation because this is a binary classification problem.

## ANN Training Details

The ANN model is compiled using:

```text
optimizer = adam
loss = binary_crossentropy
metric = accuracy
```

Training settings:

```text
epochs = 100
batch_size = 32
```

Prediction threshold:

```text
0.50
```

## ANN Performance

The ANN model achieved:

```text
Accuracy: 64.79%
```

Detailed ANN metrics:

| Metric | Value |
|---|---:|
| Accuracy | 0.6479 |
| Precision | 0.5598 |
| Recall | 0.4570 |
| F1-score | 0.5032 |
| ROC-AUC | 0.6510 |

ANN confusion matrix:

```text
[[308,  92],
 [139, 117]]
```

## Machine Learning Benchmark: XGBoost

XGBoost was also trained as a benchmark model.

This is useful because tabular datasets often work very well with tree-based machine learning models like XGBoost.

The XGBoost model achieved:

```text
Accuracy: 80.31%
```

Detailed XGBoost metrics:

| Metric | Value |
|---|---:|
| Accuracy | 0.8031 |
| Precision | 0.7677 |
| Recall | 0.6741 |
| F1-score | 0.7179 |
| ROC-AUC | 0.8715 |

XGBoost confusion matrix:

```text
[[598,  82],
 [131, 271]]
```

## ANN vs XGBoost Comparison

| Model | Type | Accuracy |
|---|---|---:|
| ANN | Deep Learning | 0.6479 |
| XGBoost | Machine Learning Benchmark | 0.8031 |

The ANN model is used as the main deep learning model for the course requirement.

XGBoost performed better because the dataset is tabular and relatively small. This is a common result, because tree-based models often outperform neural networks on structured tabular datasets.

## Saved Files

ANN files:

```text
outputs_ann/ann_water_potability.keras
outputs_ann/ann_artifacts.pkl
outputs_ann/ann_results.json
```

XGBoost files:

```text
mymodel.pkl
outputs_xgb/xgb_results.json
outputs_xgb/plots/
```

## Deployment Notes

For deployment, the app should:

1. Load the saved model artifact
2. Load the saved feature order
3. Collect user input for all 9 features
4. Arrange the input values in the correct feature order
5. Apply the saved preprocessing objects
6. Generate the prediction
7. Display whether the water is potable or not potable

## Final Conclusion

This project can be presented as a deep learning project because it builds and trains an Artificial Neural Network for water potability prediction.

XGBoost is included as a comparison model and performs better, with an accuracy of **80.31%**, while the ANN achieves **64.79%** accuracy.

The final explanation is that ANN satisfies the deep learning application requirement, and XGBoost demonstrates that tree-based models can be stronger for small structured tabular datasets.
