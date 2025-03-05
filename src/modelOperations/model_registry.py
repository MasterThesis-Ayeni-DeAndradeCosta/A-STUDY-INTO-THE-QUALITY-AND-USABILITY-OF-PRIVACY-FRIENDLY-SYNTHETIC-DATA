from .models import (
    LogisticRegressionModel, KNNModel, RandomForestModel,
    DecisionTreeModel, SVMModel, XGBoostModel
)


MODEL_REGISTRY = {
    "LogisticRegression": LogisticRegressionModel,
    "KNN": KNNModel,
    "RandomForest": RandomForestModel,
    "DecisionTree": DecisionTreeModel,
    "SVM": SVMModel,
    "XGBoost": XGBoostModel,  # New model
}


