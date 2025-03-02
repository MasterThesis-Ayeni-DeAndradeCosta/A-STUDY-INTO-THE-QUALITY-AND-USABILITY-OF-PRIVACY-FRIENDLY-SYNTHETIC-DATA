from abc import ABC, abstractmethod
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score, log_loss,
    cohen_kappa_score, matthews_corrcoef
)

class BaseEvaluationMetric(ABC):
    """Abstract base class for evaluation metrics."""
    
    @abstractmethod
    def compute(self, y_true, y_pred, model=None, X_test=None):
        """Computes the evaluation metric."""
        pass

# Accuracy Metric
class AccuracyMetric(BaseEvaluationMetric):
    def compute(self, y_true, y_pred, model=None, X_test=None):
        return round(accuracy_score(y_true, y_pred), 4)

# Precision Metric
class PrecisionMetric(BaseEvaluationMetric):
    def compute(self, y_true, y_pred, model=None, X_test=None):
        precision, _, _, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        return round(float(precision), 4)

# Recall Metric
class RecallMetric(BaseEvaluationMetric):
    def compute(self, y_true, y_pred, model=None, X_test=None):
        _, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        return round(float(recall), 4)

# F1 Score Metric
class F1Metric(BaseEvaluationMetric):
    def compute(self, y_true, y_pred, model=None, X_test=None):
        _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        return round(float(f1), 4)

# AUC-ROC Metric (Only for models supporting `predict_proba`)
class AUCRocMetric(BaseEvaluationMetric):
    def compute(self, y_true, y_pred, model=None, X_test=None):
        if model is None or not hasattr(model, "predict_proba"):
            return "N/A"  # Return "N/A" if the model doesn't support predict_proba

        prob_predictions = model.predict_proba(X_test)
        auc_roc = roc_auc_score(y_true, prob_predictions, multi_class="ovr") if len(set(y_true)) > 2 else roc_auc_score(y_true, prob_predictions[:, 1])
        return round(float(auc_roc), 4)

# Log Loss Metric (Only for models supporting `predict_proba`)
class LogLossMetric(BaseEvaluationMetric):
    def compute(self, y_true, y_pred, model=None, X_test=None):
        if model is None or not hasattr(model, "predict_proba"):
            return "N/A"  # Return "N/A" if the model doesn't support predict_proba

        prob_predictions = model.predict_proba(X_test)
        logloss = log_loss(y_true, prob_predictions)
        return round(float(logloss), 4)

# Cohen Kappa Metric
class CohenKappaMetric(BaseEvaluationMetric):
    def compute(self, y_true, y_pred, model=None, X_test=None):
        return round(cohen_kappa_score(y_true, y_pred), 4)

# Matthews Correlation Coefficient (MCC)
class MCCMetric(BaseEvaluationMetric):
    def compute(self, y_true, y_pred, model=None, X_test=None):
        return round(matthews_corrcoef(y_true, y_pred), 4)

