from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

_console = Console()

from scipy.stats import entropy
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
import optuna
from optuna_integration import OptunaSearchCV

from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm
from IPython.display import display

from deep_gp.preprocessing_data_2 import undersample_class0

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", message="Choices for a categorical distribution")
warnings.filterwarnings("ignore", message="OptunaSearchCV is experimental")
warnings.filterwarnings("ignore", message="X does not have valid feature names")


def _tune_and_predict(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    base_model: BaseEstimator,
    param_distributions: dict,
    n_trials: int = 30,
    n_jobs: int = -1,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Tune hyperparameters with Bayesian optimization (Optuna) and predict on
    the test fold.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : np.ndarray
        Training target array.
    X_test : pd.DataFrame
        Test feature matrix.
    base_model : BaseEstimator
        Base model to tune.
    param_distributions : dict
        Optuna parameter distributions to search.
    n_trials : int
        Number of Optuna trials (default: 30).
    n_jobs : int
        Number of jobs to run in parallel (default: -1).

    Returns
    -------
    y_pred : np.ndarray
        Predicted target array.
    y_prob : np.ndarray
        Predicted probability array.
    best_params : dict
        Best parameters found.
    """
    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model_clone = clone(base_model)

    tuner = OptunaSearchCV(
        model_clone,
        param_distributions,
        n_trials=n_trials,
        cv=cv_inner,
        scoring="roc_auc",
        n_jobs=n_jobs,
        random_state=42,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        tuner.fit(X_train, y_train)

    best_model = tuner.best_estimator_
    y_prob = best_model.predict_proba(X_test)[:, 1]
    y_pred = best_model.predict(X_test)

    return y_pred, y_prob, tuner.best_params_


def _binary_entropy(y_prob):
    """
    Per-sample binary entropy from a probability array.

    Parameters
    ----------
    y_prob : np.ndarray
        Predicted probability array.

    Returns
    -------
    entropy : np.ndarray
        Per-sample binary entropy.
    """
    return entropy(np.vstack([y_prob, 1 - y_prob]), base=2, axis=0)


@dataclass(kw_only=True)
class CVEvaluator:
    """
    Nested cross-validation evaluator with per-fold undersampling, hyperparameter 
    tuning, and uncertainty estimation.

    Parameters
    ----------
    base_models: dict
        Dictionary of model names and base models.
    param_grids: dict
        Dictionary of parameter grids for each model.
    n_splits: int = 5
        Number of outer CV folds.
    n_jobs: int = -1
        Number of jobs to run in parallel.
    n_trials: int = 30
        Number of Optuna trials.
    undersample: bool = True
        Whether to use undersampling for fold-level training data.
    results_: dict
        Dictionary to store the results of the evaluation.
    label_: str
        Label for the evaluation.

    Usage
    -----
    evaluator = CVEvaluator(base_models, param_grids)
    evaluator.fit(X, y, y_isup, label="All Features")
    evaluator.plot_roc_curves()
    evaluator.plot_combined_roc()
    evaluator.summary()
    """
    
    base_models: dict
    param_grids: dict
    n_splits: int = 5
    n_jobs: int = -1
    n_trials: int = 30
    undersample: bool = True
    results_: dict = field(default_factory=dict)
    label_: str = ""

    def fit(self, X, y, y_isup, label=""):
        """
        Run nested CV for all models. Populates self.results_.

        Parameters
        ----------
        X : DataFrame
            Feature matrix.
        y : np.ndarray
            Binary target array (0/1).
        y_isup : np.ndarray
            ISUP grades array (0–5), used for fold-level undersampling
            when undersample=True.
        label : str
            Label shown in plot titles and headers.
        """
        self.label_ = label
        print(f"\n=== Feature set: {label} ===")

        skf_outer = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=42
        )

        for model_name, base_model in self.base_models.items():
            print(f"\n=== Evaluation: {model_name} ===")

            y_pred_all = np.zeros(len(y))
            y_prob_all = np.zeros(len(y))
            uncertainty_all = np.zeros(len(y))
            best_params_per_fold = []

            for train_idx, test_idx in tqdm(
                skf_outer.split(X, y),
                total=self.n_splits,
                desc=f"Outer CV ({model_name})"
            ):
                if self.undersample:
                    X_train_f, y_train_isup_f = undersample_class0(
                        X.iloc[train_idx], y_isup.iloc[train_idx]
                    )
                    y_train_f = (y_train_isup_f >= 3).astype(int)
                else:
                    X_train_f = X.iloc[train_idx]
                    y_train_f = (y_isup.iloc[train_idx] >= 3).astype(int)

                y_pred, y_prob, best_params = _tune_and_predict(
                    X_train_f, y_train_f, X.iloc[test_idx],
                    base_model, self.param_grids[model_name],
                    n_trials=self.n_trials,
                    n_jobs=self.n_jobs,
                )

                y_pred_all[test_idx] = y_pred
                y_prob_all[test_idx] = y_prob
                uncertainty_all[test_idx] = _binary_entropy(y_prob)
                best_params_per_fold.append(best_params)

            self._print_fold_results(
                model_name,
                y,
                y_pred_all,
                y_prob_all,
                uncertainty_all,
                best_params_per_fold
            )

            fpr, tpr, _ = roc_curve(y, y_prob_all)
            self.results_[model_name] = {
                "fpr": fpr,
                "tpr": tpr,
                "roc_auc": auc(fpr, tpr),
                "accuracy": accuracy_score(y, y_pred_all),
                "mean_uncertainty": uncertainty_all.mean(),
                "uncertainty_0": (
                    uncertainty_all[y_pred_all == 0].mean() 
                    if (y_pred_all == 0).any() 
                    else float("nan")
                ),
                "uncertainty_1": (
                    uncertainty_all[y_pred_all == 1].mean() 
                    if (y_pred_all == 1).any() 
                    else float("nan")
                ),
            }

        return self

    def plot_roc_curves(self):
        """Plot an individual ROC curve for each model."""
        for model_name, m in self.results_.items():
            plt.figure(figsize=(6, 4))
            plt.plot(m["fpr"], m["tpr"], lw=2, label=f"AUC = {m['roc_auc']:.3f}")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"{model_name}")
            plt.legend()
            plt.grid(True)
            plt.show()

    def plot_combined_roc(self):
        """Plot all models' ROC curves on a single figure."""
        plt.figure(figsize=(8, 6))
        for name, m in self.results_.items():
            plt.plot(
                m["fpr"], m["tpr"], lw=2, label=f'{name} (AUC = {m["roc_auc"]:.3f})'
            )
        plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("All models")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

    def summary(self):
        """Display and return a summary DataFrame of metrics across models."""
        df = pd.DataFrame([
            {
                "Model": name,
                "Accuracy": m["accuracy"],
                "ROC-AUC": m["roc_auc"],
                "Mean Uncertainty": m["mean_uncertainty"],
                "Uncertainty (Pred=0)": m["uncertainty_0"],
                "Uncertainty (Pred=1)": m["uncertainty_1"],
            }
            for name, m in self.results_.items()
        ])
        display(df)
        return df

    def _print_fold_results(
        self,
        model_name: str,
        y: np.ndarray,
        y_pred_all: np.ndarray,
        y_prob_all: np.ndarray,
        uncertainty_all: np.ndarray,
        best_params_per_fold: list[dict]
    ):
        """
        Print the results of the fold.

        Parameters
        ----------
        model_name : str
            Name of the model.
        y : np.ndarray
            Binary target array (0/1).
        y_pred_all : np.ndarray
            Predicted target array.
        y_prob_all : np.ndarray
            Predicted probability array.
        uncertainty_all : np.ndarray
            Per-sample binary entropy.
        best_params_per_fold : list[dict]
            Best parameters found per fold.
        """
        acc = accuracy_score(y, y_pred_all)
        roc = roc_auc_score(y, y_prob_all)
        unc_0 = (
            uncertainty_all[y_pred_all == 0].mean() 
            if (y_pred_all == 0).any()
            else float("nan")
        )
        unc_1 = (
            uncertainty_all[y_pred_all == 1].mean() 
            if (y_pred_all == 1).any() 
            else float("nan")
        )
        cm = confusion_matrix(y, y_pred_all)
        report = classification_report(y, y_pred_all, output_dict=True)

        # — Best hyperparameters per fold —
        params_table = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style="bold cyan"
        )
        params_table.add_column("Fold", style="dim", width=6)
        params_table.add_column("Best Parameters")
        for i, params in enumerate(best_params_per_fold):
            params_table.add_row(str(i + 1), str(params))
        _console.print(
            Panel(
                params_table,
                title="Best Hyperparameters per Fold",
                border_style="cyan"
            )
        )

        # — Metrics —
        metrics_table = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style="bold green"
        )
        metrics_table.add_column("Metric", style="bold")
        metrics_table.add_column("Value")
        metrics_table.add_row("Accuracy", f"{acc:.3f}")
        metrics_table.add_row("ROC-AUC", f"{roc:.3f}")
        metrics_table.add_row("Mean Uncertainty", f"{uncertainty_all.mean():.3f}")
        metrics_table.add_row("Uncertainty (Pred=0)", f"{unc_0:.3f}")
        metrics_table.add_row("Uncertainty (Pred=1)", f"{unc_1:.3f}")
        _console.print(Panel(metrics_table, title="Metrics", border_style="green"))

        # — Confusion matrix —
        cm_table = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style="bold yellow"
        )
        cm_table.add_column("", style="bold")
        cm_table.add_column("Pred 0", justify="right")
        cm_table.add_column("Pred 1", justify="right")
        cm_table.add_row("True 0", str(cm[0, 0]), str(cm[0, 1]))
        cm_table.add_row("True 1", str(cm[1, 0]), str(cm[1, 1]))
        _console.print(Panel(cm_table, title="Confusion Matrix", border_style="yellow"))

        # — Classification report —
        cr_table = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style="bold magenta"
        )
        cr_table.add_column("Class",     style="bold")
        cr_table.add_column("Precision", justify="right")
        cr_table.add_column("Recall",    justify="right")
        cr_table.add_column("F1-Score",  justify="right")
        cr_table.add_column("Support",   justify="right")
        for label in ["0", "1", "macro avg", "weighted avg"]:
            r = report[label]
            cr_table.add_row(
                label,
                f"{r['precision']:.3f}",
                f"{r['recall']:.3f}",
                f"{r['f1-score']:.3f}",
                str(int(r["support"])),
            )
        _console.print(
            Panel(
                cr_table,
                title=f"Classification Report ({model_name})",
                border_style="magenta"
            )
        )
