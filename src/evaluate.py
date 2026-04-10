import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
	accuracy_score,
	roc_auc_score,
	precision_score,
	recall_score,
	f1_score,
	confusion_matrix,
)


def _load_data(path):
	p = Path(path)
	if p.suffix == ".parquet":
		return pd.read_parquet(p)
	if p.suffix in (".csv", ".txt"):
		return pd.read_csv(p)
	raise ValueError(f"Unsupported data format: {path}")


def evaluate_model(model, X, y, threshold=0.5):
	if hasattr(model, "predict_proba"):
		y_prob = model.predict_proba(X)
		if y_prob.ndim > 1 and y_prob.shape[1] > 1:
			y_prob = y_prob[:, 1]
		else:
			y_prob = y_prob.ravel()
	elif hasattr(model, "decision_function"):
		y_prob = model.decision_function(X)
		y_prob = np.array(y_prob).ravel()
	else:
		y_prob = model.predict(X)
		y_prob = np.array(y_prob).ravel()

	y_pred = (y_prob >= threshold).astype(int)

	metrics = {
		"accuracy": float(accuracy_score(y, y_pred)),
		"roc_auc": float(roc_auc_score(y, y_prob)) if len(np.unique(y)) > 1 else None,
		"precision": float(precision_score(y, y_pred)),
		"recall": float(recall_score(y, y_pred)),
		"f1": float(f1_score(y, y_pred)),
		"confusion_matrix": confusion_matrix(y, y_pred).tolist(),
		"n_samples": int(len(y)),
	}
	return metrics


def main():
	parser = argparse.ArgumentParser(description="Evaluate a saved churn model")
	parser.add_argument("--model", default="models/churn_pipeline.pkl")
	parser.add_argument("--X", default="data/processed/X_holdout.parquet")
	parser.add_argument("--y", default="data/processed/y_holdout.parquet")
	parser.add_argument("--threshold", type=float, default=None)
	parser.add_argument("--out", default="models/metrics.json")
	args = parser.parse_args()

	model_path = Path(args.model)
	if not model_path.exists():
		raise FileNotFoundError(f"Model not found: {model_path}")

	model = joblib.load(model_path)

	X = _load_data(args.X)
	y = _load_data(args.y)
	if isinstance(y, pd.DataFrame):
		y = y.squeeze()

	if args.threshold is None:
		meta_path = Path("models/threshold.json")
		if meta_path.exists():
			try:
				meta = json.loads(meta_path.read_text())
				threshold = float(meta.get("threshold", 0.5))
			except Exception:
				threshold = 0.5
		else:
			threshold = 0.5
	else:
		threshold = args.threshold

	metrics = evaluate_model(model, X, y, threshold=threshold)

	print("Evaluation results:")
	print(f"  Threshold : {threshold}")
	print(f"  Accuracy  : {metrics['accuracy']:.4f}")
	print(f"  ROC AUC   : {metrics['roc_auc']}")
	print(f"  Precision : {metrics['precision']:.4f}")
	print(f"  Recall    : {metrics['recall']:.4f}")
	print(f"  F1        : {metrics['f1']:.4f}")
	print("  Confusion:", metrics["confusion_matrix"])

	out_path = Path(args.out)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(json.dumps(metrics, indent=2))
	print(f"Saved metrics → {out_path}")


if __name__ == "__main__":
	main()

