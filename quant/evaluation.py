from __future__ import annotations

from typing import Dict, List, Sequence


def mse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / max(1, len(y_true))


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / max(1, len(y_true))


def accuracy(y_true: Sequence[int], y_prob: Sequence[float], threshold: float = 0.5) -> float:
    preds = [1 if p >= threshold else 0 for p in y_prob]
    correct = sum(int(a == b) for a, b in zip(y_true, preds))
    return correct / max(1, len(y_true))


def classification_metrics(y_true: Sequence[int], y_prob: Sequence[float], threshold: float = 0.5) -> Dict[str, float]:
    preds = [1 if p >= threshold else 0 for p in y_prob]
    tp = sum(1 for a, b in zip(y_true, preds) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(y_true, preds) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y_true, preds) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, preds) if a == 1 and b == 0)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2.0 * precision * recall) / max(1e-12, precision + recall)
    return {
        "accuracy": (tp + tn) / max(1, len(y_true)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def calibration_buckets(y_true: Sequence[int], y_prob: Sequence[float], bucket_size: float = 0.05) -> List[Dict[str, float]]:
    bins: Dict[tuple[float, float], List[int]] = {}
    prob_bins: Dict[tuple[float, float], List[float]] = {}
    steps = int(1.0 / bucket_size)
    for i in range(steps):
        lo = i * bucket_size
        hi = (i + 1) * bucket_size
        bins[(lo, hi)] = []
        prob_bins[(lo, hi)] = []
    for actual, p in zip(y_true, y_prob):
        idx = min(steps - 1, int(p / bucket_size))
        lo = idx * bucket_size
        hi = (idx + 1) * bucket_size
        bins[(lo, hi)].append(int(actual))
        prob_bins[(lo, hi)].append(float(p))
    result: List[Dict[str, float]] = []
    for (lo, hi), values in bins.items():
        if not values:
            continue
        probs = prob_bins[(lo, hi)]
        result.append(
            {
                "bucket_low": lo,
                "bucket_high": hi,
                "count": float(len(values)),
                "predicted_mean": sum(probs) / len(probs),
                "actual_win_rate": sum(values) / len(values),
            }
        )
    return result


def confidence_edge_analysis(y_true: Sequence[int], y_prob: Sequence[float]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for threshold in [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        preds = [(a, p) for a, p in zip(y_true, y_prob) if p > threshold]
        out[f"p_gt_{threshold:.2f}"] = {
            "threshold": float(threshold),
            "count": float(len(preds)),
            "accuracy": (sum(1 for a, _ in preds if a == 1) / len(preds)) if preds else 0.0,
        }
    return out


def error_analysis(y_test_ret: Sequence[float], up_prob: Sequence[float], ret_pred: Sequence[float], top_n: int = 5) -> Dict[str, List[Dict[str, float]]]:
    largest_errors = sorted(
        [
            {
                "index": float(i),
                "abs_error": abs(y_test_ret[i] - ret_pred[i]),
                "actual_return": y_test_ret[i],
                "pred_return": ret_pred[i],
            }
            for i in range(len(y_test_ret))
        ],
        key=lambda x: x["abs_error"],
        reverse=True,
    )[:top_n]
    high_conf_wrong = []
    for i, (ret, p) in enumerate(zip(y_test_ret, up_prob)):
        actual_up = 1 if ret > 0 else 0
        pred_up = 1 if p >= 0.5 else 0
        confidence = max(p, 1.0 - p)
        if actual_up != pred_up and confidence >= 0.7:
            high_conf_wrong.append({"index": float(i), "p_up": p, "actual_return": ret, "confidence": confidence})
    high_conf_wrong = sorted(high_conf_wrong, key=lambda x: x["confidence"], reverse=True)[:top_n]
    return {"largest_return_errors": largest_errors, "high_confidence_wrong_calls": high_conf_wrong}
