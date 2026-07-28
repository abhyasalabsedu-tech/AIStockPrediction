"""
Run once per day after market close: python -m app.core.evaluation
Scores every Prediction older than 5 sessions against the ACTUAL realized return,
computes accuracy/precision/recall/F1, writes an EvaluationRun row.
"""
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from app.models.schema import SessionLocal, Prediction, EvaluationRun
from app.services.market_data import fetch_ohlcv

LABELS = ["SELL", "HOLD", "BUY"]


def label_outcome(pct_return: float) -> str:
    if pct_return > 0.5:
        return "BUY"
    if pct_return < -0.5:
        return "SELL"
    return "HOLD"


def run_evaluation(ticker: str = "TCS.NS"):
    db = SessionLocal()
    try:
        cutoff = datetime.utcnow() - timedelta(days=7)  # need 5 trading sessions to have passed
        pending = (
            db.query(Prediction)
            .filter(Prediction.ticker == ticker, Prediction.was_correct == -1, Prediction.ts < cutoff)
            .all()
        )
        if not pending:
            print("No pending predictions old enough to evaluate.")
            return

        df = fetch_ohlcv(ticker, period="6mo", interval="1d")

        y_true, y_pred = [], []
        for p in pending:
            future = df[df.index >= p.ts.date().isoformat()]
            if len(future) < 6:
                continue
            entry_price = future["close"].iloc[0]
            exit_price = future["close"].iloc[5]
            pct_return = (exit_price / entry_price - 1) * 100

            actual = label_outcome(pct_return)
            p.actual_outcome = actual
            p.actual_return_pct = round(float(pct_return), 2)
            p.was_correct = 1 if actual == p.final_decision else 0

            y_true.append(actual)
            y_pred.append(p.final_decision)

        db.commit()

        if not y_true:
            print("No predictions had enough forward data yet.")
            return

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred) * 100,
            "precision": precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0) * 100,
            "recall": recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0) * 100,
            "f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0) * 100,
        }

        def label_acc(label):
            idx = [i for i, t in enumerate(y_true) if t == label]
            if not idx:
                return None
            correct = sum(1 for i in idx if y_pred[i] == label)
            return round(correct / len(idx) * 100, 2)

        run = EvaluationRun(
            ticker=ticker,
            accuracy=round(metrics["accuracy"], 2), precision=round(metrics["precision"], 2),
            recall=round(metrics["recall"], 2), f1=round(metrics["f1"], 2),
            buy_accuracy=label_acc("BUY"), sell_accuracy=label_acc("SELL"), hold_accuracy=label_acc("HOLD"),
            sample_size=len(y_true),
        )
        db.add(run)
        db.commit()
        print(f"Evaluation complete: {metrics}")
    finally:
        db.close()


if __name__ == "__main__":
    run_evaluation()
