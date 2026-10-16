import argparse
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--model_dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    args = parser.parse_args()

    data = np.loadtxt(os.path.join(args.train, "train.csv"), delimiter=",")
    y_train = data[:, 0].astype(int)
    X_train = data[:, 1:]

    model_rfc = RandomForestClassifier(
        n_estimators=args.n_estimators, random_state=args.random_state
    )
    model_rfc.fit(X_train, y_train)
    print(f"train accuracy (Random Forest): {model_rfc.score(X_train, y_train):.4f}")

    model_lr = LogisticRegression(
        max_iter=args.max_iter, random_state=args.random_state
    )
    model_lr.fit(X_train, y_train)
    print(f"train accuracy (Logistic Regression): {model_lr.score(X_train, y_train):.4f}")

    joblib.dump(model_rfc, os.path.join(args.model_dir, "model_rfc.joblib"))
    joblib.dump(model_lr, os.path.join(args.model_dir, "model_lr.joblib"))


if __name__ == "__main__":
    main()