import argparse
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--model_dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    args = parser.parse_args()

    data = np.loadtxt(os.path.join(args.train, "train.csv"), delimiter=",")
    y_train = data[:, 0].astype(int)
    X_train = data[:, 1:]

    model = RandomForestClassifier(
        n_estimators=args.n_estimators, random_state=args.random_state
    )
    model.fit(X_train, y_train)
    print(f"train accuracy: {model.score(X_train, y_train):.4f}")

    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))

if __name__ == "__main__":
    main()