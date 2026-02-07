from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

def linear_classifier_accuracy(
    reduced_feature,
    labeled_positions,
    y_labeled,
    test_size=0.2,
    random_state=42
):
    """
    Train a linear classifier on labeled points and return test accuracy.
    """
    # Need at least two classes
    if len(set(y_labeled)) <= 1:
        return None

    X = reduced_feature[labeled_positions]
    y = np.array(y_labeled)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        solver="lbfgs"
    )
    clf.fit(X_train_scaled, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test_scaled))
    return acc

