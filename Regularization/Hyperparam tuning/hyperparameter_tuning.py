import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import randint, uniform

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)


# ── 1. Grid Search (Exhaustive) ─────────────────────────────────────────────
# Tests EVERY combination of hyperparameters defined in param_grid.
# Pros: Guarantees finding best combo within the grid
# Cons: Computationally expensive as parameters grow (curse of dimensionality)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1,
)

grid_search.fit(X_train, y_train)

print("=" * 70)
print("GRID SEARCH RESULTS")
print("=" * 70)
print(f"Best Parameters : {grid_search.best_params_}")
print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")

grid_pred = grid_search.predict(X_test)
print(f"Test Accuracy   : {accuracy_score(y_test, grid_pred):.4f}")
print(f"\nTotal fits: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * 5}")


# ── 2. Random Search (Stochastic Sampling) ──────────────────────────────────
# Randomly samples N combinations from specified distributions.
# Pros: More efficient for large search spaces; can explore wider ranges
# Cons: No guarantee of finding the absolute best (but often finds good-enough)

param_distributions = {
    "n_estimators": randint(50, 300),
    "max_depth": [None] + list(range(3, 30)),
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 10),
    "max_features": uniform(0.1, 0.9),
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=50,          # number of random combinations to try
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    random_state=42,
    verbose=1,
)

random_search.fit(X_train, y_train)

print("\n" + "=" * 70)
print("RANDOM SEARCH RESULTS")
print("=" * 70)
print(f"Best Parameters : {random_search.best_params_}")
print(f"Best CV Accuracy: {random_search.best_score_:.4f}")

random_pred = random_search.predict(X_test)
print(f"Test Accuracy   : {accuracy_score(y_test, random_pred):.4f}")
print(f"\nTotal fits: {50 * 5}")


# ── 3. Comparison ────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)

grid_total_fits = len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * 5
random_total_fits = 50 * 5

print(f"{'Method':<20} {'Best CV Acc':<15} {'Test Acc':<15} {'Total Fits':<15}")
print("-" * 65)
print(f"{'Grid Search':<20} {grid_search.best_score_:<15.4f} {accuracy_score(y_test, grid_pred):<15.4f} {grid_total_fits:<15}")
print(f"{'Random Search':<20} {random_search.best_score_:<15.4f} {accuracy_score(y_test, random_pred):<15.4f} {random_total_fits:<15}")

print("\nKey Takeaways:")
print("- Grid Search is thorough but slow; best for small, well-defined parameter spaces.")
print("- Random Search is faster and often finds comparable results for large search spaces.")
