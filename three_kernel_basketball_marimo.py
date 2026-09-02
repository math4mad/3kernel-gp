import marimo

__generated_with = "0.17.2"
app = marimo.App(width="full")


@app.cell
def _():
    import warnings

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )
    from sklearn.model_selection import cross_val_score, train_test_split

    warnings.filterwarnings("ignore")
    return (
        ConstantKernel,
        GaussianProcessClassifier,
        RBF,
        WhiteKernel,
        accuracy_score,
        classification_report,
        confusion_matrix,
        cross_val_score,
        f1_score,
        np,
        plt,
        roc_auc_score,
        train_test_split,
    )


@app.cell
def _(np):
    def generate_basketball_data(n_samples=500, noise=0.1, random_state=42):
        np.random.seed(random_state)
        positions = np.zeros((n_samples, 2, 5, 2))
        labels = np.zeros(n_samples)

        for sample_index in range(n_samples):
            if np.random.rand() > 0.5:
                positions[sample_index, 0] = [
                    [0.3, 0.5], [0.35, 0.45], [0.1, 0.8], [0.7, 0.8], [0.5, 0.2]
                ]
                positions[sample_index, 1] = [
                    [0.4, 0.6], [0.5, 0.5], [0.2, 0.7], [0.6, 0.7], [0.5, 0.3]
                ]
                labels[sample_index] = 1
            else:
                positions[sample_index, 0] = [
                    [0.2, 0.3], [0.6, 0.4], [0.3, 0.7], [0.7, 0.6], [0.4, 0.2]
                ]
                positions[sample_index, 1] = [
                    [0.3, 0.4], [0.5, 0.4], [0.3, 0.6], [0.5, 0.6], [0.4, 0.3]
                ]
                labels[sample_index] = 0

            positions[sample_index] += np.random.randn(2, 5, 2) * noise

        return positions, labels

    def compute_rbf_kernel(point_a, point_b, sigma=1.0):
        difference = point_a - point_b
        return np.exp(-np.sum(difference**2) / (2 * sigma**2))

    def compute_three_kernel_features(positions, sigma=1.0):
        sample_count = positions.shape[0]
        features = np.zeros((sample_count, 3))

        for sample_index in range(sample_count):
            offense_positions = positions[sample_index, 0]
            defense_positions = positions[sample_index, 1]
            offense_kernel = 0.0
            defense_kernel = 0.0
            cross_kernel = 0.0

            for player_a in range(5):
                for player_b in range(5):
                    offense_kernel += compute_rbf_kernel(
                        offense_positions[player_a], offense_positions[player_b], sigma
                    )
                    defense_kernel += compute_rbf_kernel(
                        defense_positions[player_a], defense_positions[player_b], sigma
                    )
                    cross_kernel += compute_rbf_kernel(
                        offense_positions[player_a], defense_positions[player_b], sigma
                    )

            features[sample_index] = [
                offense_kernel / 25.0,
                defense_kernel / 25.0,
                cross_kernel / 25.0,
            ]

        return features

    return compute_three_kernel_features, generate_basketball_data


@app.cell
def _(plt):
    def plot_court(axis, offense_positions, defense_positions, title):
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_aspect("equal")
        axis.grid(True, alpha=0.3)
        axis.scatter(
            offense_positions[:, 0], offense_positions[:, 1],
            c="blue", s=100, label="Offense", edgecolors="black", zorder=5,
        )
        axis.scatter(
            defense_positions[:, 0], defense_positions[:, 1],
            c="red", s=100, label="Defense", marker="s", edgecolors="black", zorder=5,
        )
        for player_index, position in enumerate(offense_positions):
            axis.annotate(str(player_index + 1), position, xytext=(5, 5), textcoords="offset points")
        for player_index, position in enumerate(defense_positions):
            axis.annotate(str(player_index + 1), position, xytext=(5, 5), textcoords="offset points")
        axis.add_patch(plt.Circle((0.5, 0.5), 0.1, fill=False, color="gray", linestyle="--"))
        axis.set_title(title)
        axis.legend(loc="upper right")

    return (plot_court,)


@app.cell
def _(generate_basketball_data, np, plot_court, plt):
    positions, labels = generate_basketball_data()
    sample_figure, sample_axes = plt.subplots(2, 2, figsize=(10, 10))
    sample_axes = sample_axes.flatten()
    success_indices = np.where(labels == 1)[0][:2]
    failure_indices = np.where(labels == 0)[0][:2]

    for sample_index, axis in zip(np.concatenate([success_indices, failure_indices]), sample_axes):
        outcome = "Success" if labels[sample_index] == 1 else "Failure"
        plot_court(
            axis,
            positions[sample_index, 0],
            positions[sample_index, 1],
            f"Sample {sample_index} - {outcome}",
        )
    sample_figure.tight_layout()
    sample_figure
    return labels, positions


@app.cell
def _(ConstantKernel, GaussianProcessClassifier, RBF, WhiteKernel, compute_three_kernel_features, cross_val_score, labels, positions, train_test_split):
    feature_names = ["K_self", "K_opponent", "K_cross"]
    features = compute_three_kernel_features(positions)
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(1e-3, (1e-4, 1e-1))
    classifier = GaussianProcessClassifier(
        kernel=kernel, n_restarts_optimizer=5, max_iter_predict=100, random_state=42
    )
    classifier.fit(train_features, train_labels)
    train_predictions = classifier.predict(train_features)
    train_probabilities = classifier.predict_proba(train_features)[:, 1]
    test_predictions = classifier.predict(test_features)
    test_probabilities = classifier.predict_proba(test_features)[:, 1]
    cross_validation_scores = cross_val_score(classifier, features, labels, cv=5, scoring="accuracy")
    single_kernel_scores = []
    for _feature_index in range(3):
        single_features = features[:, _feature_index].reshape(-1, 1)
        single_kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-3)
        single_classifier = GaussianProcessClassifier(kernel=single_kernel, random_state=42)
        single_kernel_scores.append(
            cross_val_score(single_classifier, single_features, labels, cv=5, scoring="accuracy").mean()
        )
    return (
        classifier,
        cross_validation_scores,
        feature_names,
        features,
        single_kernel_scores,
        test_labels,
        test_predictions,
        test_probabilities,
        train_labels,
        train_predictions,
        train_probabilities,
    )


@app.cell
def _(accuracy_score, classification_report, confusion_matrix, cross_validation_scores, f1_score, roc_auc_score, test_labels, test_predictions, test_probabilities, train_labels, train_predictions, train_probabilities):
    metrics = {
        "training_accuracy": accuracy_score(train_labels, train_predictions),
        "training_f1": f1_score(train_labels, train_predictions),
        "training_auc": roc_auc_score(train_labels, train_probabilities),
        "test_accuracy": accuracy_score(test_labels, test_predictions),
        "test_f1": f1_score(test_labels, test_predictions),
        "test_auc": roc_auc_score(test_labels, test_probabilities),
        "cross_validation_accuracy": cross_validation_scores.mean(),
        "cross_validation_std": cross_validation_scores.std(),
    }
    print("Training metrics:", metrics["training_accuracy"], metrics["training_f1"], metrics["training_auc"])
    print("Test metrics:", metrics["test_accuracy"], metrics["test_f1"], metrics["test_auc"])
    print("Five-fold accuracy:", metrics["cross_validation_accuracy"], "+/-", metrics["cross_validation_std"])
    print("\nClassification report:\n", classification_report(test_labels, test_predictions, target_names=["Failure", "Success"]))
    print("Confusion matrix:\n", confusion_matrix(test_labels, test_predictions))
    return (metrics,)


@app.cell
def _(feature_names, features, labels, np, plt, single_kernel_scores):
    feature_figure, feature_axes = plt.subplots(1, 3, figsize=(15, 5))
    for _feature_index, feature_name in enumerate(feature_names):
        for _label in [0, 1]:
            mask = labels == _label
            feature_axes[_feature_index].scatter(
                features[mask, _feature_index], np.zeros(np.sum(mask)),
                c="red" if _label == 0 else "blue",
                alpha=0.5,
                label="Failure" if _label == 0 else "Success",
            )
        feature_axes[_feature_index].set_xlabel(feature_name)
        feature_axes[_feature_index].set_ylabel("Kernel density")
        feature_axes[_feature_index].set_title(f"{feature_name} Distribution")
        feature_axes[_feature_index].legend()
    feature_figure.tight_layout()
    print("Single-kernel cross-validation scores:", dict(zip(feature_names, single_kernel_scores)))
    feature_figure
    return


@app.cell
def _(plt, test_labels, test_probabilities, train_labels, train_probabilities):
    probability_figure, probability_axes = plt.subplots(1, 2, figsize=(12, 4))
    probability_axes[0].hist(train_probabilities[train_labels == 0], bins=20, alpha=0.5, label="Failure", color="red")
    probability_axes[0].hist(train_probabilities[train_labels == 1], bins=20, alpha=0.5, label="Success", color="blue")
    probability_axes[0].set_xlabel("Predicted probability")
    probability_axes[0].set_ylabel("Frequency")
    probability_axes[0].set_title("Training Prediction Probabilities")
    probability_axes[0].legend()
    probability_axes[1].hist(test_probabilities[test_labels == 0], bins=20, alpha=0.5, label="Failure", color="red")
    probability_axes[1].hist(test_probabilities[test_labels == 1], bins=20, alpha=0.5, label="Success", color="blue")
    probability_axes[1].set_xlabel("Predicted probability")
    probability_axes[1].set_ylabel("Frequency")
    probability_axes[1].set_title("Test Prediction Probabilities")
    probability_axes[1].legend()
    probability_figure.tight_layout()
    probability_figure
    return


if __name__ == "__main__":
    app.run()
