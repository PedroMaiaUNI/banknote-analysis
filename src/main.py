import time
import numpy as np
from utils import load_data, normalize
from cross_validation import k_fold_split
from evaluation import evaluate_model, summarize_results
from k_nearest_neighbors import KNN
from bayes_univariado import BayesUnivariado
from bayes_multivariado import BayesMultivariado

def main():
    X, y = load_data("data/data_banknote_authentication.txt")
    X = normalize(X)

    k_folds = k_fold_split(X, y, k=10, random_state=42)

    models = [
        ("KNN (Euclidiana)", KNN(k=5, metric='euclidean')),
        ("KNN (Manhattan)", KNN(k=5, metric='manhattan')),
        ("Bayes Univariado", BayesUnivariado()),
        ("Bayes Multivariado", BayesMultivariado())
    ]

    all_results = []
    for name, model in models:
        print(f"Treinando {name}...")
        results = evaluate_model(model, k_folds)
        all_results.append((name, *results))

    summarize_results(all_results, "results/results_table.txt")

if __name__ == "__main__":
    main()
