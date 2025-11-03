import numpy as np
import time
from utils import normalize
from evaluation_wrapper import evaluate_model_with_data
from evaluation import summarize_results
from k_nearest_neighbors import KNN
from bayes_univariado import NaiveBayesUnivariado
from bayes_multivariado import BayesMultivariado
from cross_validation import k_fold_split

def main():
    data = np.loadtxt("data/data_banknote_authentication.txt", delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]

    folds = k_fold_split(X, y, k=10, random_state=42)

    models = [
        ("KNN (Dist. Euclidiano)", KNN(k=5, task='euclidean')),
        ("KNN (Dist. Manhattan)", KNN(k=5, task='manhattan')),
        ("KNN (Dist. Chebyshev)", KNN(k=5, task='chebyshev')),
        ("Bayes Univariado", NaiveBayesUnivariado()),
        ("Bayes Multivariado", BayesMultivariado()),
    ]

    results = []
    for name, model in models:
        print(f"\nTreinando e testando {name} com 10-Fold CV...")

        res = evaluate_model_with_data(model, X, y, folds)

        results.append((name, res))
    summarize_results(results, "results/results_table.csv")
    print("\nResultados da validação cruzada salvos em results/results_table.csv")

if __name__ == '__main__':
    main()