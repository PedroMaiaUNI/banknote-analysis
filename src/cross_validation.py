# Implementação da validação cruzada 10-fold

import numpy as np
import math

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    if len(X) != len(y):
        raise ValueError("X e y devem ter o mesmo tamanho")

    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    np.random.shuffle(indices)
    print("Indices aleatórios:", indices)

    n_test = math.ceil(n_samples * test_size)  # arredonda pra baixo
    print("Tamanho do teste:", n_test)
    #n_test = math.ceil(n_samples * test_size)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    if X.ndim == 1:
        X_train, X_test = X[train_indices], X[test_indices]
    else:
        X_train, X_test = X[train_indices, :], X[test_indices, :]

    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test

def k_fold_split(X, y, k=10, random_state=None):
    """
    Gera os índices para a validação cruzada K-Fold.
    Retorna uma lista de tuplas (train_indices, test_indices).
    """
    if random_state:
        np.random.seed(random_state)
        
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Divide os índices em 'k' folds (partes)
    # np.array_split lida com casos onde n_samples não é divisível por k
    fold_indices = np.array_split(indices, k)
    
    folds = []
    for i in range(k):
        # O fold 'i' é o conjunto de teste
        test_indices = fold_indices[i]
        
        # Todos os outros folds são o conjunto de treino
        # np.concatenate é usado para juntar os arrays de índices
        train_indices = np.concatenate([fold_indices[j] for j in range(k) if j != i])
        
        folds.append((train_indices, test_indices))
        
    return folds