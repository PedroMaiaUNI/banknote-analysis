# Implementação do KNN (Euclidiana e Manhattan)

import numpy as np
class KNN:
    def __init__(self, k=5, task="classification"): #construtor em python
        self.k = k
        self.task = task
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))
    def predict(self, X_test):
        return np.array([self.calculate_prediction(x) for x in X_test])
    def calculate_prediction(self, x):
        #computar a distancia entre x e todos exemplos do conjunto de treino
        distances = [self.euclidean_distance(x, x_train)
                     for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        #extrair os labels dos exemplos de treino
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        if self.task == "classification":
            #retorno da classe mais frequente
            unique, counts = np.unique(k_nearest_labels, return_counts= True)
            return unique[np.argmax(counts)]
        elif self.task == "regression":
            #retorno da media dos labels
            return np.mean(k_nearest_labels)
        else:
            raise ValueError("Tarefa nao foi definida")
