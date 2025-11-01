# Projeto IA - AV2: Autenticação de Cédulas

Este projeto implementa, **sem uso de bibliotecas pré-escritas de machine learning**, os seguintes classificadores:
- **K-Vizinhos Mais Próximos (KNN)** com distâncias Euclidiana e Manhattan;
- **Classificador Bayesiano Univariado e Multivariado**.

O dataset utilizado é o [Banknote Authentication Dataset](https://archive.ics.uci.edu/ml/datasets/banknote+authentication).

## Estrutura do projeto
O código-fonte está na pasta `src/`, organizado por módulo:
- `k_nearest_neighbors.py` → Implementação do KNN;
- `bayes_univariado.py` e `bayes_multivariado.py` → Classificadores bayesianos;
- `cross_validation.py` → Implementação da validação cruzada (10 folds);
- `evaluation.py` → Cálculo de acurácia, precisão e F1-score.

## Execução
**RECOMENDADO**: abra uma ambiente virtual para a execução do projeto:
```bash
python -m venv env
```
Após isso, instale numpy:
```bash
pip install numpy
```
Por fim, execute:
```bash
python src/main.py
```

O script principal gera uma tabela comparativa (`results/results_table.txt`) com médias, desvios-padrão e tempos de execução.

## Restrições
- Não é permitido o uso de **sklearn**, **pandas** ou **bibliotecas externas de ML**.
- Todo o código foi implementado manualmente, utilizando apenas `numpy` e funções nativas do Python.
