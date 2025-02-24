import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
from DecisionTreeClassifier import main as run_decision_tree
from SVM import main as run_svm
from KNN import main as run_knn

# Função para extrair métricas de um relatório de classificação
def get_metrics_from_report(report: str) -> Dict[str, float]:
    """
    Extrai as métricas de precisão, recall e acurácia do relatório de classificação.
    Utiliza a linha 'macro avg' para extrair precision e recall.
    """
    if report is None:
        return {'accuracy': 0, 'precision': 0, 'recall': 0}
    
    lines = report.split('\n')
    metrics = {'accuracy': 0, 'precision': 0, 'recall': 0}
    
    # Extrai acurácia
    for line in lines:
        if 'accuracy' in line.lower():
            tokens = line.split()
            # Assume que a acurácia está posicionada como penúltimo token
            try:
                metrics['accuracy'] = float(tokens[-2])
            except ValueError:
                metrics['accuracy'] = 0
            break

    # Procura a linha do "macro avg" para extrair precision e recall
    for line in lines:
        if line.strip().startswith("macro avg"):
            tokens = line.split()
            try:
                metrics['precision'] = float(tokens[2])
                metrics['recall'] = float(tokens[3])
            except (IndexError, ValueError):
                metrics['precision'] = metrics['recall'] = 0
            break
    
    return metrics

# Função principal para rodar os três algoritmos e comparar seus desempenhos
def main() -> None:
    """
    Função principal para rodar os três algoritmos e comparar seus desempenhos.
    """
    algorithms = {
        'Decision Tree': run_decision_tree,
        'SVM': run_svm,
        'KNN': run_knn
    }
    
    results = {
        'Algorithm': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'Average': []
    }
    
    # Executa cada algoritmo e coleta as métricas
    for name, func in algorithms.items():
        print(f'Running {name}...')
        report = func()
        metrics = get_metrics_from_report(report)
        accuracy = metrics['accuracy']
        precision = metrics['precision']
        recall = metrics['recall']
        average = (accuracy + precision + recall) / 3
        
        results['Algorithm'].append(name)
        results['Accuracy'].append(accuracy)
        results['Precision'].append(precision)
        results['Recall'].append(recall)
        results['Average'].append(average)
    
    # Cria um DataFrame com os resultados e plota um gráfico
    df = pd.DataFrame(results)
    df.set_index('Algorithm', inplace=True)
    df.plot(kind='line', figsize=(10, 6), marker='o')
    plt.title('Comparison of Classification Algorithms')
    plt.xlabel('Algorithm')
    plt.ylabel('Scores')
    
    # Define o limite inferior do eixo y
    min_score = df.min().min()
    lower_ylim = max(min_score - 0.15, 0)
    plt.ylim(lower_ylim, 1)
    
    plt.yticks([i * 0.05 for i in range(int(lower_ylim * 20), 21)])
    plt.legend(loc='best')
    plt.grid(axis='y', linestyle='--', linewidth=0.7)
    plt.show()

if __name__ == "__main__":
    main()
