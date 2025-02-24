import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from typing import Dict
import matplotlib.pyplot as plt  # Importar biblioteca para gráficos

def main() -> str:
    """
    Função principal para carregar dados, pré-processar, treinar um modelo de classificação
    e avaliar seu desempenho.
    """
    # Carregar os dados
    data = pd.read_csv('./car.data', header=None)  # Lê os dados do arquivo CSV
    data.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']  # Define os nomes das colunas

    # Pré-processar os dados
    label_encoders: Dict[str, LabelEncoder] = {}  # Dicionário para armazenar os codificadores de rótulos
    for column in data.columns:
        label_encoders[column] = LabelEncoder()  # Cria um codificador de rótulos para cada coluna
        data[column] = label_encoders[column].fit_transform(data[column])  # Aplica a codificação de rótulos

    # Dividir os dados em conjuntos de treino e teste
    X = data.drop('class', axis=1)  # Define as variáveis independentes
    y = data['class']  # Define a variável dependente
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.32, random_state=100)  # Divide os dados

    # Treinar o modelo de classificação
    classifier = SVC(random_state=42)  # Cria o classificador SVM
    classifier.fit(X_train, y_train)  # Treina o classificador

    # Fazer previsões
    y_pred = classifier.predict(X_test)  # Faz previsões com o classificador treinado

    # Avaliar o modelo
    accuracy = accuracy_score(y_test, y_pred)  # Calcula a acurácia do modelo
    precision = precision_score(y_test, y_pred, average='macro') # Calcula a precisão do modelo
    recall = recall_score(y_test, y_pred, average='macro') # Calcula o recall do modelo
    report = classification_report(y_test, y_pred, target_names=label_encoders['class'].classes_)  # Gera o relatório de classificação

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print('Media entre Acurácia, Precisão e Recall:', (accuracy + precision + recall) / 3)
    print('Classification Report:')
    print(report)  # Imprime o relatório de classificação
        
    return report  # Retorna o relatório de classificação

if __name__ == "__main__":
    main()  # Executa a função principal