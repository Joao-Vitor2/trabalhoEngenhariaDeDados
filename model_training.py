# Parte de ciencia de dados (Pessoa B)
# Pega os dados ja tratados, separa treino/teste, treina o modelo e mostra as metricas.
#
# Escolhi RandomForest porque funciona bem sem precisar normalizar os dados,
# lida com relacoes nao lineares e nao sofre tanto com overfitting em dataset pequeno.

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# tiro a coluna churn (que e o alvo) e a frequencia.
# a frequencia sai porque foi ela que gerou o churn, entao usar ela seria colar (data leakage)
COLUNAS_NAO_FEATURES = ["churn", "frequencia"]

RANDOM_STATE = 42
TEST_SIZE = 0.30  # 70% treino, 30% teste


def separar_features_alvo(tabela):
    colunas_features = [c for c in tabela.columns if c not in COLUNAS_NAO_FEATURES]
    X = tabela[colunas_features]
    y = tabela["churn"]
    return X, y, colunas_features


def treinar_modelo(tabela):
    X, y, colunas_features = separar_features_alvo(tabela)

    # stratify pra manter a mesma proporcao de churn no treino e no teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    modelo = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
    )
    modelo.fit(X_treino, y_treino)

    # testa o modelo no que ele nao viu
    y_predito = modelo.predict(X_teste)
    metricas = {
        "acuracia": accuracy_score(y_teste, y_predito),
        "matriz_confusao": confusion_matrix(y_teste, y_predito),
        "relatorio": classification_report(y_teste, y_predito, zero_division=0),
        "tamanho_treino": len(X_treino),
        "tamanho_teste": len(X_teste),
    }

    return modelo, colunas_features, metricas


def exibir_metricas(metricas):
    print("=" * 55)
    print("AVALIACAO DO MODELO (RandomForest)")
    print("=" * 55)
    print(f"Amostras de treino: {metricas['tamanho_treino']}")
    print(f"Amostras de teste:  {metricas['tamanho_teste']}")
    print(f"\nAcuracia: {metricas['acuracia']:.2%}")
    print("\nMatriz de Confusao (linha=real, coluna=previsto):")
    print(metricas["matriz_confusao"])
    print("\nRelatorio de Classificacao:")
    print(metricas["relatorio"])


if __name__ == "__main__":
    from data_processing import processar_dados

    dados = processar_dados("data/raw_transactions.csv")
    modelo, features, metricas = treinar_modelo(dados)
    exibir_metricas(metricas)
