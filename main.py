# Script principal (Pessoa C)
# Junta tudo: trata os dados, treina o modelo e faz a previsao de um cliente novo.
# Roda com: python main.py

import os
import pandas as pd

from data_processing import processar_dados
from model_training import treinar_modelo, exibir_metricas

# monta o caminho do csv a partir da pasta do proprio arquivo,
# assim funciona de qualquer lugar que rodar o script
CAMINHO_CSV = os.path.join(os.path.dirname(__file__), "data", "raw_transactions.csv")


def montar_novo_cliente(colunas_features):
    # cliente de exemplo que comprou pouco, deve cair como churn
    gastos_novo_cliente = {
        "Alimentos": 45.00,
        "Casa": 0.0,
        "Eletronicos": 0.0,
        "Livros": 30.00,
        "Roupas": 0.0,
    }

    # garante que as colunas fiquem na mesma ordem do treino (o que faltar vira 0)
    registro = {coluna: gastos_novo_cliente.get(coluna, 0.0) for coluna in colunas_features}
    return pd.DataFrame([registro], columns=colunas_features)


def main():
    print("\n>>> ETAPA 1: Tratamento dos dados")
    dados = processar_dados(CAMINHO_CSV)
    print(f"Clientes processados: {len(dados)}")
    print("Previa da tabela com churn:")
    print(dados.head())

    print("\n>>> ETAPA 2: Treino e avaliacao do modelo")
    modelo, colunas_features, metricas = treinar_modelo(dados)
    exibir_metricas(metricas)

    print("\n>>> ETAPA 3: Previsao de um cliente novo")
    novo_cliente = montar_novo_cliente(colunas_features)
    print("Dados do cliente novo (gasto por categoria):")
    print(novo_cliente)

    predicao = modelo.predict(novo_cliente)[0]
    probabilidade = modelo.predict_proba(novo_cliente)[0]

    rotulo = "CHURN (risco de nao voltar)" if predicao == 1 else "ATIVO (sem risco)"
    print("\n" + "-" * 55)
    print(f"RESULTADO DA PREVISAO: {rotulo}")
    print(f"Classe prevista: {predicao}")
    print(f"Probabilidade [ativo=0, churn=1]: "
          f"[{probabilidade[0]:.2%}, {probabilidade[1]:.2%}]")
    print("-" * 55)


if __name__ == "__main__":
    main()
