# Parte da engenharia de dados (Pessoa A)
# Le o csv de transacoes, monta a tabela dinamica por cliente e cria a coluna de churn.

import pandas as pd

# cliente com menos de 8 compras a gente considera que ta em risco de sair
LIMITE_FREQUENCIA_CHURN = 8


def carregar_dados(caminho_csv):
    df = pd.read_csv(caminho_csv)
    return df


def gerar_tabela_dinamica(df):
    # soma quanto cada cliente gastou em cada categoria
    tabela_gastos = pd.pivot_table(
        df,
        index="id_cliente",
        columns="categoria",
        values="valor",
        aggfunc="sum",
        fill_value=0,
    )

    # conta quantas compras cada cliente fez
    frequencia = pd.pivot_table(
        df,
        index="id_cliente",
        values="id_transacao",
        aggfunc="count",
    ).rename(columns={"id_transacao": "frequencia"})

    # junta gasto por categoria + frequencia numa tabela so
    tabela = tabela_gastos.join(frequencia)
    tabela.columns.name = None
    return tabela


def gerar_flag_churn(tabela):
    tabela = tabela.copy()
    tabela["churn"] = (tabela["frequencia"] < LIMITE_FREQUENCIA_CHURN).astype(int)
    return tabela


def processar_dados(caminho_csv):
    # roda tudo em sequencia: le -> pivota -> gera churn
    df_bruto = carregar_dados(caminho_csv)
    tabela = gerar_tabela_dinamica(df_bruto)
    tabela_final = gerar_flag_churn(tabela)
    return tabela_final


if __name__ == "__main__":
    dados = processar_dados("data/raw_transactions.csv")
    print("Tabela com a flag de churn:")
    print(dados.head(10))
    print("\nDistribuicao de churn:")
    print(dados["churn"].value_counts())
