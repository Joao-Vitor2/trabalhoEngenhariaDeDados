# Pipeline Modularizado de Classificação de Churn

Solução em Python dividida em módulos reutilizáveis para identificar clientes
inativos com risco de não voltarem a comprar (churn), a partir de um dataset
bruto de transações.

## Estrutura do projeto

```
projeto_modular_trio/
├── data/
│   └── raw_transactions.csv   # Dataset bruto de transações
├── data_processing.py         # Módulo de tratamento e pivot (Pessoa A)
├── model_training.py          # Módulo de treino e avaliação (Pessoa B)
├── main.py                    # Orquestração e inferência (Pessoa C)
├── requirements.txt           # Dependências
└── README.md
```

## Divisão de responsabilidades

| Módulo | Pessoa | Responsabilidade |
|--------|--------|------------------|
| `data_processing.py` | A — Engenharia de Dados | Ler o CSV, aplicar `pd.pivot_table` para agregar gasto por categoria e frequência de compras, gerar a flag `churn`. |
| `model_training.py` | B — Ciência de Dados | Separar features/alvo, split treino/teste 70/30, treinar o modelo e calcular métricas (acurácia + matriz de confusão). |
| `main.py` | C — MLOps & Integração | Importar os módulos A e B, coordenar o pipeline e prever o churn de um novo cliente. |

## Regra de negócio (churn)

- `churn = 1` quando a **frequência de compras é menor que 8** (cliente em risco).
- `churn = 0` caso contrário.

O limite é configurável em `data_processing.LIMITE_FREQUENCIA_CHURN`.

## Modelo escolhido

`RandomForestClassifier` (scikit-learn). Justificativa:

- Captura relações não lineares entre gasto por categoria e churn.
- Não exige normalização das features.
- Robusto a overfitting em datasets pequenos (ensemble de árvores).

> Observação sobre data leakage: a coluna `frequencia` é usada para **gerar** o
> alvo, então ela é **excluída das features** de treino. O modelo aprende a
> prever churn a partir dos padrões de gasto por categoria.

## Como executar

1. Instalar as dependências:

   ```bash
   pip install -r requirements.txt
   ```

2. Rodar o pipeline completo:

   ```bash
   python main.py
   ```

Isso executa as três etapas em sequência: tratamento dos dados, treino/avaliação
do modelo e inferência para um novo cliente de exemplo.

## Saída esperada (exemplo)

```
>>> ETAPA 1: Tratamento dos dados
Clientes processados: 50
...
>>> ETAPA 2: Treino e avaliação do modelo
Acurácia: 100.00%
Matriz de Confusão:
[[8 0]
 [0 7]]
...
>>> ETAPA 3: Inferência para um novo cliente
RESULTADO DA PREVISÃO: CHURN (risco de não voltar)
```
