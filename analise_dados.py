# ============================================================
# PROJETO EDA - ANÁLISE EXPLORATÓRIA DE DADOS
# Análise de Risco à Saúde
# Curso: Análise de Dados com Python - SENAI
# Ambiente: VS Code + Python
# ============================================================

# ------------------------------------------------------------
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------------------------------------
# 2. CARREGAMENTO DOS DADOS
# ------------------------------------------------------------

# O arquivo CSV deve estar na mesma pasta deste script.
df = pd.read_csv('dataset_saude.csv')


# ------------------------------------------------------------
# 3. INSPEÇÃO E QUALIDADE DOS DADOS
# ------------------------------------------------------------

print('\n' + '=' * 60)
print('1. INSPEÇÃO INICIAL DOS DADOS')
print('=' * 60)

print('\nPrimeiras linhas:')
print(df.head())

print('\nInformações do DataFrame:')
df.info()

print('\nQuantidade de linhas e colunas:')
print(df.shape)

print('\nValores ausentes:')
print(df.isnull().sum())

print('\nTotal de valores ausentes:')
print(df.isnull().sum().sum())

print('\nValores duplicados:')
print(df.duplicated().sum())

print('\nEstatísticas descritivas:')
print(df.describe().round(2))


# ------------------------------------------------------------
# 4. DISTRIBUIÇÃO DAS VARIÁVEIS CATEGÓRICAS
# ------------------------------------------------------------

print('\n' + '=' * 60)
print('2. DISTRIBUIÇÃO DAS VARIÁVEIS CATEGÓRICAS')
print('=' * 60)

variaveis_categoricas = [
    'exercicio',
    'consumo_acucar',
    'fumante',
    'alcool',
    'casado',
    'profissao',
    'risco_saude'
]

for coluna in variaveis_categoricas:
    print(f'\n{coluna}:')
    print(df[coluna].value_counts())


# ------------------------------------------------------------
# 5. ANÁLISE UNIVARIADA
# ------------------------------------------------------------

print('\n' + '=' * 60)
print('3. ANÁLISE UNIVARIADA')
print('=' * 60)

# 5.1 - Distribuição do risco de saúde
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x='risco_saude')
plt.title('Distribuição do Risco à Saúde')
plt.xlabel('Risco à Saúde')
plt.ylabel('Quantidade')
plt.tight_layout()
plt.show()


# 5.2 - Distribuição da idade
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='idade', bins=15, kde=True)
plt.title('Distribuição da Idade')
plt.xlabel('Idade')
plt.ylabel('Quantidade')
plt.tight_layout()
plt.show()


# 5.3 - Distribuição do IMC
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='imc', bins=20, kde=True)
plt.title('Distribuição do IMC')
plt.xlabel('IMC')
plt.ylabel('Quantidade')
plt.tight_layout()
plt.show()


# 5.4 - Distribuição do nível de exercício
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x='exercicio')
plt.title('Distribuição do Nível de Exercício')
plt.xlabel('Nível de Exercício')
plt.ylabel('Quantidade')
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 6. ANÁLISE BIVARIADA
# Relação entre variáveis e risco de saúde
# ------------------------------------------------------------

print('\n' + '=' * 60)
print('4. ANÁLISE BIVARIADA')
print('=' * 60)


# 6.1 - Exercício x risco
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='exercicio', hue='risco_saude')
plt.title('Risco de Saúde por Nível de Exercício')
plt.xlabel('Nível de Exercício')
plt.ylabel('Quantidade')
plt.legend(title='Risco de Saúde')
plt.tight_layout()
plt.show()

proporcao_exercicio = pd.crosstab(
    df['exercicio'],
    df['risco_saude'],
    normalize='index'
) * 100

print('\nProporção de risco por nível de exercício:')
print(proporcao_exercicio.round(2).astype(str) + '%')


# 6.2 - Sono x risco
media_sono = df.groupby('risco_saude')['sono'].mean()

print('\nMédia de sono por risco de saúde:')
print(media_sono.round(2))

plt.figure(figsize=(7, 5))
sns.barplot(x=media_sono.index, y=media_sono.values)
plt.title('Média de Horas de Sono por Risco de Saúde')
plt.xlabel('Risco de Saúde')
plt.ylabel('Média de Horas de Sono')
plt.tight_layout()
plt.show()


# 6.3 - Tabagismo x risco
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x='fumante', hue='risco_saude')
plt.title('Risco de Saúde por Status de Tabagismo')
plt.xlabel('Fumante')
plt.ylabel('Quantidade')
plt.legend(title='Risco de Saúde')
plt.tight_layout()
plt.show()

proporcao_fumante = pd.crosstab(
    df['fumante'],
    df['risco_saude'],
    normalize='index'
) * 100

print('\nProporção de risco por status de tabagismo:')
print(proporcao_fumante.round(2).astype(str) + '%')


# 6.4 - Álcool x risco
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x='alcool', hue='risco_saude')
plt.title('Risco de Saúde por Consumo de Álcool')
plt.xlabel('Consumo de Álcool')
plt.ylabel('Quantidade')
plt.legend(title='Risco de Saúde')
plt.tight_layout()
plt.show()

proporcao_alcool = pd.crosstab(
    df['alcool'],
    df['risco_saude'],
    normalize='index'
) * 100

print('\nProporção de risco por consumo de álcool:')
print(proporcao_alcool.round(2).astype(str) + '%')


# 6.5 - Consumo de açúcar x risco
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='consumo_acucar', hue='risco_saude')
plt.title('Risco de Saúde por Nível de Consumo de Açúcar')
plt.xlabel('Consumo de Açúcar')
plt.ylabel('Quantidade')
plt.legend(title='Risco de Saúde')
plt.tight_layout()
plt.show()

proporcao_acucar = pd.crosstab(
    df['consumo_acucar'],
    df['risco_saude'],
    normalize='index'
) * 100

print('\nProporção de risco por consumo de açúcar:')
print(proporcao_acucar.round(2).astype(str) + '%')


# 6.6 - IMC x risco
media_imc = df.groupby('risco_saude')['imc'].mean()

print('\nMédia de IMC por risco de saúde:')
print(media_imc.round(2))

plt.figure(figsize=(7, 5))
sns.barplot(x=media_imc.index, y=media_imc.values)
plt.title('Média de IMC por Risco de Saúde')
plt.xlabel('Risco de Saúde')
plt.ylabel('Média de IMC')
plt.tight_layout()
plt.show()


# 6.7 - Idade x risco
media_idade = df.groupby('risco_saude')['idade'].mean()

print('\nMédia de idade por risco de saúde:')
print(media_idade.round(2))

plt.figure(figsize=(7, 5))
sns.barplot(x=media_idade.index, y=media_idade.values)
plt.title('Média de Idade por Risco de Saúde')
plt.xlabel('Risco de Saúde')
plt.ylabel('Média de Idade')
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 7. CORRELAÇÃO ENTRE VARIÁVEIS NUMÉRICAS
# ------------------------------------------------------------

print('\n' + '=' * 60)
print('5. ANÁLISE DE CORRELAÇÃO')
print('=' * 60)

variaveis_numericas = [
    'idade',
    'peso',
    'altura',
    'sono',
    'imc'
]

correlacao = df[variaveis_numericas].corr()

print('\nMatriz de correlação:')
print(correlacao.round(2))

plt.figure(figsize=(8, 6))
sns.heatmap(
    correlacao,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)
plt.title('Matriz de Correlação das Variáveis Numéricas')
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 8. PRINCIPAIS INSIGHTS
# ------------------------------------------------------------

print('\n' + '=' * 60)
print('6. PRINCIPAIS INSIGHTS')
print('=' * 60)

percentual_alto_risco = np.mean(df['risco_saude'] == 'alto') * 100

print(f'\n1. Distribuição do risco:')
print(f'   {percentual_alto_risco:.2f}% da amostra foi classificada como alto risco.')

if 'baixo' in proporcao_exercicio.index and 'nenhum' in proporcao_exercicio.index:
    alto_baixo_exercicio = proporcao_exercicio.loc['baixo', 'alto']
    alto_nenhum_exercicio = proporcao_exercicio.loc['nenhum', 'alto']

    print('\n2. Exercício:')
    print(
        f'   Baixo exercício: {alto_baixo_exercicio:.2f}% de alto risco.'
    )
    print(
        f'   Nenhum exercício: {alto_nenhum_exercicio:.2f}% de alto risco.'
    )

if 'sim' in proporcao_fumante.index:
    alto_fumantes = proporcao_fumante.loc['sim', 'alto']

    print('\n3. Tabagismo:')
    print(f'   Fumantes: {alto_fumantes:.2f}% de alto risco.')

if 'sim' in proporcao_alcool.index:
    alto_alcool = proporcao_alcool.loc['sim', 'alto']

    print('\n4. Álcool:')
    print(f'   Consumidores: {alto_alcool:.2f}% de alto risco.')

if 'alto' in proporcao_acucar.index:
    alto_acucar = proporcao_acucar.loc['alto', 'alto']

    print('\n5. Consumo de açúcar:')
    print(f'   Consumo alto: {alto_acucar:.2f}% de alto risco.')

print('\n6. IMC:')
print(f"   Alto risco: {media_imc['alto']:.2f}")
print(f"   Baixo risco: {media_imc['baixo']:.2f}")

print('\n7. Idade:')
print(f"   Alto risco: {media_idade['alto']:.2f} anos")
print(f"   Baixo risco: {media_idade['baixo']:.2f} anos")

print('\n8. Sono:')
print(f"   Alto risco: {media_sono['alto']:.2f} horas")
print(f"   Baixo risco: {media_sono['baixo']:.2f} horas")

print('\n9. Correlações de maior destaque:')
print(f"   Peso x IMC: {correlacao.loc['peso', 'imc']:.2f}")
print(f"   Altura x IMC: {correlacao.loc['altura', 'imc']:.2f}")


# ------------------------------------------------------------
# 9. LIMITAÇÕES DA ANÁLISE
# ------------------------------------------------------------

print('\n' + '=' * 60)
print('7. LIMITAÇÕES DA ANÁLISE')
print('=' * 60)

print("""
- Os resultados representam associações observadas na amostra.
- Correlação não implica causalidade.
- Peso, altura e IMC possuem relação matemática direta, portanto
  suas correlações devem ser interpretadas com cautela.
- A análise exploratória identifica padrões, mas não permite,
  sozinha, afirmar relações de causa e efeito.
""")


# ------------------------------------------------------------
# 10. CONCLUSÃO
# ------------------------------------------------------------

print('\n' + '=' * 60)
print('8. CONCLUSÃO')
print('=' * 60)

print("""
A análise exploratória permitiu identificar padrões relevantes
entre características demográficas, comportamentais e a
classificação de risco à saúde.

Na amostra analisada, foram observadas diferenças na proporção
de alto risco entre os níveis de exercício, tabagismo, consumo
de álcool e consumo de açúcar. Também foram observadas diferenças
nas médias de idade, IMC e horas de sono entre os grupos de risco.

A análise de correlação destacou principalmente a relação entre
peso e IMC, além da relação entre altura e IMC.

Os resultados demonstram como técnicas de EDA podem ser utilizadas
para organizar, explorar, visualizar e interpretar dados, apoiando
a identificação de padrões relevantes para análises posteriores.
""")


# ============================================================
# FIM DO PROJETO
# ============================================================