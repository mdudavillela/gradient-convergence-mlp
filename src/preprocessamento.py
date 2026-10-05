"""
preprocessamento.py
--------------------
Módulo responsável por transformar o csv coletado da API do TMDB em
dados prontos para treinar o Perceptron: filtra registros inválidos, cria
a variável-alvo, seleciona/constrói as features, e separa treino/teste
já normalizado.

Lógica para o pré-processamento (como?):
tmdb_raw.csv -> carregar_dados_validos() -> criar_alvo() -> selecionar_features() ->
tratar_categoricas() -> dividir_e_normalizar() -> (x_treino, x_teste, y_treino, y_teste)

Como usar este módulo dentro de um notebook:

    from preprocessamento import (
        carregar_dados_validos, criar_alvo, selecionar_features,
        tratar_categoricas, dividir_e_normalizar,
    )

    df = carregar_dados_validos("data/tmdb_raw.csv")
    df = criar_alvo(df)
    x_numerico = selecionar_features(df)
    x_categorico = tratar_categoricas(df)
    y = df["log_revenue"].values

    x_treino, x_teste, y_treino, y_teste, nomes_colunas = dividir_e_normalizar(
        x_numerico, x_categorico, y
    )
"""

"""
A seguinte função faz um processo chamado *multi-hot encoding*.

*Multi-hot encoding* é uma técnica de pré-processamento usada em aprendizado de máquina
para representar dados categóricos em formato numérico binário quando um único item pode pertencer a várias categorias ao mesmo tempo.

**Como funciona?**

• **Vetor binário**: Cria-se um vetor de tamanho fixo, onde cada posição corresponde a uma categoria possível.

• **Múltiplos "`1`s"**: Diferente do one-hot encoding tradicional (que ativa apenas uma posição com o valor 1),
o *multi-hot encoding* define como `1` todas as posições referentes às categorias presentes na amostra, mantendo as demais como `0`.

Esta célula de código implementa uma etapa de engenharia de recursos (feature engineering) e pré-processamento de variáveis categóricas.
O objetivo dela é transformar dados textuais complexos (gêneros e idiomas, nesse caso) em uma matriz de variáveis numéricas binárias (0 ou 1).
"""

import ast
# biblioteca ast: serve pra processar, analisar e modificar o próprio
# código-fonte do python antes de ele ser executado

import numpy as np
import pandas as pd

# ferramentas da biblioteca scikit-learn (sklearn) para preparação obrigatória
# de dados no perceptron
# train_test_split: divide o dataset de forma aleatória em treino e teste
# StandardScaler: padroniza a escala das variáveis numéricas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def carregar_dados_validos(caminho_csv):
  # carregar_dados_validos: lê o csv bruto da coleta e remove os filmes que
  # não servem pra esse projeto (os que têm algum dado ausente)

  df = pd.read_csv(caminho_csv)

  # garante que dados faltantes de budget e revenue não entrem no df
  df = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()

  # remove NA dessas colunas específicas que serão usadas futuramente
  df = df.dropna(subset = ["runtime", "release_date"]).copy()

  return df


def criar_alvo(df):
  # criar_alvo: cria a variável que o modelo vai tentar prever (log_revenue),
  # e também o log_budget e o release_year, que viram features depois

  df = df.copy()

  # log(1+x) é a transformação padrão na literatura de bilheteria pra deixar
  # a distribuição mais próxima da normal
  df["log_revenue"] = np.log1p(df["revenue"])
  df["log_budget"] = np.log1p(df["budget"])

  # converte pra um tipo de data de verdade (não como texto)
  # e .dt.year extrai só o ano (pq o ano de lançamento captura tendência ao
  # longo do tempo sem precisar lidar com mês/dia)
  df["release_year"] = pd.to_datetime(df["release_date"], errors = "coerce").dt.year

  # remove NA
  df = df.dropna(subset = ["release_year"]).copy()

  return df


def selecionar_features(df):
  # selecionar_features: isola só as colunas numéricas pré-lançamento que vão
  # virar entrada do modelo (orçamento em log (log_budget), duração (runtime),
  # ano de lançamento (release_year) e número de produtoras envolvidas
  # (production_companies_count)), deixando popularity/vote_average/vote_count
  # de fora por serem medidas pós-lançamento

  colunas_numericas = ["log_budget", "runtime", "release_year", "production_companies_count"]
  df_features = df[colunas_numericas].copy()

  return df_features


def tratar_categoricas(df, top_n_generos = 10, top_n_idiomas = 5):
  # tratar_categoricas: transforma genres e original_language em colunas numéricas
  # 0/1, mantendo só as categorias mais frequentes
  # retorna as colunas novas pra depois juntar com o x_numerico
  # top_n_generos = 10 e top_n_idiomas = 5: esses limites servem para aplicar
  # uma redução de dimensionalidade, impedindo que categorias muito raras ou
  # irrelevantes criem centenas de colunas desnecessárias na matriz final, o que
  # deixaria o cálculo do gradiente instável

  # genres foi salvo no csv como texto de uma lista de python, ast.literal_eval
  # converte esse texto de volta para uma lista de verdade
  genres = df["genres"].apply(ast.literal_eval)

  # explode: transforma cada elemento de uma lista em uma linha separada. ex: se
  # um filme pertencia a 3 gêneros, ele passa a ocupar 3 linhas no objeto
  # temporário gerado, replicando o índice original
  # value_counts: conta a frequência absoluta de cada gênero individual após explode
  generos_frequentes = genres.explode().value_counts().head(top_n_generos).index.tolist()

  dummies_genero = pd.DataFrame(index = df.index) # cria um df vazio
  for genero in generos_frequentes:
    # para cada gênero popular, cria uma coluna binária: 1 se aquele filme tem
    # esse gênero na lista, 0 caso contrário usando .astype(int) para transformar
    # os booleanos retornados pelo lambda lista: genero in lista, em 0 ou 1
    dummies_genero[f"genero_{genero}"] = genres.apply(lambda lista: genero in lista).astype(int)

  # calcula os idiomas mais recorrentes
  idiomas_frequentes = df["original_language"].value_counts().head(top_n_idiomas).index.tolist()

  # em seguida, o método .where() analisa a coluna de idiomas. para cada linha,
  # ele verifica se o idioma está presente na lista idiomas_frequentes através
  # do método .isin() e se estiver presente, ele mantém o valor original
  # caso contrário (para todas as dezenas de idiomas com poucas amostras), ele
  # substitui o valor pela string genérica "outro"
  # isso evita o problema de esparsidade de dados (sparse matrix)
  idioma_agrupado = df["original_language"].where(df["original_language"].isin(idiomas_frequentes), "outro")

  # cria uma coluna binária para cada categoria única encontrada na série
  # idioma_agrupado, adicionando o prefixo "idioma_" aos nomes das colunas
  dummies_idioma = pd.get_dummies(idioma_agrupado, prefix="idioma").astype(int)

  # pd.concat: junta (concatena) as tabelas dummies_genero e dummies_idioma
  # lado a lado. axis = 1 é pra fazer essa junção no eixo das colunas
  return pd.concat([dummies_genero, dummies_idioma], axis = 1)


def dividir_e_normalizar(x_numerico, x_categorico, y, tamanho_teste = 0.2, semente = 42):
    # dividir_e_normalizar: junta as features numéricas e categóricas em um x só,
    # separa treino/teste, e normaliza

    x = pd.concat([x_numerico, x_categorico], axis = 1)

    # y contínuo, random_state fixo garante que toda vez que rodar o código, o
    # split saia igual (importante pra reprodutibilidade dos experimentos)
    x_treino, x_teste, y_treino, y_teste = train_test_split(
        x.values, y, test_size = tamanho_teste, random_state = semente
    )

    # StandardScaler: subtrai a média e divide pelo desvio padrão de cada coluna,
    # deixando todas na mesma escala (média 0, desvio 1). ajusta (fit) só no
    # treino, e aplica (transform) no teste. usar o teste pra calcular a
    # média/desvio seria "espiar" dados que o modelo não deveria conhecer
    # ainda (vazamento entre treino e teste)
    scaler = StandardScaler()
    x_treino_norm = scaler.fit_transform(x_treino)
    x_teste_norm = scaler.transform(x_teste)

    return x_treino_norm, x_teste_norm, y_treino, y_teste, x.columns.tolist()
