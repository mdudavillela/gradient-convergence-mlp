"""
coleta.py
---------
Módulo responsável por buscar dados de filmes diretamente na API oficial do
TMDB (The Movie Database) e salvar o resultado em um CSV local, com cache. 

Como?
TMDB -> descobrir quais filmes existem -> pegar os IDs (para cada ID: buscar os detalhes) -> juntar tudo -> transformar em tabela -> salvar csv
Funções criadas:
BASE_URL (variável global) -> tmdb_get -> coletar_ids_candidatos (para cada ID: buscar_detalhes) -> coletar_dataset -> df (variável) -> df.to_csv

Conceitos usados (para eu lembrar):

- API REST: interface que permite a comunicação entre diferentes sistemas de software usando o protocolo HTTP da internet.
  A gente manda uma URL + parâmetros, e ele devolve dados em formato JSON (que emPython vira um dicionário, automaticamente, com resp.json()).

- Paginação: a API não devolve todos os filmes de uma vez, ela devolve em "páginas" de ~20 resultados. Por isso pedimos
  página 1, depois página 2, etc., até juntar a quantidade que queremos.

- Paralelismo (ThreadPoolExecutor): roda várias requisições ao mesmo tempo (em paralelo), daí o tempo total cai bastante, porque
  o computador não fica parado esperando a resposta de uma requisição para só então mandar a próxima.

- Cache: guarda o resultado de um processo caro (aqui, milhares de chamadas de API) em disco, para não precisar refazer o processo toda
  vez que o código roda de novo.
  
- No Python, a maioria das bibliotecas usadas para "ciência de dados" são equivalentes aos pacotes do tidyverse do R.

Como usar este módulo (dentro de um notebook):

    from coleta import coletar_dataset

    df = coletar_dataset(
        api_key = API_KEY,
        cache_path = "../data/tmdb_raw.csv",
    )
"""
# os: biblioteca que permite ao python interagir como sistema operacional (operational system), por ex.: verificar se
# um arquivo existe; criar pastas; descobrir caminhos; trabalhar com arquivos e diretórios.
# equivalência no R: base R e fs do tidyverse
import os

# time: serve para lidar com funções de tempo de baixo nível, como medir a duração de códigos, pausar a execução e converter formatos de data e hora.
# equivalência no R: lubridate do tidyverse
import time

# requests: serve para fazer requisições HTTP, ou seja, permite que o python converse com sites e APIs.
# equivalência no R: httpr2 e jsonlite (para ler arquivos JSON)
import requests

# pandas: é a principal ferramenta do python para análise, manipulação e limpeza de dados.
# equivalência no R: dplyer, tidyr, conceito de tibble e o pipe %>%
import pandas as pd

# serve para fazer programação concorrente (paralelismo/multithreading), ou seja, ele permite que 
# o python execute várias tarefas ao mesmo tempo (em paralelo), em vez de esperar uma terminar para começar a outra.
# ThreadPoolExecutor: cria um "pool" (grupo) de linhas de execução (threads). você define quantas tarefas quer rodar em paralelo (ex: 5 por vez).
# as_completed: controla a execução e avisa o código assim que qualquer uma das tarefas paralelas terminar, permitindo pegar
# o resultado imediatamente, sem ter que esperar a ordem exata de envio.
# equivalência no R: não conheço
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://api.themoviedb.org/3"
# guarda a parte principal da URL da API, ou seja, guarda algo que será usado várias vezes em uma varíavel
# em vez de escrever, por exemplo, https://api.themoviedb.org/3/movie/550 toda vez, podemos escrever BASE_URL + "/movie/550"

def tmdb_get(api_key, caminho_api, filtros = None, max_retries = 3): #criando uma função
# tmdb_get: função que faz uma chamada GET à API
    filtros = dict(filtros or {})
    filtros["api_key"] = api_key # dando o nome de "api_key" para a api_key (às vezes o óbvio precisa ser dito)

    for _ in range(max_retries): # usamos _ em vez de i porque só queremos que o loop se repita X vezes, mas o número da iteração atual não importa
        resposta_api = requests.get(f"{BASE_URL}{caminho_api}", params = filtros, timeout = 10) # f" " junta duas varáveis; requests.get: busca/consulta dodos na API
        if resposta_api.status_code == 200: # 200 significa sucesso no padrão internacional criado pelos inventores do world wide web
            return resposta_api.json() # retorna o parsing (conversão estrutural), que lê os dados em JSON e transforma em um dicionário do python
        if resposta_api.status_code == 429: # 429 significa muitas requisições
            time.sleep(1.5) # suspende temporariamente a execução da thread atual pra dar tempo de renovar a cota de requisições
            continue
        resposta_api.raise_for_status() # analisa o código de status HTTP e lança a exceção que pode ser qualquer erro (menos 200 e 429 pq já foram verificados)
    raise RuntimeError(f"Falha ao buscar {caminho_api} após {max_retries} tentativas") # para a execução e lança uma classe de erro genérica do python (RuntimeError)


def coletar_ids_candidatos(api_key, n_paginas = 250, qtd_votos_min = 100): #criando outra função
  # coletar_ids_candidatos: função que faz a paginação (lista bonitinha sem repetições) e extração em lotes (chunks?) dos IDs
    ids = []
    for pagina in range(1, n_paginas + 1):
        dados = tmdb_get(api_key, "/discover/movie", {
            "sort_by": "popularity.desc",   # filmes mais populares primeiro (filtra em ordem decrescente)
            "vote_count.gte": qtd_votos_min, # quantidade maior ou igual (.gte) para qtd_votos_min
            "include_adult": "false",
            "page": pagina,
        })
      
        resultados = dados.get("results", [])
        if not resultados:
            break

        # para cada filme (movie, por isso m) na lista de resultados, pegamos só o campo "id"
        # Equivale a:
        #   ids_da_pagina = []
        #   for m in results:
        #       ids_da_pagina.append(m["id"])
        ids.extend([m["id"] for m in resultados]) # list comprehension: é uma forma curta e legível de criar uma nova lista baseada em outra sequência
        if pagina % 50 == 0:  # só para mostrar progresso a cada 50 páginas
            print(f"  página {pagina}/{n_paginas} | {len(ids)} ids coletados até agora")

    # list(dict.fromkeys(ids)): para remover duplicados de uma lista mantendo a ordem original
    return list(dict.fromkeys(ids))


def buscar_detalhes(api_key, movie_id):
  # buscar_detalhes: função que serve pra extrair e isolar dados específicos de um único filme a partir de seu número de identificação (movie_id)
  # try/except: é uma estrutura de segurança. o python tentará executar tudo o que está dentro do bloco try. se qualquer erro acontecer ali dentro,
  # o interpretador ignora o erro, salta imediatamente para o bloco except Exception: e executa o comando return None (retorna um valor nulo).
  # isso garante que o script continue rodando mesmo se um filme da lista der problema
    try:
        dic = tmdb_get(api_key, f"/movie/{movie_id}") # dicionário com o número do movie_id diretamente dentro do texto da url
        return {
            "id": dic.get("id"),
            "title": dic.get("title"),
            "budget": dic.get("budget"),
            "revenue": dic.get("revenue"),
            "runtime": dic.get("runtime"),
            "popularity": dic.get("popularity"),
            "vote_average": dic.get("vote_average"),
            "vote_count": dic.get("vote_count"),
            "release_date": dic.get("release_date"),
            "original_language": dic.get("original_language"),
            "genres": [g["name"] for g in (dic.get("genres") or [])], # list comprehension
            "production_companies_count": len(dic.get("production_companies") or []), # número total de produtoras envolvidas no filme
        }
    except Exception:
        return None


def coletar_dataset(api_key, cache_path, n_paginas = 250, qtd_votos_min = 100,
                     numero_threads = 10, atualizar_do_zero = False):
   # coletar_dataset: função que junta todas as outras funções anteriores para criar, gerenciar e salvar o dataset final
   # em um arquivo csv, utilizando paralelismo para acelerar o processo
                       
    if os.path.exists(cache_path) and not atualizar_do_zero:
        print(f"Cache encontrado em '{cache_path}' — carregando em vez de recoletar.")
        return pd.read_csv(cache_path)

    print("Etapa 1a: descobrindo IDs de filmes candidatos...")
    ids = coletar_ids_candidatos(api_key, n_paginas, qtd_votos_min) # faz as requisições de paginação e armazena os dados brutos na memória
    print(f"Total de IDs candidatos: {len(ids)}")

    print("\nEtapa 1b: buscando detalhes de cada filme (em paralelo)...")
    registros = []

    with ThreadPoolExecutor(max_workers = numero_threads) as executor: # o comando with garante que todas as threads sejam fechadas e limpas da memória corretamente quando o bloco terminar
        futures = {executor.submit(buscar_detalhes, api_key, mid): mid for mid in ids} # mid: movie id (lista de ids dos filmes)
        # future (promessa): um espaço reservado na memória que receberá o resultado real assim que a thread terminar de processar aquela requisição.
        # as_completed(futures) devolve cada "promessa" assim que a thread terminar de processar aquela requisição
        for i, future in enumerate(as_completed(futures), 1): # entrega os objetos future na ordem exata em que eles terminam de receber a resposta da internet
            res = future.result()   # pega o retorno de buscar_detalhes
            if res is not None:     # ignora os que falharam (None)
                registros.append(res)
            if i % 500 == 0:        # só para mostrar progresso a cada 500 filmes
                print(f"  {i}/{len(ids)} filmes processados...")

    df = pd.DataFrame(registros) # cria um dataframe no qual cada dicionário vira uma linha e as keys dos dicionários viram as colunas
    pasta = os.path.dirname(cache_path) # isola a string correspondente ao caminho da pasta onde o arquivo será gravado
    if pasta:
        os.makedirs(pasta, exist_ok = True) # garante que a pasta de destino existe antes de tentar salvar o arquivo nela, se não existe ela é criada

    df.to_csv(cache_path, index = False)
    print(f"\nColeta concluída e salva em '{cache_path}'. Shape: {df.shape}") # df.shape: retorna uma tupla indicando as dimensões exatas da tabela gerada
    return df


if __name__ == "__main__":
    # Este bloco só roda quando o arquivo é executado diretamente pelo terminal (ex: `python coleta.py`), NÃO roda quando o arquivo é
    # importado por outro código (`from coleta import coletar_dataset`)
    import getpass
    api_key = getpass.getpass("Cole sua API Key (v3 auth) do TMDB: ")
    df = coletar_dataset(api_key, cache_path="data/tmdb_raw.csv")
    print(df.head())

