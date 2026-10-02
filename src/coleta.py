"""
coleta.py
---------
Módulo responsável por buscar dados de filmes diretamente na API oficial do
TMDB (The Movie Database) e salvar o resultado em um CSV local, com cache. Como?
TMDB -> descobrir quais filmes existem -> pegar os IDs (para cada ID: buscar os detalhes) -> juntar tudo -> transformar em tabela -> salvar csv

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

url_base = "https://api.themoviedb.org/3"
# guarda a parte principal da URL da API, ou seja, guarda algo que será usado várias vezes em uma varíavel
# em vez de escrever, por exemplo, https://api.themoviedb.org/3/movie/550 toda vez, podemos escrever url_base + "/movie/550"

def tmdb_get(api_key, endpoint, params = None, max_retries = 3): #criando uma função
# tmdb_get: função que descobre quais filmes existem no dataset
# params: filtros
  
    params = dict(params or {})
    params["api_key"] = api_key # dando o nome de "api_key" para a api_key (às vezes o óbvio precisa ser dito)

    for _ in range(max_retries): # usamos _ em vez de i porque só queremos que o loop se repita X vezes, mas o número da iteração atual não importa
        resposta_api = requests.get(f"{url_base}{endpoint}", params = params, timeout = 10) # f" " junta duas varáveis; requests.get: busca/consulta dodos na API
        if resposta_api.status_code == 200: # 200 significa sucesso no padrão internacional criado pelos inventores do world wide web
            return resposta_api.json() # retorna o parsing (conversão estrutural), que lê os dados em JSON e transforma em um dicionário do python
        if resposta_api.status_code == 429: # 429 significa muitas requisições
            time.sleep(1.5) # suspende temporariamente a execução da thread atual pra dar tempo de renovar a cota de requisições
            continue
        resposta_api.raise_for_status() # analisa o código (nesse caso, 200 ou 429), se for 200 é vida que segue, se for 429 ele para e fala qual erro ocorreu
    raise RuntimeError(f"Falha ao buscar {endpoint} após {max_retries} tentativas") # para a execução e lança uma classe de erro genérica do python (RuntimeError)


def coletar_ids_candidatos(api_key, n_pages = 250, min_vote_count = 100): #criando outra função
  # coletar_ids_candidatos: função que faz a paginação (lista bonitinha sem repetições) e extração em lotes (chunks?) dos IDs
    ids = []
    for page in range(1, n_pages + 1):
        data = tmdb_get(api_key, "/discover/movie", {
            "sort_by": "popularity.desc",   # filmes mais populares primeiro
            "vote_count.gte": min_vote_count,
            "include_adult": "false",
            "page": page,
        })

        # A API devolve um dicionário com uma chave "results", que é a
        # lista de filmes daquela página. Se vier vazia, significa que
        # acabaram os filmes disponíveis (chegamos ao fim do catálogo
        # filtrado), nesse caso, paramos o loop mais cedo com `break`,
        # mesmo que ainda não tenhamos chegado em n_pages.
        results = data.get("results", [])
        if not results:
            break

        # Para cada filme (m) na lista de resultados, pegamos só o campo "id"
        # Equivale a:
        #   ids_da_pagina = []
        #   for m in results:
        #       ids_da_pagina.append(m["id"])
        ids.extend([m["id"] for m in results])

        if page % 50 == 0:  # só para mostrar progresso a cada 50 páginas
            print(f"  página {page}/{n_pages} | {len(ids)} ids coletados até agora")

    # list(dict.fromkeys(ids)): para remover duplicados de uma lista mantendo a ordem original.
    return list(dict.fromkeys(ids))


def buscar_detalhes(api_key, movie_id):
    """Busca os detalhes completos de UM filme (endpoint /movie/{id}),
    que inclui campos que não vêm no /discover: budget, revenue, genres,
    production_companies etc.

    Retorna um dicionário "limpo", já só com os campos que nos interessam
    (em vez de devolver a resposta bruta da API, que tem muito mais
    informação do que vamos usar).
    
    """
    try:
        d = tmdb_get(api_key, f"/movie/{movie_id}")
        return {
            "id": d.get("id"),
            "title": d.get("title"),
            "budget": d.get("budget"),
            "revenue": d.get("revenue"),
            "runtime": d.get("runtime"),
            "popularity": d.get("popularity"),
            "vote_average": d.get("vote_average"),
            "vote_count": d.get("vote_count"),
            "release_date": d.get("release_date"),
            "original_language": d.get("original_language"),
            # 'genres' vem como uma lista de dicionários, tipo
            # [{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}].
            # Aqui extraímos só o "name" de cada um, guardando como lista
            # de strings: ["Action", "Adventure"].
            "genres": [g["name"] for g in (d.get("genres") or [])],
            "production_companies_count": len(d.get("production_companies") or []),
        }
    except Exception:
        # Captura qualquer tipo de erro (falha de rede, campo ausente,
        # filme não encontrado etc.) e devolve None em vez de travar tudo.
        return None


def coletar_dataset(api_key, cache_path, n_pages = 250, min_vote_count = 100,
                     max_workers = 10, force_refresh = False):
    """Função principal desa parte: orquestra a coleta completa
    (descobrir IDs -> buscar detalhes de cada um -> salvar em CSV) e
    cuida do cache, para não repetir esse trabalho pesado sem necessidade.

    Parâmetros
    ----------
    cache_path : str
        Caminho do arquivo CSV onde o resultado é salvo/lido, ex:
        "data/tmdb_raw.csv".
    max_workers : int
        Quantas requisições rodar simultaneamente durante a busca de
        detalhes. Um número maior acelera a coleta, mas também aumenta o
        risco de esbarrar no limite de requisições da API (erro 429).
        10 é um valor equilibrado.
    force_refresh : bool
        Se True, ignora um cache existente e coleta tudo de novo na API.
        USAR ISSO SOMENTE QUANDO QUISER DADOS ATUALIZADOS DE PROPÓSITO (DEPOIS DE ACABAR A IC)!!!!!!!!!!

    Retorna
    -------
    pandas.DataFrame
        Uma linha por filme coletado.
    """
    # Primeiro checamos se já existe um cache em disco. Se existir e não
    # pedirmos refresh forçado, nem chegamos a tocar na API, só lemos o
    # CSV, que é quase instantâneo comparado a milhares de requisições.
    if os.path.exists(cache_path) and not force_refresh:
        print(f"Cache encontrado em '{cache_path}' — carregando em vez de recoletar.")
        return pd.read_csv(cache_path)

    print("Etapa 1a: descobrindo IDs de filmes candidatos...")
    ids = coletar_ids_candidatos(api_key, n_pages, min_vote_count)
    print(f"Total de IDs candidatos: {len(ids)}")

    print("\nEtapa 1b: buscando detalhes de cada filme (em paralelo)...")
    registros = []

    # ThreadPoolExecutor cria um "time" de até `max_workers` threads, que
    # ficam pegando tarefas da fila e executando ao mesmo tempo.
    #
    # `executor.submit(buscar_detalhes, api_key, mid)` agenda UMA chamada
    # de buscar_detalhes(api_key, mid) para ser executada assim que houver
    # uma thread livre, e devolve imediatamente um objeto "future" (uma
    # espécie de "promessa": o resultado ainda não existe, mas vai
    # existir quando a chamada terminar).
    #
    # O dicionário {future: mid for mid in ids} guarda, para cada
    # "promessa" criada, a qual id de filme ela corresponde — útil caso a
    # gente quisesse saber qual filme deu erro (aqui não usamos essa
    # informação diretamente, mas é comum guardar por precaução).
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(buscar_detalhes, api_key, mid): mid for mid in ids}

        # as_completed(futures) devolve cada "promessa" NA ORDEM EM QUE
        # ELA TERMINA (não na ordem em que foi criada) — ou seja, à medida
        # que cada requisição individual vai respondendo, processamos o
        # resultado dela, sem esperar as outras.
        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()   # pega o retorno de buscar_detalhes(...)
            if res is not None:     # ignora os que falharam (None)
                registros.append(res)
            if i % 500 == 0:        # só para mostrar progresso a cada 500 filmes
                print(f"  {i}/{len(ids)} filmes processados...")

    # Transforma a lista de dicionários (um por filme) em uma tabela
    # (DataFrame) do pandas — cada dicionário vira uma linha, e as chaves
    # dos dicionários viram as colunas.
    df = pd.DataFrame(registros)

    # Garante que a pasta de destino existe antes de tentar salvar o
    # arquivo nela (ex: se cache_path="data/tmdb_raw.csv" e a pasta
    # "data/" ainda não existe, isso a cria).
    pasta = os.path.dirname(cache_path)
    if pasta:
        os.makedirs(pasta, exist_ok=True)

    df.to_csv(cache_path, index=False)
    print(f"\nColeta concluída e salva em '{cache_path}'. Shape: {df.shape}")
    return df


if __name__ == "__main__":
    # Este bloco só roda quando o arquivo é executado DIRETAMENTE pelo
    # terminal (ex: `python coleta.py`), NÃO roda quando o arquivo é
    # importado por outro código (`from coleta import coletar_dataset`).
    import getpass
    api_key = getpass.getpass("Cole sua API Key (v3 auth) do TMDB: ")
    df = coletar_dataset(api_key, cache_path="data/tmdb_raw.csv")
    print(df.head())

