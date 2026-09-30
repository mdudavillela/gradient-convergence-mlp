"""
coleta.py
---------
Módulo responsável por buscar dados de filmes diretamente na API oficial do
TMDB (The Movie Database) e salvar o resultado em um CSV local, com cache.

Conceitos usados (para eu lembrar):

- API REST: interface que permite a comunicação entre diferentes sistemas de software usando o protocolo HTTP da internet.
  A gente manda uma URL + parâmetros, e ele devolve dados em formato JSON (que emPython vira um dicionário, automaticamente, com resp.json()).

- Paginação: a API não devolve todos os filmes de uma vez, ela devolve em "páginas" de ~20 resultados. Por isso pedimos
  página 1, depois página 2, etc., até juntar a quantidade que queremos.

- Paralelismo (ThreadPoolExecutor): roda várias requisições ao mesmo tempo (em paralelo), daí o tempo total cai bastante, porque
  o computador não fica parado esperando a resposta de uma requisição para só então mandar a próxima.

- Cache: guarda o resultado de um processo caro (aqui, milhares de chamadas de API) em disco, para não precisar refazer o processo toda
  vez que o código roda de novo.

Como usar este módulo (dentro de um notebook):

    from coleta import coletar_dataset

    df = coletar_dataset(
        api_key=API_KEY,
        cache_path="../data/tmdb_raw.csv",
    )
"""

import os
import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# URL base da API. Todo endpoint que chamamos é "colado" depois dela,
# por exemplo: BASE_URL + "/movie/550" vira a URL completa do filme de id 550.
BASE_URL = "https://api.themoviedb.org/3"


def tmdb_get(api_key, endpoint, params=None, max_retries=3):
    """Faz uma requisição GET a um endpoint da API do TMDB e devolve a
    resposta já convertida de JSON para dicionário Python.

    Parâmetros
    ----------
    api_key : str
        Minha chave de autenticação do TMDB (v3 auth).
    endpoint : str
        O caminho do endpoint, ex: "/movie/550" ou "/discover/movie".
        Isso NÃO inclui o BASE_URL, essa função monta a URL completa.
    params : dict, opcional
        Parâmetros extras da requisição (ex: {"page": 2}). A api_key é
        adicionada automaticamente a esse dicionário, então não
        precisa passar ela aqui de novo.
    max_retries : int
        Quantas vezes tentar de novo se a API responder "429 Too Many
        Requests" (limite de requisições por segundo atingido) antes de
        desistir.
    """
    # dict(params or {}): se params for None, usa um dicionário vazio;
    # senão, faz uma CÓPIA do dicionário recebido. Fazemos uma cópia (em
    # vez de usar o dicionário original diretamente) para não modificar,
    # por acidente, um dicionário que quem chamou essa função ainda vai usar depois
    params = dict(params or {})
    params["api_key"] = api_key

    for _ in range(max_retries):
        resp = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=10)

        if resp.status_code == 200:
            # 200 = sucesso. resp.json() faz o parsing do corpo da resposta
            # (que vem como texto, no formato JSON) para um dict Python.
            return resp.json()

        if resp.status_code == 429:
            time.sleep(1.5)
            continue
        resp.raise_for_status()
    raise RuntimeError(f"Falha ao buscar {endpoint} após {max_retries} tentativas")


def coletar_ids_candidatos(api_key, n_pages=250, min_vote_count=100):
    """Percorre o endpoint /discover/movie, página por página, e retorna
    uma lista com os IDs dos filmes encontrados (sem duplicatas).

    Parâmetros
    ----------
    n_pages : int
        Quantas páginas buscar no máximo (cada página tem ~20 filmes).
        Ex: n_pages=250 tenta reunir até ~5000 filmes candidatos.
    min_vote_count : int
        Exige que o filme já tenha pelo menos esse número de votos no
        TMDB. Isso filtra filmes muito obscuros, cuja nota de público
        (vote_average) seria calculada em cima de poucas avaliações e,
        por isso, pouco confiável estatisticamente (o mesmo raciocínio
        que já usamos com o número mínimo de reviews da Steam).
    """
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
        # filtrado) — nesse caso, paramos o loop mais cedo com `break`,
        # mesmo que ainda não tenhamos chegado em n_pages.
        results = data.get("results", [])
        if not results:
            break

        # Para cada filme (m) na lista de resultados, pegamos só o campo
        # "id" — isso é uma "list comprehension", um jeito compacto em
        # Python de escrever um loop que constrói uma lista nova.
        # Equivale a:
        #   ids_da_pagina = []
        #   for m in results:
        #       ids_da_pagina.append(m["id"])
        ids.extend([m["id"] for m in results])

        if page % 50 == 0:  # só para mostrar progresso a cada 50 páginas
            print(f"  página {page}/{n_pages} | {len(ids)} ids coletados até agora")

    # list(dict.fromkeys(ids)): um truque comum em Python para remover
    # duplicados de uma lista MANTENDO a ordem original. Um dicionário não
    # pode ter chaves repetidas, então ao usar os ids como chaves e
    # converter de volta para lista, os duplicados somem automaticamente.
    # (Por que pode haver duplicados? Porque a lista de filmes populares
    # pode mudar de posição entre uma página e outra, e o mesmo filme
    # ocasionalmente aparecer em mais de uma página consultada.)
    return list(dict.fromkeys(ids))


def buscar_detalhes(api_key, movie_id):
    """Busca os detalhes completos de UM filme (endpoint /movie/{id}),
    que inclui campos que não vêm no /discover: budget, revenue, genres,
    production_companies etc.

    Retorna um dicionário "limpo", já só com os campos que nos interessam
    (em vez de devolver a resposta bruta da API, que tem muito mais
    informação do que vamos usar).

    Por que devolver None em caso de erro, em vez de deixar o erro
    "estourar"? Porque esta função vai ser chamada milhares de vezes (uma
    por filme). Se UM filme específico falhar (ex: foi removido do TMDB
    entre a etapa de descoberta e esta etapa), não queremos que isso
    derrube a coleta inteira — melhor pular esse filme e seguir com os
    outros. Quem chama esta função (a `coletar_dataset`) já sabe ignorar
    os `None` que aparecerem.
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
        # Captura QUALQUER tipo de erro (falha de rede, campo ausente,
        # filme não encontrado etc.) e devolve None em vez de travar tudo.
        return None


def coletar_dataset(api_key, cache_path, n_pages=250, min_vote_count=100,
                     max_workers=10, force_refresh=False):
    """Função principal deste módulo: orquestra a coleta completa
    (descobrir IDs -> buscar detalhes de cada um -> salvar em CSV) e
    cuida do cache, para não repetir esse trabalho pesado sem necessidade.

    Parâmetros
    ----------
    cache_path : str
        Caminho do arquivo CSV onde o resultado é salvo/lido, ex:
        "data/tmdb_raw.csv".
    max_workers : int
        Quantas requisições rodar SIMULTANEAMENTE durante a busca de
        detalhes. Um número maior acelera a coleta, mas também aumenta o
        risco de esbarrar no limite de requisições da API (erro 429).
        10 é um valor equilibrado.
    force_refresh : bool
        Se True, ignora um cache existente e coleta tudo de novo na API.
        Use isso só quando quiser dados atualizados de propósito.

    Retorna
    -------
    pandas.DataFrame
        Uma linha por filme coletado.
    """
    # Primeiro checamos se já existe um cache em disco. Se existir E não
    # pedirmos refresh forçado, nem chegamos a tocar na API — só lemos o
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

