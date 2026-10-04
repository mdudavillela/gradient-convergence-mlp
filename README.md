# Análise da convergência de métodos de gradiente na otimização de redes neurais Perceptron

Projeto de Iniciação Científica Voluntária (PICV)

**Por:** Maria Eduarda Villéla Silva

**Orientadora:** Petra Maria Bartmeyer

## Sobre o projeto

Este projeto investiga como diferentes métodos de otimização se comportam ao treinar redes neurais simples (Perceptrons). Otimização é o processo central por trás de qualquer modelo de aprendizado de máquina: é como o modelo ajusta seus parâmetros internos, repetidamente, até reduzir ao máximo o erro entre o que ele prevê e o que realmente aconteceu nos dados. O trabalho tem como objetivo mapear como propriedades algébricas, como o número de condição de uma matriz e não-convexidade, traduzem-se em desafios de treinamento em aprendizado de máquina.

Como primeira etapa prática do projeto, implementamos o Gradiente Descendente aplicado a um problema concreto: prever a bilheteria de filmes a partir de características gerais, como orçamento, duração e gênero. Os dados vêm direto da API oficial do TMDB (The Movie Database), e todo o processo de coleta, preparação e modelagem está documentado e reproduzível neste repositório.

## Estrutura do repositório

```
gradient-convergence-mlp/
├── data/
│   └── tmdb_raw.csv           # dados brutos coletados da API (cache, commitado uma única vez)
├── src/                       # src: source (código-fonte)
│   ├── coleta.py              # busca os dados na API do TMDB e salva em CSV, com cache
│   └── preprocessamento.py    # filtra, cria o alvo, seleciona features, normaliza, separa treino/teste
├── notebooks/
│   └── main.ipynb             # orquestra tudo: clona o repo, importa os módulos, roda cada etapa
└── README.md
```

## Como rodar

Este projeto roda no **Google Colab**, buscando os módulos direto deste repositório do GitHub.

1. **Abra o notebook** direto do GitHub, colando este link no navegador:
   `colab.research.google.com/github/mdudavillela/gradient-convergence-mlp/blob/main/notebooks/main.ipynb`

2. **Configure sua API key do TMDB** nos Secrets do Colab (uma única vez, vale para qualquer sessão futura):
   - Crie uma conta gratuita em [themoviedb.org](https://www.themoviedb.org/signup) e solicite uma API Key (v3 auth) em *Configurações → API*.
   - No Colab, clique no ícone de chave 🔑 na barra lateral → *Add new secret*.
   - Nome: `TMDB_API_KEY` — Valor: sua chave. Ative o *Notebook access*.

3. **Rode as células do `main.ipynb` em ordem**, de cima para baixo:
   - Clona este repositório para dentro da máquina temporária do Colab.
   - Importa os módulos de `src/`.
   - Carrega a API key.
   - Roda a coleta (`coletar_dataset`). Como `data/tmdb_raw.csv` já está commitado no repositório, isso carrega o cache instantaneamente, sem refazer as milhares de requisições à API.
   - Roda o pré-processamento (`preprocessamento.py`) até gerar os conjuntos de treino/teste normalizados.

> Se `data/tmdb_raw.csv` não existir (por exemplo, num fork do repositório), a primeira coleta pode levar entre 10 e 20 minutos, por causa do volume de requisições à API. Depois de rodar uma vez e commitar o CSV resultante, isso nunca mais precisa ser refeito.

## Status atual

- [x] Coleta de dados direto da API do TMDB, com cache (`src/coleta.py`)
- [x] Pré-processamento: filtragem, variável-alvo, features, normalização (`src/preprocessamento.py`)
- [ ] Implementação do Perceptron com Gradiente Descendente (`src/perceptron.py`)
