# Share Price Prediction API

## Introdução

Este projeto tem como objetivo a predição do valor de ações com base em séries temporais dos preços de fechamento dos papéis.

Ele conta com:

- Uma API REST com FastAPI 
- Uma rede neural artifical LSTM

## Sumário

- [Share Price Prediction API](#share-price-prediction-api)
  - [Introdução](#introdução)
  - [Sumário](#sumário)
  - [Pré-requisitos](#pré-requisitos)
  - [Executando e treinando](#executando-e-treinando)
  - [Estrutura](#estrutura)

## Pré-requisitos

Esta aplicação utiliza Docker para executar de forma simplificada e sem muitas configurações, para executar ela, você deve ter instalado em seu computador:

- [Docker](https://www.docker.com/)
- [docker-compose](https://docs.docker.com/compose/)

## Executando e treinando

Para executar esta API na sua máquina, somente digite `docker-compose up` em seu terminal, dentro da pasta deste projeto.

Feito isto, a aplicação está online por meio de dois containers, sendo eles:

- [Documentação e interação com a API](http://localhost:8000/docs)
- [Jupyter Notebook para treinar modelo](http://localhost:8888/lab/tree/notebook)

Caso deseje treinar um novo modelo, basta abrir o segundo link e abrir o notebook `train.ipynb`, por meio do qual o modelo será treinado. Note que o notebook não está carregado de código Python, ao invés disto, ele utiliza módulos Python que carregam dentro de si as lógicas para treinamento e obtenção de dados. Veja a próxima seção para mais detalhes da estrutura de arquivos.

Para utilizar os modelos treinados você deve utilizar o primeiro link, o qual conta com um [Swagger](https://swagger.io/) por meio do qual você poderá tanto visualizar uma documentação da API como interagir com ela de forma facilitada.

No conteúdo salvo deste repositório, já existirá um modelo para o papel `PETR4.SA`, que foi salvo no repositório por meio de [git lfs](https://git-lfs.com/), e a estrutura deste repositório foi pensada para permitir que outros modelos possam ser salvos aqui também, de modo a permitir melhor versionamento dos modelos.

Caso você tente consultar um papel para o qual o modelo não tenha sido treinado, você receberá um 404.

## Estrutura

A estrututa de arquivos desta aplicação conta com:

- `lib/train/`: contém os utilitários de pré-processamento, criação dos datasets e treinamento;
- `src/api/`: define os endpoints REST;
- `models/`: pasta onde os modelos, scalers e datasets são salvos por símbolo;
- `notebooks/train.ipynb`: notebook usado para treinar modelos.

A pasta models, por sua vez, espera que os arquivos sejam salvos da seguinte forma:

- `models/{symbol}_model.h5`
- `models/{symbol}_scaler.pkl`
- `models/{symbol}_dataset.pkl`
