# 🤖⚽ Placar da Máquina — Copa do Mundo 2026

**Cultura Eclética × RecomendeMe**  
Quadro de pós-jogo: Ivan Santos × Inteligência Artificial

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/NVIDIA-CUDA%2013-76b900)](https://developer.nvidia.com/cuda-toolkit)
[![Powered by Nosana](https://img.shields.io/badge/Compute-Nosana-purple)](https://nosana.io/)

---

## O que é

O **Placar da Máquina** é um quadro de análise preditiva exibido nos programas de pós-jogo da Copa do Mundo 2026 no canal **Cultura Eclética**, em parceria com a **RecomendeMe**. A cada rodada, um modelo treinado em GPU prevê os resultados das partidas e enfrenta os palpites do comentarista Ivan Santos. Quem acerta mais ao longo do torneio — o olho clínico ou a estatística?

---

## Infraestrutura de Compute

| Componente | Especificação |
|---|---|
| **GPU** | NVIDIA GeForce RTX 3090 (24 GB VRAM, CUDA 13) |
| **CPU** | AMD Ryzen 7 5800X 8-Core Processor |
| **RAM** | ~32 GB (31995 MB) |
| **Disco** | 705 GB |
| **OS** | Linux |
| **País do Node** | United Arab Emirates |
| **Download / Upload** | 94 Mbps / 76 Mbps |
| **Ping** | 6 ms |
| **Plataforma** | [Nosana](https://nosana.io/) — compute descentralizado |
| **Node ID** | `7dk2c92w2SeQahvcZs3ox9vtL9zuh4GC4mc8XHsgRrXh` |
| **Deployer** | `9K2GJQdPnYbomydfYZZaADJXNe8zcWMHf7Y2b1khGV9z` |
| **Ambiente** | Jupyter Notebook — container Kubernetes (`node.k8s.prd.nos.ci`) |

---

## Dataset

**"International Football Results from 1872 to 2024"**  
Autor: [Mart Jürisoo](https://github.com/martj42/international_results)  
Fonte: `https://raw.githubusercontent.com/martj42/international_results/master/results.csv`

| Atributo | Valor |
|---|---|
| Partidas carregadas (após limpeza) | **15.871** |
| Seleções distintas no dataset | **312** |
| Período coberto | 1872 — 2024 |
| Licença | Open data |

**Colunas utilizadas:** `date`, `home_team`, `away_team`, `home_score`, `away_score`, `tournament`

**Pré-processamento aplicado:**
- Remoção de linhas com `home_score` ou `away_score` ausentes (`dropna`)
- Remoção de partidas com placar negativo
- Filtragem opcional por data mínima (configurável no script)

---

## Arquitetura do Modelo

### Modelo de Embeddings de Força (`ForcaSelecaoModel`)

Cada seleção ganha dois vetores aprendidos de dimensão 8:

- **`ataque[i]`** — vetor de força ofensiva da seleção `i`
- **`defesa[j]`** — vetor de capacidade defensiva da seleção `j`

O `λ` (taxa de gols esperados, parâmetro da distribuição de Poisson) de cada confronto é calculado como:

```
λ_mandante = exp( ataque[i] · defesa[j] )
λ_visitante = exp( ataque[j] · defesa[i] )
```

O produto escalar entre o vetor de ataque de um time e o vetor de defesa do adversário captura a interação específica entre os dois estilos de jogo. O `exp()` garante que λ seja sempre positivo (requisito da Poisson).

### Função de Perda

**Poisson Negative Log-Likelihood (PoissonNLLLoss)**:

```
L = λ - k · log(λ)
```

Onde `k` é o número real de gols marcados. Esta é a função de perda correta para variáveis de contagem (gols), equivalente a maximizar a verossimilhança do modelo de Poisson sobre os dados observados. É estatisticamente mais adequada que erro quadrático médio (MSE) para esse tipo de dado.

### Treinamento

| Hiperparâmetro | Valor |
|---|---|
| Otimizador | Adam |
| Learning rate | 0.005 |
| Épocas | 300 |
| Gradient clipping | `clip_grad_norm_` (norma máx. = 1.0) |
| Dimensão do embedding | 8 |
| Device | CUDA (RTX 3090) |

**Convergência observada:**

```
Época   0 | Perda: 3.7213
Época  50 | Perda: 3.3669
Época 100 | Perda: 3.0949
Época 150 | Perda: 3.0232
Época 200 | Perda: 2.9940
Época 250 | Perda: 2.9778
✅ Treinamento concluído.
```

---

## Simulação Monte Carlo

Após o treinamento, os λ aprendidos alimentam uma simulação de Monte Carlo de **10.000 iterações por jogo**:

```python
gols_mandante = np.random.poisson(λ_mandante, 10_000)
gols_visitante = np.random.poisson(λ_visitante, 10_000)

prob_vitoria   = (gols_mandante > gols_visitante).mean() * 100
prob_empate    = (gols_mandante == gols_visitante).mean() * 100
prob_derrota   = (gols_mandante < gols_visitante).mean() * 100
```

O palpite final (placar) é o resultado mais frequente entre as 10.000 simulações. Os placares com maior frequência são exibidos no painel do programa.

---

## Sistema de Blend (Histórico + Forma Atual)

O modelo puro reflete a força histórica geral de cada seleção ao longo de 150 anos. Para capturar também o momento atual na Copa 2026, aplicamos uma **média ponderada**:

```
λ_final = α · λ_histórico + (1 - α) · λ_forma_atual
```

**Parâmetro padrão:** `α = 0.6` (60% histórico, 40% desempenho nesta Copa)

O `λ_forma_atual` é calculado a partir dos dados da fase de grupos (gols marcados/sofridos, aproveitamento de pontos, fator de pressão contextual — se a seleção precisa vencer para avançar).

Esta abordagem resolve um problema real identificado nos testes: seleções debutantes na Copa (ex: Cabo Verde) têm poucos dados históricos, o que distorce o embedding treinado. O blend reduz esse viés dando mais peso à performance recente.

---

## Output

O script gera um arquivo `palpites_maquina_blend.json` com a seguinte estrutura por jogo:

```json
{
  "mandante": "Norway",
  "visitante": "France",
  "lambda_mandante": 1.12,
  "lambda_visitante": 1.38,
  "prob_mandante": 28.4,
  "prob_empate": 31.2,
  "prob_visitante": 40.4,
  "palpite": "1x1",
  "placares_frequentes": [
    "1x1 → 14.3%",
    "0x1 → 13.1%",
    "1x2 → 10.8%"
  ],
  "modelo": "embeddings ataque/defesa treinados (PyTorch + GPU)"
}
```

---

## Estrutura do Repositório

```
placar-da-maquina/
├── treino_forca_selecoes.py   # Script principal (treinamento + simulação)
├── placar_da_maquina.html     # Página web com SEO para RecomendeMe
├── palpites_maquina.json      # Output do modelo puro
├── palpites_maquina_blend.json # Output do modelo com blend
├── results.csv                # Dataset (baixado automaticamente via urllib)
└── README.md
```

---

## Como Executar

### Requisitos

```bash
pip install torch pandas numpy
```

> PyTorch com suporte CUDA recomendado. O script detecta automaticamente via `torch.cuda.is_available()` e usa CPU como fallback.

### Execução

```bash
# O script baixa o dataset automaticamente (urllib, sem precisar do Kaggle)
python treino_forca_selecoes.py
```

Ou abra `treino_forca_selecoes.py` no Jupyter Notebook e execute célula por célula.

### Células do Notebook

| Célula | Função |
|---|---|
| 1 | Imports e detecção de device (CUDA/CPU) |
| 2 | Download e limpeza do dataset |
| 3 | Indexação das seleções |
| 4 | Definição do modelo (`ForcaSelecaoModel`) |
| 5 | Treinamento (Adam + gradient clipping) |
| 6 | Função `prever_jogo()` — modelo histórico puro |
| 7 | Loop pelos jogos da rodada + export JSON |
| 8 | Função `prever_jogo_blend()` — blend histórico + forma atual |

---

## Limitações Conhecidas

- O modelo não incorpora dados de lesões/suspensões em tempo real
- Seleções com poucos jogos históricos (ex: debutantes na Copa) geram embeddings menos confiáveis — mitigado pelo blend com `α` ajustável
- O λ captura força geral, não contexto tático específico (formação, estilo de pressão)
- A loss ainda convergia lentamente nas 300 épocas — ampliar para 800–1000 épocas melhora a calibração sem custo de tempo relevante na RTX 3090

---

## Créditos

| Entidade | Papel |
|---|---|
| [RecomendeMe](https://recomendeme.com.br) | Desenvolvimento do modelo e infraestrutura técnica |
| [Cultura Eclética](https://youtube.com/@culturaecletica) | Apresentação — Ivan Santos |
| [Nosana](https://nosana.io/) | Compute descentralizado (GPU NVIDIA RTX 3090) |
| [Mart Jürisoo](https://github.com/martj42/international_results) | Dataset histórico de futebol internacional |

---

## Licença

MIT License — open source, livre para uso e modificação com atribuição.

---

*Powered by NVIDIA RTX 3090 · PyTorch · Nosana Compute · Open Data*
