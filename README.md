# 🤖⚽ IVAN VS A MÁQUINA — Copa do Mundo 2026: Powered By Nvidia

**Cultura Eclética × RecomendeMe**  
Quadro de pós-jogo: Ivan Santos × Inteligência Artificial

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Monte%20Carlo-013243)](https://numpy.org/)
[![Transfermarkt](https://img.shields.io/badge/Dados-Transfermarkt-1a6cfa)](https://transfermarkt.com/)

---

## O que é

O **Placar da Máquina** é um quadro de análise preditiva exibido nos programas de pós-jogo da Copa do Mundo 2026 no canal **Cultura Eclética**, em parceria com a **RecomendeMe**. A cada rodada, um modelo prevê os resultados das partidas e enfrenta os palpites do comentarista Ivan Santos. Quem acerta mais ao longo do torneio — o olho clínico ou a estatística?

---

## Histórico de versões

### v1 — Dixon-Coles com embeddings neurais (Rodada 1)

O modelo original treinava embeddings de ataque e defesa por seleção via produto escalar, estilo Dixon-Coles, sobre 15.871 partidas históricas desde 1872. Rodava em GPU (NVIDIA RTX 3090 via Nosana) com PyTorch.

**Problema identificado:** o treinamento apresentou `loss: nan` desde a época 0 na execução de produção, resultando em lambdas inválidos. Além disso, o embedding histórico mistura gerações de jogadores e estilos de décadas distintas — a França de 1998 e a França de 2026 compartilham o escudo, não o elenco. O caso Cabo Verde (0–0 com Arábia Saudita) ilustrou o problema: seleções debutantes com pouuco histórico geram embeddings ruidosos independente do gradient clipping.

**Resultado:** 3 acertos em 6 jogos — empatado com Ivan Santos (3/6).

### v2 — Transfermarkt + Forma atual (Rodada 3 em diante) ✅ atual

Arquitetura completamente reescrita. Sem PyTorch, sem GPU, sem histórico de 150 anos. O modelo passa a raciocinar como Ivan raciocina: **elenco atual + momento da Copa**.

---

## Arquitetura v2

### Fonte de força

Dois componentes combinados com pesos configuráveis:

**1. Valor de elenco (Transfermarkt)**  
Valor de mercado total do elenco convocado, em milhões de euros. Data de referência: junho/2026.

```
França     €1.520M    Inglaterra  €1.360M    Portugal    €1.010M
Argentina    €807M    Croácia       €387M    Colômbia      €302M
Argélia      €257M    Áustria       €245M    Gana          €234M
RD Congo     €144M    Uzbequistão    €85M    Panamá         €34M
Jordânia      €20M    ...
```

O valor é normalizado em escala logarítmica para suavizar diferenças extremas (Argentina 807M vs Jordânia 20M) sem apagar a hierarquia de qualidade.

**2. Forma atual nesta Copa**  
Calculada a partir dos dados da fase de grupos em andamento: gols marcados, gols sofridos, aproveitamento de pontos e fator de pressão contextual (quem precisa do resultado ataca mais).

```python
forca_forma = (
    0.40 * ataque_medio +       # gols/jogo normalizado
    0.35 * saldo_tanh +         # saldo gols via tanh
    0.25 * aproveitamento       # pts / (jogos * 3)
)
```

**Combinação final:**

```
forca = 0.6 * forca_elenco + 0.4 * forca_forma
```

### Cálculo de lambda

```python
lambda_mandante  = BASE_GOLS * (forca_m / (forca_m + forca_v)) * 2.0
lambda_visitante = BASE_GOLS * (forca_v / (forca_m + forca_v)) * 2.0
```

Com `BASE_GOLS = 2.6` (média de gols/jogo em Copas recentes). Um fator de pressão de ±12% é aplicado quando há diferença de aproveitamento entre os times.

### Simulação Monte Carlo

```python
gols_m = np.random.poisson(lambda_mandante,  50_000)
gols_v = np.random.poisson(lambda_visitante, 50_000)

prob_vitoria = (gols_m > gols_v).mean() * 100
prob_empate  = (gols_m == gols_v).mean() * 100
prob_derrota = (gols_m < gols_v).mean() * 100
```

50.000 iterações por jogo. Tempo de execução: < 1 segundo em CPU comum. **GPU não é necessária.**

---

## Sistema Hedge (aprendizado adaptativo)

A partir da Rodada 2, Ivan e a Máquina operam como **especialistas competidores** dentro de um sistema de pesos multiplicativos (algoritmo Hedge / Multiplicative Weights):

```python
# quem errou perde peso; quem acertou mantém
peso *= (1 - eta)   # erro
peso *= (1 + eta * 0.3)  # acerto

# normalizado para somar 1 ao final de cada rodada
```

**Parâmetro:** `eta = 0.25` (taxa de aprendizado suave — evita oscilação com poucos jogos).

**Estado após Rodada 1:**

| Especialista | Peso | Acertos |
|---|---|---|
| Ivan Santos | **58.9%** | 3/6 |
| A Máquina   | 41.1% | 3/6 |

> O mesmo número de acertos gera pesos diferentes porque os jogos acertados/errados se sobrepõem de formas distintas ao longo das iterações. Ivan acertou Uruguai x Espanha (que a Máquina errou), o que pesou mais.

O palpite combinado final é um voto ponderado pelos pesos correntes — se Ivan está com mais peso, o sistema confia mais nele automaticamente.

---

## Output

O script gera `palpites_maquina_v2.json` com a seguinte estrutura:

```json
{
  "rodada": "3ª rodada — Fase de Grupos",
  "data": "27-28/06/2026",
  "modelo": "v2.0 — Transfermarkt + forma atual | sem histórico | Monte Carlo Poisson",
  "pesos_hedge": {
    "Máquina": 0.411,
    "Ivan": 0.589
  },
  "jogos": [
    {
      "mandante": "Panama",
      "visitante": "England",
      "forca_mandante": 0.159,
      "forca_visitante": 0.919,
      "lambda": { "mandante": 0.86, "visitante": 4.17 },
      "prob_vitoria_mandante": 3.2,
      "prob_empate": 6.1,
      "prob_vitoria_visitante": 90.7,
      "palpite_wdl": "Vitória England",
      "palpite_placar": "0x4",
      "placares_frequentes": [
        { "placar": "0x4", "freq_pct": 8.2 },
        { "placar": "0x3", "freq_pct": 7.8 },
        { "placar": "1x4", "freq_pct": 7.3 }
      ]
    }
  ]
}
```

---

## Estrutura do Repositório

```
placar-da-maquina/
├── simulacao_maquina_v2.py       # Script principal v2 (atual)
├── treino_forca_selecoes.py      # Script v1 — Dixon-Coles + GPU (legado)
├── placar_da_maquina.html        # Página web (RecomendeMe) — consome o JSON
├── palpites_maquina_v2.json      # Output v2 (rodada atual)
├── palpites_maquina_blend.json   # Output v1 — fallback legado
└── README.md
```

---

## Como Executar

### Requisitos

```bash
pip install numpy
```

Sem PyTorch. Sem CUDA. Sem dependências pesadas.

### Execução

```bash
python simulacao_maquina_v2.py
```

### Atualizar para nova rodada

1. Edite `FORMA_ATUAL` com gols/pontos atualizados de cada seleção
2. Edite `JOGOS` com os confrontos da nova rodada
3. Adicione um bloco `HISTORICO_RODADA_N` com resultados reais + palpites do Ivan
4. Execute — os pesos Hedge se atualizam automaticamente
5. Suba o `palpites_maquina_v2.json` gerado na mesma pasta do HTML

---

## Limitações conhecidas

- Valor de mercado (Transfermarkt) tem viés de liga: jogadores de Premier League e La Liga tendem a ser supervalorizados relativamente a ligas menores, o que pode reproduzir o efeito Cabo Verde em direção inversa.
- O dado de elenco tem defasagem temporal — não reflete lesões ou suspensões de última hora. Um campo `ajuste_desfalque` manual está disponível no dataclass `Forma` para correção pontual.
- Com poucos jogos por Copa, os pesos Hedge oscilam mais do que em amostras maiores. A taxa `eta = 0.25` foi escolhida para suavizar isso.
- O modelo não captura contexto tático (formação, estilo de pressão, histórico de confronto direto).

---

## Créditos

| Entidade | Papel |
|---|---|
| [RecomendeMe](https://recomendeme.com.br) | Desenvolvimento do modelo e infraestrutura técnica |
| [Cultura Eclética](https://youtube.com/@culturaecletica) | Apresentação — Ivan Santos |
| [Transfermarkt](https://transfermarkt.com/) | Dados de valor de elenco (jun/2026) |
| [Mart Jürisoo](https://github.com/martj42/international_results) | Dataset histórico v1 |
| [Nosana](https://nosana.io/) | Compute descentralizado v1 (GPU NVIDIA RTX 3090) |

---

## Licença

MIT License — open source, livre para uso e modificação com atribuição.

---

*v2.0 · Monte Carlo Poisson · Transfermarkt · CPU · < 1s por rodada*
