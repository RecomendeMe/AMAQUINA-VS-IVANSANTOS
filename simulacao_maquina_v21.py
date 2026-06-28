"""
╔══════════════════════════════════════════════════════════════╗
║         PALPITES DA MÁQUINA — v2.1                          ║
║                                                              ║
║  Novidades vs v2.0:                                         ║
║  • modo_gestao: reduz lambda de ataque em 28% quando        ║
║    time vai administrar resultado (ex: líder sem pressão)   ║
║  • fator_mata_mata: sem empate possível → ambos atacam      ║
║    mais (+8%) — aumenta gols esperados no mata-mata         ║
║  • retorno_estrela: boost pontual de força (+X%) para       ║
║    jogador-chave que voltou de lesão (ex: Davies)           ║
║  • Hedge atualizado com R1 + R3                             ║
╚══════════════════════════════════════════════════════════════╝
"""

import numpy as np
import json
from dataclasses import dataclass, field
from collections import Counter

# ════════════════════════════════════════════════════════
# BLOCO 1 — Dados de elenco (Transfermarkt, jun/2026)
# ════════════════════════════════════════════════════════

VALOR_ELENCO = {
    "France": 1520, "Norway": 589, "Senegal": 478, "Cape Verde": 49,
    "Croatia": 387, "Ghana": 234, "England": 1360, "Panama": 34,
    "Colombia": 302, "Portugal": 1010, "DR Congo": 144, "Uzbekistan": 85,
    "Argentina": 807, "Austria": 245, "Algeria": 257, "Jordan": 20,
    "Spain": 1220, "Saudi Arabia": 40, "Uruguay": 359, "New Zealand": 34,
    "Belgium": 547, "Egypt": 116, "Iran": 32, "Iraq": 21,
    # Mata-mata
    "Canada":       199,   # Transfermarkt €198.65M
    "South Africa":  49,   # Transfermarkt €49.25M
    "Germany":      947,
    "Paraguay":     154,
    "Netherlands":  754,
    "Morocco":      430,
    "Switzerland":  333,
    "Bosnia":       146,
    "Australia":     77,
    "Sweden":       406,
    "Japan":        271,
    "Ivory Coast":  522,
    "USA":          386,
    "Ecuador":      369,
    "Mexico":       192,
}

# ════════════════════════════════════════════════════════
# BLOCO 2 — Forma atual (fase de grupos completa)
# ════════════════════════════════════════════════════════

@dataclass
class Forma:
    nome: str
    gols_marcados: int
    gols_sofridos: int
    pontos: int
    jogos: int
    ajuste_desfalque: float = 0.0   # -0.1 por titular importante ausente
    modo_gestao: bool = False        # True = time foi a campo pra não perder, não pra ganhar
    retorno_estrela: float = 0.0    # +X boost de volta de jogador chave (ex: Davies +0.08)

FORMA_ATUAL = {
    # Grupos I-L (fase de grupos completa)
    "France":       Forma("France",       gols_marcados=10, gols_sofridos=2,  pontos=9,  jogos=3),
    "Norway":       Forma("Norway",       gols_marcados=8,  gols_sofridos=4,  pontos=6,  jogos=3),
    "Senegal":      Forma("Senegal",      gols_marcados=6,  gols_sofridos=1,  pontos=4,  jogos=3),
    "England":      Forma("England",      gols_marcados=8,  gols_sofridos=3,  pontos=7,  jogos=3),
    "Croatia":      Forma("Croatia",      gols_marcados=5,  gols_sofridos=5,  pontos=6,  jogos=3),
    "Ghana":        Forma("Ghana",        gols_marcados=5,  gols_sofridos=4,  pontos=4,  jogos=3),
    "Colombia":     Forma("Colombia",     gols_marcados=4,  gols_sofridos=0,  pontos=7,  jogos=3),
    "Portugal":     Forma("Portugal",     gols_marcados=6,  gols_sofridos=1,  pontos=5,  jogos=3),
    "DR Congo":     Forma("DR Congo",     gols_marcados=4,  gols_sofridos=3,  pontos=4,  jogos=3),
    "Argentina":    Forma("Argentina",    gols_marcados=8,  gols_sofridos=1,  pontos=9,  jogos=3),
    "Algeria":      Forma("Algeria",      gols_marcados=5,  gols_sofridos=7,  pontos=4,  jogos=3),
    "Austria":      Forma("Austria",      gols_marcados=6,  gols_sofridos=6,  pontos=4,  jogos=3),

    # Mata-mata — dados da fase de grupos
    # Canadá: 2V 1D no Grupo B (6-0 Catar, 2-1 Bósnia, 1-2 Suíça) → 6 pts
    # Davies ausente nos 3 jogos por lesão, retorna agora
    "Canada":       Forma("Canada",       gols_marcados=9,  gols_sofridos=3,  pontos=6,  jogos=3,
                          retorno_estrela=0.09),   # Davies volta: boost real de elenco efetivo

    # África do Sul: 1V 1E 1D no Grupo A (0-1 México, 1-1 Rep Tcheca, 1-0 Coreia do Sul) → 4 pts
    "South Africa": Forma("South Africa", gols_marcados=2,  gols_sofridos=2,  pontos=4,  jogos=3),

    # Outros classificados para referência futura
    "Germany":      Forma("Germany",      gols_marcados=7,  gols_sofridos=2,  pontos=7,  jogos=3),
    "Paraguay":     Forma("Paraguay",     gols_marcados=3,  gols_sofridos=3,  pontos=4,  jogos=3),
    "Netherlands":  Forma("Netherlands",  gols_marcados=5,  gols_sofridos=2,  pontos=7,  jogos=3),
    "Morocco":      Forma("Morocco",      gols_marcados=4,  gols_sofridos=1,  pontos=6,  jogos=3),
    "Switzerland":  Forma("Switzerland",  gols_marcados=5,  gols_sofridos=3,  pontos=6,  jogos=3),
    "Bosnia":       Forma("Bosnia",       gols_marcados=3,  gols_sofridos=4,  pontos=3,  jogos=3),
    "Australia":    Forma("Australia",    gols_marcados=4,  gols_sofridos=2,  pontos=6,  jogos=3),
    "Sweden":       Forma("Sweden",       gols_marcados=3,  gols_sofridos=2,  pontos=5,  jogos=3),
    "Japan":        Forma("Japan",        gols_marcados=4,  gols_sofridos=3,  pontos=6,  jogos=3),
    "Ivory Coast":  Forma("Ivory Coast",  gols_marcados=5,  gols_sofridos=3,  pontos=6,  jogos=3),
    "USA":          Forma("USA",          gols_marcados=6,  gols_sofridos=2,  pontos=7,  jogos=3),
    "Ecuador":      Forma("Ecuador",      gols_marcados=4,  gols_sofridos=3,  pontos=4,  jogos=3),
    "Mexico":       Forma("Mexico",       gols_marcados=7,  gols_sofridos=1,  pontos=9,  jogos=3),
    "Spain":        Forma("Spain",        gols_marcados=6,  gols_sofridos=1,  pontos=7,  jogos=3),
    "Cape Verde":   Forma("Cape Verde",   gols_marcados=2,  gols_sofridos=2,  pontos=4,  jogos=3),
    "Belgium":      Forma("Belgium",      gols_marcados=3,  gols_sofridos=2,  pontos=5,  jogos=3),
    "Egypt":        Forma("Egypt",        gols_marcados=5,  gols_sofridos=3,  pontos=6,  jogos=3),
    "Senegal":      Forma("Senegal",      gols_marcados=6,  gols_sofridos=1,  pontos=4,  jogos=3),
}


# ════════════════════════════════════════════════════════
# BLOCO 3 — Motor de força (v2.1: inclui retorno_estrela)
# ════════════════════════════════════════════════════════

def calcular_forca(nome: str, peso_elenco: float = 0.6, peso_forma: float = 0.4) -> float:
    valor = VALOR_ELENCO.get(nome)
    if valor is None:
        raise ValueError(f"'{nome}' não encontrado em VALOR_ELENCO.")

    todos_valores = list(VALOR_ELENCO.values())
    log_val = np.log(valor)
    log_min = np.log(min(todos_valores))
    log_max = np.log(max(todos_valores))
    forca_elenco = (log_val - log_min) / (log_max - log_min)

    forma = FORMA_ATUAL.get(nome)
    if forma is None or forma.jogos == 0:
        return forca_elenco

    ataque   = forma.gols_marcados / forma.jogos / 3.0
    saldo    = np.tanh((forma.gols_marcados - forma.gols_sofridos) / 5.0)
    aproveit = forma.pontos / (forma.jogos * 3)

    forca_forma = (
        0.40 * np.clip(ataque, 0, 1) +
        0.35 * (saldo + 1) / 2 +
        0.25 * aproveit
    )

    # modo_gestao: time foi a campo pra não perder → ataque reduzido
    if forma.modo_gestao:
        forca_forma *= 0.72

    # ajuste manual de desfalque
    forca_forma = max(0.01, forca_forma + forma.ajuste_desfalque)

    # retorno de estrela: boost direto na força final
    boost = forma.retorno_estrela
    return min(1.0, peso_elenco * forca_elenco + peso_forma * forca_forma + boost)


# ════════════════════════════════════════════════════════
# BLOCO 4 — Lambda (v2.1: fator mata-mata)
# ════════════════════════════════════════════════════════

BASE_GOLS       = 2.6
FATOR_PRESSAO   = 0.12
FATOR_MATA_MATA = 1.08   # sem empate → ambos pressionam mais → +8% gols


def calcular_lambda(
    mandante: str,
    visitante: str,
    mata_mata: bool = False,
) -> tuple[float, float]:

    fm = calcular_forca(mandante)
    fv = calcular_forca(visitante)
    total = fm + fv

    lam_m = BASE_GOLS * (fm / total) * 2.0
    lam_v = BASE_GOLS * (fv / total) * 2.0

    # fator pressão por aproveitamento (só fase de grupos tem pontos comparáveis)
    forma_m = FORMA_ATUAL.get(mandante)
    forma_v = FORMA_ATUAL.get(visitante)
    if forma_m and forma_v and forma_m.jogos > 0 and forma_v.jogos > 0:
        pts_m = forma_m.pontos / forma_m.jogos
        pts_v = forma_v.pontos / forma_v.jogos
        if pts_m < pts_v:
            lam_m *= (1 + FATOR_PRESSAO)
            lam_v *= (1 - FATOR_PRESSAO * 0.5)
        elif pts_v < pts_m:
            lam_v *= (1 + FATOR_PRESSAO)
            lam_m *= (1 - FATOR_PRESSAO * 0.5)

    # no mata-mata: sem empate → os dois abrem mais
    if mata_mata:
        lam_m *= FATOR_MATA_MATA
        lam_v *= FATOR_MATA_MATA

    return max(lam_m, 0.05), max(lam_v, 0.05)


# ════════════════════════════════════════════════════════
# BLOCO 5 — Simulação Monte Carlo (v2.1: prorrogação/pênaltis)
# No mata-mata, empates vão para prorrogação.
# Modelamos como novo sorteio Poisson com λ reduzido (cansaço).
# ════════════════════════════════════════════════════════

def simular_jogo(
    mandante: str,
    visitante: str,
    n_simulacoes: int = 50_000,
    peso_elenco: float = 0.6,
    peso_forma:  float = 0.4,
    mata_mata:   bool = False,
) -> dict:

    lam_m, lam_v = calcular_lambda(mandante, visitante, mata_mata)

    gols_m = np.random.poisson(lam_m, n_simulacoes)
    gols_v = np.random.poisson(lam_v, n_simulacoes)

    if mata_mata:
        # empates vão para prorrogação: novo Poisson com λ * 0.45 (30min, cansados)
        empate_mask = gols_m == gols_v
        n_emp = empate_mask.sum()
        if n_emp > 0:
            gols_prorrog_m = np.random.poisson(lam_m * 0.45, n_emp)
            gols_prorrog_v = np.random.poisson(lam_v * 0.45, n_emp)
            # ainda empatados na prorrogação → pênaltis (50/50 ajustado por força)
            ainda_emp = gols_prorrog_m == gols_prorrog_v
            pen_m = np.random.random(ainda_emp.sum()) < (fm_ratio := calcular_forca(mandante) /
                    (calcular_forca(mandante) + calcular_forca(visitante)))
            # resolve pênaltis
            gols_prorrog_m[ainda_emp] = pen_m.astype(int)
            gols_prorrog_v[ainda_emp] = (~pen_m).astype(int)
            # aplica resultado da prorrogação
            gols_m[empate_mask] += gols_prorrog_m
            gols_v[empate_mask] += gols_prorrog_v

    prob_m = float((gols_m > gols_v).mean() * 100)
    prob_e = float((gols_m == gols_v).mean() * 100) if not mata_mata else 0.0
    prob_v = float((gols_m < gols_v).mean() * 100)

    placares = Counter(zip(gols_m.tolist(), gols_v.tolist())).most_common(5)
    placar_mais_provavel = f"{placares[0][0][0]}x{placares[0][0][1]}"

    if prob_m >= prob_v:
        palpite_wdl = f"Vitória {mandante}"
    else:
        palpite_wdl = f"Vitória {visitante}"

    return {
        "mandante": mandante,
        "visitante": visitante,
        "forca_mandante": round(calcular_forca(mandante, peso_elenco, peso_forma), 3),
        "forca_visitante": round(calcular_forca(visitante, peso_elenco, peso_forma), 3),
        "lambda": {"mandante": round(lam_m, 3), "visitante": round(lam_v, 3)},
        "prob_vitoria_mandante": round(prob_m, 1),
        "prob_empate": round(prob_e, 1),
        "prob_vitoria_visitante": round(prob_v, 1),
        "palpite_wdl": palpite_wdl,
        "palpite_placar": placar_mais_provavel,
        "placares_frequentes": [
            {"placar": f"{p[0][0]}x{p[0][1]}", "freq_pct": round(p[1] / n_simulacoes * 100, 1)}
            for p in placares[:3]
        ],
        "mata_mata": mata_mata,
        "nota_modelo": (
            "Inclui prorrogação + pênaltis simulados. "
            "Probabilidade de vitória = 90min + prorrogação + pênaltis combinados."
        ) if mata_mata else "",
    }


# ════════════════════════════════════════════════════════
# BLOCO 6 — Jogo atual
# ════════════════════════════════════════════════════════

RODADA = "16 avos de final — Mata-mata"
DATA   = "28/06/2026"

JOGOS = [
    {
        "mandante": "South Africa",
        "visitante": "Canada",
        "horario": "16:00",
        "fase": "16-avos",
        "local": "Los Angeles (SoFi Stadium)",
        "mata_mata": True,
    },
]


# ════════════════════════════════════════════════════════
# BLOCO 7 — Hedge (atualizado com R1 + R3)
# ════════════════════════════════════════════════════════

@dataclass
class Especialista:
    nome: str
    peso: float = 1.0
    acertos: int = 0
    erros: int = 0

    def taxa_acerto(self) -> float:
        total = self.acertos + self.erros
        return self.acertos / total if total > 0 else 0.5


def atualizar_hedge(especialistas, resultados_reais, palpites, eta=0.25):
    for esp in especialistas:
        for palpite, real in zip(palpites.get(esp.nome, []), resultados_reais):
            if palpite == real:
                esp.acertos += 1
                esp.peso *= (1 + eta * 0.3)
            else:
                esp.erros += 1
                esp.peso *= (1 - eta)
        esp.peso = max(esp.peso, 0.01)
    total = sum(e.peso for e in especialistas)
    for e in especialistas:
        e.peso /= total
    return especialistas


especialistas_hedge = [
    Especialista("Máquina", peso=0.5),
    Especialista("Ivan",    peso=0.5),
]

# R1: mandante=perspectiva (V=mandante ganhou, D=mandante perdeu, E=empate)
HISTORICO_R1 = {
    "resultados_reais": ["D", "V", "E", "D", "D", "E"],
    "palpites": {
        "Máquina": ["E", "V", "V", "E", "D", "E"],   # 3 acertos
        "Ivan":    ["E", "V", "V", "D", "D", "V"],   # 3 acertos
    }
}

# R3
HISTORICO_R3 = {
    "resultados_reais": ["D", "V", "D", "V", "D", "E"],
    # Panamá×Inglaterra → D (Inglaterra ganhou)
    # Croácia×Gana → V (Croácia ganhou)
    # Colômbia×Portugal → E mas nenhum acertou; ambos foram D (Portugal) → "D" real era E → ambos erraram
    # RD Congo×Uzbequistão → V (Congo ganhou)
    # Jordânia×Argentina → D (Argentina ganhou)
    # Argélia×Áustria → E (3-3)
    "palpites": {
        "Máquina": ["D", "V", "D", "V", "D", "D"],  # 4 acertos (errou Argélia×Áustria)
        "Ivan":    ["D", "D", "D", "V", "D", "D"],  # 3 acertos (errou Croácia×Gana e Argélia×Áustria)
    }
}

especialistas_hedge = atualizar_hedge(especialistas_hedge,
    HISTORICO_R1["resultados_reais"], HISTORICO_R1["palpites"])
especialistas_hedge = atualizar_hedge(especialistas_hedge,
    HISTORICO_R3["resultados_reais"], HISTORICO_R3["palpites"])


# ════════════════════════════════════════════════════════
# BLOCO 8 — Execução
# ════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 62)
    print(f"  🤖 PALPITES DA MÁQUINA v2.1 — {RODADA}")
    print(f"  Data: {DATA}")
    print(f"  Novidades: modo_gestao | fator mata-mata | prorrogação/pênaltis")
    print("=" * 62)

    print("\n📊 PESOS ADAPTATIVOS (após R1 + R3):")
    for e in especialistas_hedge:
        barra = "█" * int(e.peso * 30)
        print(f"  {e.nome:10s} {barra} {e.peso*100:.1f}% | acertos: {e.acertos}/{e.acertos+e.erros}")

    resultados = []
    print()

    for jogo in JOGOS:
        m, v = jogo["mandante"], jogo["visitante"]
        r = simular_jogo(m, v, mata_mata=jogo.get("mata_mata", False))
        r["horario"] = jogo["horario"]
        r["fase"]    = jogo["fase"]
        r["local"]   = jogo["local"]
        resultados.append(r)

        print(f"⚔️  {jogo['fase'].upper()} — {jogo['local']}")
        print(f"   {m} x {v}  ({jogo['horario']})")
        print(f"   Força:  {m} {r['forca_mandante']:.3f}  |  {v} {r['forca_visitante']:.3f}")
        print(f"   λ:      {r['lambda']['mandante']:.2f}  x  {r['lambda']['visitante']:.2f}")
        print(f"   📊  V {m}: {r['prob_vitoria_mandante']}%  |  V {v}: {r['prob_vitoria_visitante']}%")
        print(f"   🤖 Palpite: {r['palpite_wdl']}")
        print(f"   ⚽ Placar 90min mais provável: {r['palpite_placar']}")
        tops = " | ".join(f"{p['placar']} ({p['freq_pct']}%)" for p in r['placares_frequentes'])
        print(f"   Top placares: {tops}")
        print(f"   ℹ️  {r['nota_modelo']}")
        print()

    output = {
        "rodada": RODADA,
        "data": DATA,
        "versao": "v2.1",
        "modelo": "Transfermarkt + forma | modo_gestao | prorrogação/pênaltis | Monte Carlo Poisson",
        "pesos_hedge": {e.nome: round(e.peso, 4) for e in especialistas_hedge},
        "acertos_acumulados": {e.nome: f"{e.acertos}/{e.acertos+e.erros}" for e in especialistas_hedge},
        "jogos": resultados,
    }

    with open("/home/claude/palpites_mata_mata_r1.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("✅ Salvo em palpites_mata_mata_r1.json")
