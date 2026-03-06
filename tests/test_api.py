"""
test_api.py — Testes da API de Previsão de Ciclo de Vendas
===========================================================
Pré-requisito: API rodando em localhost:8000
    uvicorn src.api:app --reload --port 8000

Executar:
    # Modo manual (com prints detalhados):
    python tests/test_api.py

    # Modo pytest:
    pytest tests/test_api.py -v
"""

import json
import requests
import pytest

BASE_URL = "http://localhost:8000"

# ── Payloads de teste ─────────────────────────────────────────────────────────

# Lead de Stand → ciclo rápido esperado (~15 dias)
VENDA_STAND = {
    "id_venda"      : 501,
    "data_venda"    : "2025-03-15",
    "empreendimento": "Grand Maré",
    "tipologia"     : "2 Quartos",
    "area_m2"       : 72.5,
    "valor_tabela"  : 650000.0,
    "forma_pagamento": "Financiamento",
    "corretor"      : "Juliana Souza",
    "imobiliaria"   : "Prime Realty",
    "origem_lead"   : "Stand",
}

# Lead de Instagram → ciclo longo esperado (~69 dias)
VENDA_INSTAGRAM = {
    "id_venda"      : 502,
    "data_venda"    : "2025-06-01",
    "empreendimento": "Villa Portofino",
    "tipologia"     : "Cobertura",
    "area_m2"       : 180.0,
    "valor_tabela"  : 4500000.0,
    "forma_pagamento": "À vista",
    "corretor"      : "Bruno Santos",
    "imobiliaria"   : "Oceano Imobiliária",
    "origem_lead"   : "Instagram",
}

BATCH_PAYLOAD = {"vendas": [VENDA_STAND, VENDA_INSTAGRAM]}


# ── Testes manuais ────────────────────────────────────────────────────────────

def test_health():
    print("\n[1/5] GET /health")
    r = requests.get(f"{BASE_URL}/health")
    print(f"  Status : {r.status_code}")
    print(f"  Body   : {json.dumps(r.json(), indent=2, ensure_ascii=False)}")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    print("  ✅ OK")


def test_predict_stand():
    print("\n[2/5] POST /predict  (Stand → ciclo rápido)")
    r = requests.post(f"{BASE_URL}/predict", json=VENDA_STAND)
    data = r.json()
    print(f"  Status        : {r.status_code}")
    print(f"  Dias previsto : {data.get('dias_previsto')} dias")
    print(f"  Intervalo     : {data.get('intervalo_inf')} – {data.get('intervalo_sup')} dias")
    print(f"  Canal         : {data.get('canal')}")
    assert r.status_code == 200
    assert data["dias_previsto"] > 0
    print("  ✅ OK")


def test_predict_instagram():
    print("\n[3/5] POST /predict  (Instagram → ciclo longo)")
    r = requests.post(f"{BASE_URL}/predict", json=VENDA_INSTAGRAM)
    data = r.json()
    print(f"  Status        : {r.status_code}")
    print(f"  Dias previsto : {data.get('dias_previsto')} dias")
    print(f"  Intervalo     : {data.get('intervalo_inf')} – {data.get('intervalo_sup')} dias")
    assert r.status_code == 200
    # Stand deve ser mais rápido que Instagram
    r_stand = requests.post(f"{BASE_URL}/predict", json=VENDA_STAND).json()
    assert data["dias_previsto"] > r_stand["dias_previsto"], \
        "Instagram deveria ter ciclo mais longo que Stand"
    print("  ✅ OK — Instagram mais lento que Stand (como esperado)")


def test_predict_batch():
    print("\n[4/5] POST /predict/batch")
    r = requests.post(f"{BASE_URL}/predict/batch", json=BATCH_PAYLOAD)
    data = r.json()
    print(f"  Status : {r.status_code}")
    print(f"  Total  : {data['total']}")
    for p in data["previsoes"]:
        print(f"  id={p['id_venda']} | canal={p['canal']:<12} | "
              f"dias={p['dias_previsto']:>3}  [{p['intervalo_inf']}–{p['intervalo_sup']}]")
    assert r.status_code == 200
    assert data["total"] == 2
    # Resultado deve estar ordenado por dias_previsto
    dias = [p["dias_previsto"] for p in data["previsoes"]]
    assert dias == sorted(dias), "Batch deve estar ordenado por dias_previsto"
    print("  ✅ OK")


def test_invalid_input():
    print("\n[5/5] POST /predict  (dados inválidos → 422)")
    payload_ruim = {**VENDA_STAND, "area_m2": -10}  # área negativa
    r = requests.post(f"{BASE_URL}/predict", json=payload_ruim)
    print(f"  Status : {r.status_code}  (esperado: 422)")
    assert r.status_code == 422
    print("  ✅ OK — validação funcionando")


# ── Classes pytest ────────────────────────────────────────────────────────────

class TestHealth:
    def test_status_ok(self):
        assert requests.get(f"{BASE_URL}/health").status_code == 200

    def test_modelo_carregado(self):
        r = requests.get(f"{BASE_URL}/health").json()
        assert r["artefatos"] == "carregados"


class TestPredict:
    def test_schema_resposta(self):
        r = requests.post(f"{BASE_URL}/predict", json=VENDA_STAND).json()
        assert {"dias_previsto", "intervalo_inf", "intervalo_sup", "canal"}.issubset(r.keys())

    def test_dias_positivo(self):
        r = requests.post(f"{BASE_URL}/predict", json=VENDA_STAND).json()
        assert r["dias_previsto"] > 0

    def test_intervalo_coerente(self):
        r = requests.post(f"{BASE_URL}/predict", json=VENDA_STAND).json()
        assert r["intervalo_inf"] <= r["dias_previsto"] <= r["intervalo_sup"]

    def test_stand_mais_rapido_que_instagram(self):
        dias_stand     = requests.post(f"{BASE_URL}/predict", json=VENDA_STAND).json()["dias_previsto"]
        dias_instagram = requests.post(f"{BASE_URL}/predict", json=VENDA_INSTAGRAM).json()["dias_previsto"]
        assert dias_stand < dias_instagram

    def test_area_negativa_rejeitada(self):
        payload = {**VENDA_STAND, "area_m2": -10}
        assert requests.post(f"{BASE_URL}/predict", json=payload).status_code == 422

    def test_tipologia_invalida_rejeitada(self):
        payload = {**VENDA_STAND, "tipologia": "Kitnet"}
        assert requests.post(f"{BASE_URL}/predict", json=payload).status_code == 422

    def test_canal_invalido_rejeitado(self):
        payload = {**VENDA_STAND, "origem_lead": "TikTok"}
        assert requests.post(f"{BASE_URL}/predict", json=payload).status_code == 422


class TestBatch:
    def test_total_correto(self):
        r = requests.post(f"{BASE_URL}/predict/batch", json=BATCH_PAYLOAD).json()
        assert r["total"] == 2

    def test_ordenado_por_dias(self):
        r = requests.post(f"{BASE_URL}/predict/batch", json=BATCH_PAYLOAD).json()
        dias = [p["dias_previsto"] for p in r["previsoes"]]
        assert dias == sorted(dias)

    def test_lista_vazia_rejeitada(self):
        r = requests.post(f"{BASE_URL}/predict/batch", json={"vendas": []})
        assert r.status_code == 422


# ── Execução direta ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Testes — Previsão Ciclo de Vendas Imobiliárias")
    print("=" * 55)
    test_health()
    test_predict_stand()
    test_predict_instagram()
    test_predict_batch()
    test_invalid_input()
    print("\n" + "=" * 55)
    print("  Todos os testes passaram ✅")
    print("=" * 55)
