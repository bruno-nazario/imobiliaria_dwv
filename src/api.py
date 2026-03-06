"""
api.py — API de Previsão de Ciclo de Vendas Imobiliárias
=========================================================
Sobe a API:
    uvicorn src.api:app --reload --port 8000

Endpoints:
    GET  /health          → status
    POST /predict         → previsão individual
    POST /predict/batch   → previsão em lote
    GET  /docs            → Swagger UI
"""

from contextlib import asynccontextmanager
from typing import List, Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.pipeline import VendasPipeline

# ── Pipeline (singleton carregado no startup) ─────────────────────────────────
pipeline = VendasPipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline.load()
    yield


app = FastAPI(
    title="Previsão de Ciclo de Vendas Imobiliárias",
    description=(
        "Prevê quantos dias uma venda levará para fechar "
        "com base nas características do imóvel, canal de lead, "
        "corretor e empreendimento."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class VendaInput(BaseModel):
    """Dados de uma venda para previsão."""

    id_venda: Optional[int] = Field(None, description="ID da venda (opcional)", example=501)

    data_venda: str = Field(
        ...,
        description="Data da venda no formato YYYY-MM-DD",
        example="2025-03-15",
    )
    empreendimento: Literal[
        "Villa Portofino", "Ocean Tower", "Grand Maré",
        "Mirante Sul", "Edifício Brisa", "Residencial Atlântico",
        "Jardins de Itapema",
    ] = Field(..., example="Grand Maré")

    tipologia: Literal["Studio", "1 Quarto", "2 Quartos", "3 Quartos", "Cobertura"] = Field(
        ..., example="2 Quartos"
    )
    area_m2: float = Field(..., gt=0, le=300, description="Área em m²", example=72.5)
    valor_tabela: float = Field(..., gt=0, description="Valor de tabela em R$", example=650000.0)

    forma_pagamento: Literal["Financiamento", "Parcelamento direto", "À vista"] = Field(
        ..., example="Financiamento"
    )
    corretor: str = Field(..., description="Nome do corretor", example="Juliana Souza")
    imobiliaria: str = Field(..., description="Nome da imobiliária", example="Prime Realty")

    origem_lead: Literal["Stand", "Indicação", "WhatsApp", "Portal", "Instagram"] = Field(
        ..., example="WhatsApp"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "id_venda": 501,
                "data_venda": "2025-03-15",
                "empreendimento": "Grand Maré",
                "tipologia": "2 Quartos",
                "area_m2": 72.5,
                "valor_tabela": 650000.0,
                "forma_pagamento": "Financiamento",
                "corretor": "Juliana Souza",
                "imobiliaria": "Prime Realty",
                "origem_lead": "WhatsApp",
            }
        }
    }


class PrevisaoOutput(BaseModel):
    id_venda: Optional[int]
    dias_previsto: int = Field(..., description="Mediana prevista de dias para fechar")
    intervalo_inf: int = Field(..., description="Estimativa otimista (−20%)")
    intervalo_sup: int = Field(..., description="Estimativa pessimista (+40%)")
    canal: str         = Field(..., description="Canal de origem do lead")


class BatchInput(BaseModel):
    vendas: List[VendaInput] = Field(..., min_length=1, max_length=5000)


class BatchOutput(BaseModel):
    total: int
    previsoes: List[PrevisaoOutput]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Status"])
def health():
    """Verifica se a API e o modelo estão prontos."""
    return {
        "status"   : "ok",
        "modelo"   : type(pipeline.model).__name__,
        "artefatos": "carregados",
    }


@app.post(
    "/predict",
    response_model=PrevisaoOutput,
    tags=["Predição"],
    summary="Previsão individual",
    description="Recebe os dados de **uma venda** e retorna a previsão de dias para fechar.",
)
def predict_single(venda: VendaInput):
    try:
        df     = pd.DataFrame([venda.model_dump()])
        result = pipeline.predict(df)
        row    = result.iloc[0]
        return PrevisaoOutput(
            id_venda      = venda.id_venda,
            dias_previsto = int(row["dias_previsto"]),
            intervalo_inf = int(row["intervalo_inf"]),
            intervalo_sup = int(row["intervalo_sup"]),
            canal         = venda.origem_lead,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict/batch",
    response_model=BatchOutput,
    tags=["Predição"],
    summary="Previsão em lote",
    description="Recebe uma lista de vendas e retorna previsões para todas, ordenadas por dias previsto.",
)
def predict_batch(payload: BatchInput):
    try:
        df     = pd.DataFrame([v.model_dump() for v in payload.vendas])
        result = pipeline.predict(df)
        result["canal"] = df["origem_lead"].values

        previsoes = [
            PrevisaoOutput(
                id_venda      = int(row["id_venda"]) if "id_venda" in result.columns else None,
                dias_previsto = int(row["dias_previsto"]),
                intervalo_inf = int(row["intervalo_inf"]),
                intervalo_sup = int(row["intervalo_sup"]),
                canal         = str(row["canal"]),
            )
            for _, row in result.sort_values("dias_previsto").iterrows()
        ]
        return BatchOutput(total=len(previsoes), previsoes=previsoes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
