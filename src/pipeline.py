"""
pipeline.py — Classe de pré-processamento e inferência
=======================================================
Carrega os artefatos salvos pelo train.py e expõe o método predict().
"""

import os
import pickle
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

TIPOLOGIA_RANK = {
    "Studio": 1, "1 Quarto": 2, "2 Quartos": 3,
    "3 Quartos": 4, "Cobertura": 5,
}


def _load(filename: str):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Artefato não encontrado: {path}\n"
            f"→ Execute 'python src/train.py' primeiro."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


class VendasPipeline:
    """
    Pipeline de inferência para previsão de dias_para_fechar.

    Uso
    ---
    >>> pipeline = VendasPipeline()
    >>> pipeline.load()
    >>> resultado = pipeline.predict(dados_df)

    Colunas obrigatórias no DataFrame de entrada:
        area_m2, valor_tabela, tipologia, origem_lead,
        empreendimento, corretor, imobiliaria, data_venda
        (ano ou data_venda para extrair ano)
    """

    def __init__(self):
        self.model        = None
        self.scaler       = None
        self.te_corretor  = None
        self.te_imob      = None
        self.metadata     = None
        self._loaded      = False

    def load(self) -> None:
        """Carrega todos os artefatos de disco."""
        logger.info("Carregando artefatos...")
        self.model       = _load("model.pkl")
        self.scaler      = _load("scaler.pkl")
        self.te_corretor = _load("te_corretor.pkl")
        self.te_imob     = _load("te_imobiliaria.pkl")
        self.metadata    = _load("ohe_metadata.pkl")
        self._loaded     = True
        logger.info("Artefatos carregados.")

    # ── Etapas de transformação ───────────────────────────────────────────────

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["data_venda"] = pd.to_datetime(df["data_venda"])
        df["ano"]        = df["data_venda"].dt.year
        df["mes"]        = df["data_venda"].dt.month
        df["trimestre"]  = df["data_venda"].dt.quarter
        df["mes_sin"]    = np.sin(2 * np.pi * df["mes"] / 12)
        df["mes_cos"]    = np.cos(2 * np.pi * df["mes"] / 12)
        df["tri_sin"]    = np.sin(2 * np.pi * df["trimestre"] / 4)
        df["tri_cos"]    = np.cos(2 * np.pi * df["trimestre"] / 4)
        df["valor_log"]  = np.log1p(df["valor_tabela"])
        df["tipologia_ord"] = df["tipologia"].map(TIPOLOGIA_RANK)
        return df

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        gm_cor  = self.metadata["global_mean_corretor"]
        gm_imob = self.metadata["global_mean_imobiliaria"]

        df["te_corretor"]    = df["corretor"].map(self.te_corretor).fillna(gm_cor)
        df["te_imobiliaria"] = df["imobiliaria"].map(self.te_imob).fillna(gm_imob)

        # OHE — canal (origem_lead)
        ohe_lead = pd.get_dummies(df["origem_lead"], prefix="canal", drop_first=True, dtype=float)
        ohe_lead = ohe_lead.reindex(columns=self.metadata["canal_cols"], fill_value=0.0)

        # OHE — empreendimento
        ohe_emp = pd.get_dummies(df["empreendimento"], prefix="emp", drop_first=True, dtype=float)
        ohe_emp = ohe_emp.reindex(columns=self.metadata["emp_cols"], fill_value=0.0)

        NUM_COLS = ["area_m2", "valor_log", "ano", "mes_sin", "mes_cos",
                    "tri_sin", "tri_cos", "tipologia_ord",
                    "te_corretor", "te_imobiliaria"]

        X = pd.concat([
            df[NUM_COLS].reset_index(drop=True),
            ohe_lead.reset_index(drop=True),
            ohe_emp.reset_index(drop=True),
        ], axis=1)

        return X[self.metadata["all_feats"]]

    # ── Inferência ────────────────────────────────────────────────────────────

    def predict(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Retorna DataFrame com:
            dias_previsto  — mediana prevista de dias para fechar
            intervalo_inf  — estimativa conservadora  (P25 proxy: +20%)
            intervalo_sup  — estimativa pessimista     (P75 proxy: +40%)
        """
        if not self._loaded:
            raise RuntimeError("Chame pipeline.load() antes de predict().")

        df = raw_df.pipe(self._feature_engineering).pipe(self._encode)

        X_scaled = self.scaler.transform(df)
        preds    = np.clip(self.model.predict(X_scaled), 1, None)

        result = pd.DataFrame({
            "dias_previsto" : np.round(preds).astype(int),
            "intervalo_inf" : np.round(preds * 0.80).astype(int),
            "intervalo_sup" : np.round(preds * 1.40).astype(int),
        })

        if "id_venda" in raw_df.columns:
            result.insert(0, "id_venda", raw_df["id_venda"].values)

        return result
