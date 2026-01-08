import pytest
import pandas as pd
import numpy as np
from typing import Dict, List, Any

@pytest.fixture
def sample_date_series():
    """Série de exemplo com datas válidas e inválidas."""
    return pd.Series([
        '15/07/2023',
        '2023-07-15',
        '15-07-2023',
        'not a date',
        '2023/07/15',
        np.nan,
        '07-15-2023',  # Formato americano (inválido para dayfirst=True)
        '15/13/2023',  # Mês inválido
    ])

@pytest.fixture
def sample_categorical_series():
    """Série de exemplo com dados categóricos."""
    return pd.Series([
        'São Paulo',
        'sao paulo',
        'SÃO PAULO',
        'Rio de Janeiro',
        'rio de janeiro',
        'Curitiba',
        'curitiba',
        'PORTO ALEGRE',
        np.nan,
        '',
        '  rio de janeiro  ',
    ])

@pytest.fixture
def sample_text_series():
    """Série de exemplo com texto livre."""
    return pd.Series([
        'Olá, mundo!',
        'café com açúcar',
        'São Paulo - SP',
        'João da Silva',
        'email@exemplo.com',
        'Rua das Flores, 123',
        np.nan,
        '   texto com espaços   ',
        'AÇÃO & REAÇÃO',
    ])

@pytest.fixture
def sample_id_series():
    """Série de exemplo com IDs."""
    return pd.Series([
        'ABC123',
        'abc123',
        'ABC 123',
        'XYZ-789',
        'CPF: 123.456.789-00',
        '987654321',
        'ID-001',
        '  ID002  ',
        np.nan,
        '',
        '123',
    ])

@pytest.fixture
def reference_dictionary():
    """Dicionário de referência para correção híbrida."""
    return [
        'São Paulo',
        'Rio de Janeiro',
        'Curitiba',
        'Porto Alegre',
        'Belo Horizonte',
        'Salvador',
    ]

@pytest.fixture
def hybrid_corrector():
    """Instância do HybridCorrector para testes."""
    from src.cleaning import HybridCorrector
    return HybridCorrector()