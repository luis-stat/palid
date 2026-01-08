import pandas as pd
import numpy as np
from typing import Any, Dict

def safe_apply(series: pd.Series, func, default: Any = np.nan) -> pd.Series:
    """
    Aplica uma função de forma segura, tratando exceções.
    
    Args:
        series: Série de dados
        func: Função a aplicar
        default: Valor padrão em caso de erro
        
    Returns:
        Série com função aplicada
    """
    def safe_wrapper(x):
        try:
            return func(x)
        except:
            return default
    
    return series.apply(safe_wrapper)

def calculate_cardinality_stats(series: pd.Series) -> Dict[str, float]:
    """
    Calcula estatísticas de cardinalidade.
    
    Returns:
        Dicionário com estatísticas
    """
    non_null = series.dropna()
    total = len(series)
    
    if total == 0:
        return {
            'non_null_ratio': 0,
            'cardinality': 0,
            'cardinality_ratio': 0
        }
    
    non_null_count = len(non_null)
    unique_count = non_null.nunique()
    
    return {
        'non_null_ratio': non_null_count / total,
        'cardinality': unique_count,
        'cardinality_ratio': unique_count / non_null_count if non_null_count > 0 else 0
    }