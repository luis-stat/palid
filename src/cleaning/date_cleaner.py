import pandas as pd
import numpy as np
from typing import Any, Dict, Tuple
from .base_cleaner import BaseCleaner

class DateCleaner(BaseCleaner):
    """Cleaner especializado em dados de data/hora com trava de segurança"""
    
    def __init__(self, date_format: str = '%d/%m/%Y'):
        self.date_format = date_format
        
    def clean(self, series: pd.Series, **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        stats = {
            'method': 'date_standardization',
            'rows_corrected': 0,
            'rows_removed': 0,
            'safety_lock_triggered': False
        }
        
        # Contagem inicial de dados válidos (não nulos)
        count_original = series.count()
        
        # Tenta limpar
        cleaned_series = series.apply(self._standardize_date)
        
        # Contagem final
        count_final = cleaned_series.count()
        
        # Se a coluna tinha dados (>0) e a limpeza destruiu mais de 50% deles
        # Significa que a classificação da IA provavelmente está errada (ex: Texto classificado como Data)
        if count_original > 0 and (count_final / count_original) < 0.5:
            stats['method'] = 'skipped_safety_lock (IA Error Detected)'
            stats['safety_lock_triggered'] = True
            return series, stats  # Devolve o original em caso de falha
            
        # Se passou na segurança, calcula estatísticas reais
        na_before = series.isna().sum()
        na_after = cleaned_series.isna().sum()
        stats['rows_removed'] = int(na_after - na_before)
        
        return cleaned_series, stats
    
    def _standardize_date(self, date_str: Any) -> Any:
        if pd.isna(date_str):
            return np.nan
        try:
            # Força string e remove espaços
            d_str = str(date_str).strip()
            if not d_str: return np.nan
            
            # Tenta converter
            return pd.to_datetime(
                d_str, 
                dayfirst=True, 
                errors='coerce'
            ).strftime(self.date_format)
        except:
            return np.nan
    
    def can_handle(self, dtype: str, sample_data: pd.Series) -> bool:
        return dtype == 'DATA_HORA'