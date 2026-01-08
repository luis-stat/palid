import pandas as pd
import re
from typing import Any, Dict, Tuple, Optional
from .base_cleaner import BaseCleaner

class NumericCleaner(BaseCleaner):
    """Cleaner numérico com trava de segurança anti-destruição"""
    
    def __init__(self):
        pass
    
    def clean(self, series: pd.Series, **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        stats = {
            'method': 'numeric_cleaning',
            'rows_corrected': 0,
            'rows_removed': 0,
            'converted_to_float': 0,
            'safety_lock_triggered': False
        }
        
        count_original = series.count()
        cleaned_series = series.copy()
        
        # Aplica conversão linha a linha primeiro para contar sucessos
        temp_cleaned = series.apply(self._clean_numeric)
        
        # Verifica quantos viraram número válido (não NaN e não string original inalterada se fosse lixo)
        # Verifica-se quantos são float/int no final
        temp_numeric = pd.to_numeric(temp_cleaned, errors='coerce')
        count_final = temp_numeric.count()

        # Caso perca 50% dos dados
        if count_original > 0 and (count_final / count_original) < 0.5:
             stats['method'] = 'skipped_safety_lock (IA Error Detected)'
             stats['safety_lock_triggered'] = True
             return series, stats # Cancela a operação
        
        # Se passou, aplica a conversão final
        cleaned_series = temp_numeric
        
        # Calcula stats
        for idx, val in series.items():
            if pd.notna(val) and pd.notna(cleaned_series[idx]):
                if val != cleaned_series[idx]:
                    stats['rows_corrected'] += 1
        
        return cleaned_series, stats
    
    def _clean_numeric(self, value: Any) -> Any:
        if pd.isna(value): return value
        str_val = str(value).strip()
        # Remove R$, espaços, etc
        str_val = re.sub(r'[R$\€\£\¥\s]', '', str_val)
        
        if ',' in str_val and '.' in str_val:
            if str_val.rfind(',') > str_val.rfind('.'): # 1.000,00
                str_val = str_val.replace('.', '').replace(',', '.')
            else: # 1,000.00
                str_val = str_val.replace(',', '')
        elif ',' in str_val:
            str_val = str_val.replace(',', '.')
            
        return str_val

    def can_handle(self, dtype: str, sample_data: pd.Series) -> bool:
        return dtype == 'NUMERICO'