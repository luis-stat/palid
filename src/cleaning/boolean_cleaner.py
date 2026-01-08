import pandas as pd
from typing import Any, Dict, Tuple
from .base_cleaner import BaseCleaner

class BooleanCleaner(BaseCleaner):
    """Cleaner especializado em dados booleanos."""
    
    def __init__(self):
        self.true_values = ['true', 'verdadeiro', 'sim', 's', 'yes', 'y', '1', 1, True, 'v']
        self.false_values = ['false', 'falso', 'não', 'nao', 'n', 'no', '0', 0, False, 'f']
        
        self.true_values_lower = [str(v).lower() for v in self.true_values]
        self.false_values_lower = [str(v).lower() for v in self.false_values]
    
    def clean(self, series: pd.Series, **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        stats = {
            'method': 'boolean_cleaning',
            'rows_corrected': 0,
            'rows_removed': 0,
            'converted_to_true': 0,
            'converted_to_false': 0
        }
        
        cleaned_series = series.copy()
        
        for idx, value in series.items():
            if pd.isna(value): continue
            
            cleaned = self._clean_boolean(value)
            
            if cleaned != value:
                cleaned_series.iloc[idx] = cleaned
                stats['rows_corrected'] += 1
                
                if cleaned is True: stats['converted_to_true'] += 1
                elif cleaned is False: stats['converted_to_false'] += 1
        
        return cleaned_series, stats
    
    def _clean_boolean(self, value: Any) -> Any:
        if pd.isna(value): return value
        if isinstance(value, bool): return value
        
        str_val = str(value).strip().lower()
        
        if str_val.endswith('.0'):
            str_val = str_val[:-2]

        if str_val in self.true_values_lower: return True
        if str_val in self.false_values_lower: return False
        
        return value
    
    def can_handle(self, dtype: str, sample_data: pd.Series) -> bool:
        """Heurística forte para capturar booleanos."""
        if 'BOOLEANO' in dtype.upper(): return True
        
        sample = sample_data.dropna().head(50)
        if len(sample) == 0: return False
        
        bool_matches = 0
        for val in sample:
            c = self._clean_boolean(val)
            if isinstance(c, bool):
                bool_matches += 1
        
        return (bool_matches / len(sample)) > 0.6