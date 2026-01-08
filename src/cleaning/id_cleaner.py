import pandas as pd
import re
from typing import Any, Dict, Tuple, Optional
from .base_cleaner import BaseCleaner

class IDCleaner(BaseCleaner):
    def __init__(self, strip_whitespace: bool = True, case_sensitive: bool = False, 
                 preserve_cpf_format: bool = True):
        self.strip_whitespace = strip_whitespace
        self.case_sensitive = case_sensitive
        self.preserve_cpf_format = preserve_cpf_format
    
    def clean(self, series: pd.Series, **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        stats = {'method': 'id_cleaning', 'rows_corrected': 0, 'rows_removed': 0, 'cpfs_formatados': 0}
        cleaned_series = series.copy()
        
        sample = series.dropna().astype(str).head(50)
        cpf_matches = sum(1 for x in sample if self._looks_like_cpf(x))
        is_cpf_column = (len(sample) > 0) and (cpf_matches / len(sample) > 0.4)
        
        for idx, value in series.items():
            if pd.isna(value): continue
            original = str(value)
            
            if ' ' in original.strip() and not is_cpf_column: continue
            if 'R$' in original or '$' in original: continue

            cleaned = original.strip() if self.strip_whitespace else original
            
            if is_cpf_column and self.preserve_cpf_format:
                digits = re.sub(r'\D', '', cleaned)
                if len(digits) == 11:
                    cleaned = f"{digits[:3]}.{digits[3:6]}.{digits[6:9]}-{digits[9:]}"
                    if cleaned != original: stats['cpfs_formatados'] += 1
            elif not self.case_sensitive:
                cleaned = cleaned.upper()
                
            if cleaned != original:
                cleaned_series.iloc[idx] = cleaned
                stats['rows_corrected'] += 1
                
        return cleaned_series, stats

    def _looks_like_cpf(self, value: str) -> bool:
        digits = re.sub(r'\D', '', value)
        return len(digits) == 11

    def can_handle(self, dtype: str, sample_data: pd.Series) -> bool:
        sample = sample_data.dropna().astype(str).head(50)
        if len(sample) == 0: return False

        forbidden = {'true', 'false', 'sim', 'nao', 'yes', 'no', 'r$', '$'} # Adicionado R$
        sample_lower = [x.lower() for x in sample]
        
        # Se contiver termos proibidos
        if any(any(f in val for f in forbidden) for val in sample_lower):
            return False
            
        # Se tiver muitos espaços (Texto)
        if sum(1 for x in sample if ' ' in x.strip()) / len(sample) > 0.1:
            return False

        cpf_count = sum(1 for x in sample if self._looks_like_cpf(x))
        if cpf_count / len(sample) > 0.4: return True
            
        if dtype == 'ID': return True
        
        code_count = 0
        for x in sample:
            clean = x.strip()
            # Tem número ou é upper, curto, sem espaço
            if len(clean) < 25 and (any(c.isdigit() for c in clean) or clean.isupper()):
                code_count += 1
                
        return code_count / len(sample) > 0.7