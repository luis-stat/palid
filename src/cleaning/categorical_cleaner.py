import pandas as pd
from typing import Any, Dict, List, Tuple, Optional
from .base_cleaner import BaseCleaner
from .hybrid_corrector import HybridCorrector

class CategoricalCleaner(BaseCleaner):
    """Cleaner especializado em dados categóricos."""
    
    def __init__(self, corrector: Optional[HybridCorrector] = None):
        self.corrector = corrector or HybridCorrector()
        
    def clean(self, series: pd.Series, **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Limpa dados categóricos aplicando:
        1. Title case (para português)
        2. Correção híbrida se houver dicionário de referência
        3. Fallback baseado em frequência se necessário
        """
        reference_values = kwargs.get('reference_values', [])
        threshold = kwargs.get('similarity_threshold', 85)
        categoria_type = kwargs.get('categoria_type', 'CATEGORICO_NOMINAL')
        
        stats = {
            'method': 'categorical_cleaning',
            'rows_corrected': 0,
            'rows_removed': 0,
            'corrections_made': {}
        }
        
        # 1. Aplica title case em português
        cleaned_series = series.apply(self._to_title_case_br)
        
        # 2. Se tem dicionário, aplica correção híbrida
        if reference_values:
            cleaned_series, correction_stats = self.corrector.correct(
                cleaned_series, 
                reference_values, 
                threshold
            )
            stats.update(correction_stats)
            stats['method'] = 'hybrid_dictionary'
        # 3. Fallback baseado em frequência para categorias
        elif 'CATEGORICO' in categoria_type:
            frequent_values = self._get_frequent_values(cleaned_series)
            if frequent_values:
                cleaned_series, correction_stats = self.corrector.correct(
                    cleaned_series,
                    frequent_values,
                    threshold
                )
                stats.update(correction_stats)
                stats['method'] = 'frequency_fallback'
        
        return cleaned_series, stats
    
    def _to_title_case_br(self, text: Any) -> Any:
        """Title case para português (extraído do código original)."""
        if pd.isna(text):
            return text
        text = str(text).strip()
        if not text:
            return text
        small = {'de', 'da', 'do', 'das', 'dos', 'e', 'em', 'para', 'com'}
        words = text.split()
        return ' '.join([
            w.capitalize() if i == 0 or w.lower() not in small else w.lower()
            for i, w in enumerate(words)
        ])
    
    def _get_frequent_values(self, series: pd.Series, min_count: int = 5) -> List[str]:
        """Identifica valores frequentes para fallback."""
        counts = series.value_counts()
        min_required = max(min_count, int(len(series) * 0.01))
        return counts[counts >= min_required].index.tolist()
    
    def can_handle(self, dtype: str, sample_data: pd.Series) -> bool:
        return any(cat_type in dtype for cat_type in [
            'CATEGORICO_NOMINAL', 
            'CATEGORICO_ORDINAL', 
            'CATEGORICO_ESTADO'
        ])