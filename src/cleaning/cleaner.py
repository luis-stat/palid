import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from .cleaner_selector import CleanerSelector
from .hybrid_corrector import HybridCorrector

class DataCleaner:
    """Orquestra o processo de limpeza usando cleaners especializados."""
    
    def __init__(self, factory: Optional[CleanerSelector] = None):
        self.factory = factory or CleanerSelector()
        self.corrector = HybridCorrector()
        self.cleaning_report = {}
    
    def clean_dataset(
        self, 
        df: pd.DataFrame, 
        type_predictions: Dict[str, str],
        reference_dictionaries: Dict[str, List[str]] = {},
        similarity_threshold: float = 85
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Versão refatorada do método original, agora usando cleaners especializados.
        """
        cleaned_df = df.copy()
        report = {
            'columns_cleaned': {},
            'total_rows_corrected': 0,
            'total_rows_removed': 0
        }
        
        for col_name in df.columns:
            if col_name not in type_predictions:
                continue
                
            # 1. Seleciona o cleaner apropriado
            dtype = type_predictions[col_name]
            sample_data = df[col_name].head(100)  # Amostra para decisão
            cleaner = self.factory.get_cleaner_for_column(dtype, sample_data)
            
            # 2. Prepara parâmetros específicos
            kwargs = {
                'similarity_threshold': similarity_threshold,
                'categoria_type': dtype if 'CATEGORICO' in dtype else None
            }
            
            # Adiciona dicionário de referência se disponível
            if col_name in reference_dictionaries:
                kwargs['reference_values'] = reference_dictionaries[col_name]
            
            # 3. Executa limpeza
            cleaned_series, col_stats = cleaner.clean(
                df[col_name], 
                **kwargs
            )
            
            # 4. Atualiza DataFrame e relatório
            cleaned_df[col_name] = cleaned_series
            report['columns_cleaned'][col_name] = col_stats
            
            # 5. Atualiza totais
            report['total_rows_corrected'] += col_stats.get('rows_corrected', 0)
            report['total_rows_removed'] += col_stats.get('rows_removed', 0)
        
        return cleaned_df, report