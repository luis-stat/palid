from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import pandas as pd

class BaseCleaner(ABC):
    """Interface para todos os cleaners especializados."""
    
    @abstractmethod
    def clean(self, series: pd.Series, **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Limpa uma série de dados, retornando a série limpa e estatísticas.
        
        Args:
            series: Série de dados a ser limpa
            **kwargs: Argumentos específicos de cada cleaner
            
        Returns:
            Tuple contendo:
                - Série limpa
                - Dicionário com estatísticas de limpeza
        """
        pass
    
    @abstractmethod
    def can_handle(self, dtype: str, sample_data: pd.Series) -> bool:
        """
        Determina se este cleaner pode processar o tipo de dado.
        
        Args:
            dtype: Tipo predito pelo modelo (ex: 'DATA_HORA')
            sample_data: Amostra dos dados para verificação adicional
            
        Returns:
            True se este cleaner pode processar, False caso contrário
        """
        pass