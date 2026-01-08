"""
Módulo de limpeza de dados com arquitetura baseada em estratégia.
Exporta todas as classes principais para uso externo.
"""

from .base_cleaner import BaseCleaner
from .cleaner_selector import CleanerSelector
from .cleaner import DataCleaner
from .date_cleaner import DateCleaner
from .categorical_cleaner import CategoricalCleaner
from .text_cleaner import TextCleaner
from .id_cleaner import IDCleaner
from .hybrid_corrector import HybridCorrector
from .email_cleaner import EmailCleaner      
from .numeric_cleaner import NumericCleaner  
from .boolean_cleaner import BooleanCleaner  

__all__ = [
    'BaseCleaner',
    'CleanerSelector',
    'DataCleaner',
    'DateCleaner',
    'CategoricalCleaner',
    'TextCleaner',
    'IDCleaner',
    'HybridCorrector',
    'EmailCleaner',            
    'NumericCleaner',   
    'BooleanCleaner', 
]