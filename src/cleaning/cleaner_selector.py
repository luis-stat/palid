from typing import List, Optional, Any
import pandas as pd
import numpy as np
import unicodedata
import re
from .base_cleaner import BaseCleaner
from .date_cleaner import DateCleaner
from .categorical_cleaner import CategoricalCleaner
from .text_cleaner import TextCleaner
from .id_cleaner import IDCleaner
from .email_cleaner import EmailCleaner
from .numeric_cleaner import NumericCleaner
from .boolean_cleaner import BooleanCleaner

class CleanerSelector:
    def __init__(self, cleaners: Optional[List[BaseCleaner]] = None):
        self.cleaners = cleaners or self._get_default_cleaners()
    
    def _get_default_cleaners(self) -> List[BaseCleaner]:
        return [
            BooleanCleaner(),
            NumericCleaner(),
            DateCleaner(),
            EmailCleaner(),
            IDCleaner(), 
            CategoricalCleaner(), 
            TextCleaner()
        ]
    
    def _normalize_string(self, text: Any) -> str:
        """Normaliza strings para comparação segura."""
        if not isinstance(text, str):
            return str(text).upper()
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
        return text.strip().upper()

    def _looks_like_number(self, sample_data: pd.Series) -> bool:
        """
        Verifica se o CONTEÚDO da coluna é numérico, 
        ignorando se o dtype oficial é object/string.
        """
        if sample_data is None or sample_data.empty:
            return False
            
        sample = sample_data.dropna().astype(str).head(50)
        if len(sample) == 0: return False

        def is_convertible(val):
            try:
                val_clean = re.sub(r'[R$\s]', '', val).replace(',', '.')
                float(val_clean)
                return True
            except ValueError:
                return False

        numeric_count = sum(sample.apply(is_convertible))
        
        return (numeric_count / len(sample)) > 0.8

    def get_cleaner_for_column(self, dtype: str, sample_data: pd.Series) -> BaseCleaner:
        dtype_norm = self._normalize_string(dtype)
        
        if 'NUMERICO' in dtype_norm or 'FLOAT' in dtype_norm or 'INT' in dtype_norm or 'MONETARIO' in dtype_norm:
            return NumericCleaner()
        
        if 'DATA' in dtype_norm or 'DATE' in dtype_norm:
            return DateCleaner()
        
        if pd.api.types.is_numeric_dtype(sample_data):
            return NumericCleaner()
            
        if pd.api.types.is_datetime64_any_dtype(sample_data):
            return DateCleaner()

        if self._looks_like_number(sample_data):
            return NumericCleaner()

        bool_cleaner = BooleanCleaner()
        if bool_cleaner.can_handle(dtype_norm, sample_data):
            return bool_cleaner

        email_cleaner = EmailCleaner()
        if 'EMAIL' in dtype_norm or email_cleaner.can_handle(dtype_norm, sample_data):
            return email_cleaner

        if any(x in dtype_norm for x in ['CATEGORICO', 'ESTADO', 'UF', 'GENERO']):
            return CategoricalCleaner()

        if 'ID' in dtype_norm or 'CPF' in dtype_norm:
             return IDCleaner()

        return TextCleaner()

    def register_cleaner(self, cleaner: BaseCleaner) -> None:
        self.cleaners.insert(0, cleaner)