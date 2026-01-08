from typing import List, Optional
import pandas as pd
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
            IDCleaner(preserve_cpf_format=True), 
            CategoricalCleaner(), 
            TextCleaner()
        ]
    
    def get_cleaner_for_column(self, dtype: str, sample_data: pd.Series) -> BaseCleaner:
        dtype_upper = dtype.upper()
        
        bool_cleaner = BooleanCleaner()
        if bool_cleaner.can_handle(dtype_upper, sample_data):
            return bool_cleaner

        if 'NUMERICO' in dtype_upper or 'FLOAT' in dtype_upper or 'INT' in dtype_upper:
            return NumericCleaner()

        if 'DATA' in dtype_upper:
            return DateCleaner()
            
        if any(x in dtype_upper for x in ['CATEGORICO', 'ESTADO', 'UF', 'CIDADE']):
            return CategoricalCleaner()

        email_cleaner = EmailCleaner()
        if email_cleaner.can_handle(dtype_upper, sample_data):
            return email_cleaner

        id_cleaner = IDCleaner()
        if dtype_upper == 'ID' or 'CPF' in dtype_upper:
             if id_cleaner.can_handle(dtype_upper, sample_data):
                 return id_cleaner
             else:
                 return TextCleaner()
        
        if id_cleaner.can_handle(dtype_upper, sample_data):
            return id_cleaner

        return TextCleaner()

    def register_cleaner(self, cleaner: BaseCleaner) -> None:
        self.cleaners.insert(0, cleaner)