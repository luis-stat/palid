import pandas as pd
import numpy as np
import unicodedata
import re
from typing import Any, Dict, Tuple
from .base_cleaner import BaseCleaner

class TextCleaner(BaseCleaner):
    """
    Cleaner genérico para texto livre (nomes, observações, cidades, etc)
    """
    
    def __init__(
        self, 
        use_title_case: bool = True,
        normalize_accents: bool = False,
        remove_extra_spaces: bool = True
    ):
        self.use_title_case = use_title_case
        self.normalize_accents = normalize_accents
        self.remove_extra_spaces = remove_extra_spaces
    
    def clean(self, series: pd.Series, **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        stats = {
            'method': 'text_standardization',
            'rows_corrected': 0,
            'rows_removed': 0
        }
        
        cleaned_series = series.copy()
        
        for idx, value in series.items():
            if pd.isna(value): continue
            
            original = str(value)
            cleaned = original
            
            # 1. Remove espaços extras (inicio, fim e duplos no meio)
            if self.remove_extra_spaces:
                cleaned = " ".join(cleaned.split())
            
            # 2. Title Case (João da Silva)
            if self.use_title_case:
                cleaned = self._to_title_case_br(cleaned)
                
            # 3. Remove acentos (Opcional, padrão desligado para nomes)
            if self.normalize_accents:
                cleaned = unicodedata.normalize('NFKD', cleaned).encode('ASCII', 'ignore').decode('ASCII')
            
            if cleaned != original:
                cleaned_series.iloc[idx] = cleaned
                stats['rows_corrected'] += 1
                
        return cleaned_series, stats
    
    def _to_title_case_br(self, text: str) -> str:
        """Converte para Title Case respeitando preposições brasileiras."""
        if not text: return ""
        
        # Lista de preposições que devem ficar minúsculas
        small_words = {'de', 'da', 'do', 'das', 'dos', 'e', 'em', 'para', 'com'}
        
        words = text.split()
        new_words = []
        
        for i, word in enumerate(words):
            word_lower = word.lower()
            # Se for a primeira palavra ou não for preposição, capitaliza
            if i == 0 or word_lower not in small_words:
                new_words.append(word.capitalize())
            else:
                new_words.append(word_lower)
                
        return " ".join(new_words)
    
    def can_handle(self, dtype: str, sample_data: pd.Series) -> bool:
        """
        Aceita qualquer coisa que o modelo diga que é TEXTO
        OU que tenha características de texto (espaços, letras).
        """
        # 1. Se a IA disse que é texto, aceita imediatamente
        if 'TEXTO' in dtype.upper():
            return True
            
        sample = sample_data.dropna().astype(str).head(20)
        if len(sample) == 0: return False
        
        # Conta quantos itens têm letras e espaços
        text_like_count = 0
        for x in sample:
            has_letters = any(c.isalpha() for c in x)
            has_spaces = ' ' in x
            # Se tem letras e espaços, é quase certeza que é texto livre
            if has_letters and has_spaces:
                text_like_count += 1
                
        # Se mais de 30% da amostra parece texto, aceita
        return (text_like_count / len(sample)) > 0.3