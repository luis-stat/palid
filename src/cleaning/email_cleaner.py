import pandas as pd
import re
from typing import Any, Dict, Tuple
from .base_cleaner import BaseCleaner

class EmailCleaner(BaseCleaner):
    """Cleaner especializado em endereços de email"""
    
    def __init__(self):
        self.email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    
    def clean(self, series: pd.Series, **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        """Limpa e valida endereços de email"""
        stats = {
            'method': 'email_cleaning',
            'rows_corrected': 0,
            'rows_removed': 0,
            'invalid_emails': 0
        }
        
        cleaned_series = series.copy()
        invalid_count = 0
        
        for idx, email in series.items():
            if pd.isna(email):
                continue
            
            email_str = str(email).strip()
            cleaned_email = self._clean_email(email_str)
            
            if cleaned_email != email_str:
                cleaned_series.iloc[idx] = cleaned_email
                stats['rows_corrected'] += 1
            
            if not self._is_valid_email(cleaned_email):
                invalid_count += 1
        
        stats['invalid_emails'] = invalid_count
        
        return cleaned_series, stats
    
    def _clean_email(self, email: str) -> str:
        """Limpa email individual."""
        email = email.strip().lower()
        
        # Remove espaços
        email = email.replace(' ', '')
        
        # Corrige erros comuns
        email = email.replace('@gmail', '@gmail.com')
        email = email.replace('@hotmail', '@hotmail.com')
        email = email.replace('@yahoo', '@yahoo.com')
        email = email.replace('@outlook', '@outlook.com')
        
        # Garante que termina com domínio válido
        if '@' in email and '.' not in email.split('@')[-1]:
            email += '.com'
        
        return email
    
    def _is_valid_email(self, email: str) -> bool:
        """Valida formato de email."""
        if pd.isna(email) or not email:
            return False
        
        return bool(self.email_pattern.match(email))
    
    def can_handle(self, dtype: str, sample_data: pd.Series) -> bool:
        """Lida com colunas que parecem conter emails."""
        if 'email' in dtype.lower() or 'e-mail' in dtype.lower():
            return True
        
        sample = sample_data.dropna().head(20)
        if len(sample) == 0:
            return False
        
        email_like_count = 0
        for val in sample:
            if pd.isna(val):
                continue
            
            str_val = str(val).strip()
            if '@' in str_val and '.' in str_val:
                email_like_count += 1
        
        # Se mais de 70% parecem emails
        return email_like_count / len(sample) > 0.7