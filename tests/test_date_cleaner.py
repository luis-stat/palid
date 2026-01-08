import pytest
import pandas as pd
import numpy as np
from src.cleaning import DateCleaner

class TestDateCleaner:
    """Testes para o DateCleaner."""
    
    def test_clean_valid_dates(self, sample_date_series):
        """Testa limpeza de datas válidas."""
        # Arrange
        cleaner = DateCleaner(date_format='%d/%m/%Y')
        
        # Act
        cleaned_series, stats = cleaner.clean(sample_date_series)
        
        # Assert
        # Verifica estatísticas
        assert stats['method'] == 'date_standardization'
        assert stats['rows_removed'] > 0  # Deve remover algumas datas inválidas
        
        # Verifica datas válidas foram padronizadas
        assert cleaned_series.iloc[0] == '15/07/2023'  # Já estava no formato correto
        assert cleaned_series.iloc[1] == '15/07/2023'  # ISO -> dd/mm/yyyy
        assert cleaned_series.iloc[3] is pd.NA  # "not a date" deve virar NA
        
        # Verifica que NA permanece NA
        assert pd.isna(cleaned_series.iloc[5])
    
    def test_can_handle_date_type(self):
        """Testa detecção de tipo DATA_HORA."""
        # Arrange
        cleaner = DateCleaner()
        sample_data = pd.Series(['2023-01-01', '2023-01-02'])
        
        # Act & Assert
        assert cleaner.can_handle('DATA_HORA', sample_data) == True
        assert cleaner.can_handle('TEXTO_LIVRE', sample_data) == False
    
    def test_different_date_formats(self):
        """Testa diferentes formatos de data."""
        # Arrange
        cleaner_us = DateCleaner(date_format='%m/%d/%Y')
        cleaner_br = DateCleaner(date_format='%d/%m/%Y')
        
        test_series = pd.Series(['07/15/2023', '15/07/2023'])
        
        # Act
        cleaned_us, _ = cleaner_us.clean(test_series)
        cleaned_br, _ = cleaner_br.clean(test_series)
        
        # Assert
        # O mesmo input gera resultados diferentes dependendo do formato
        assert cleaned_us.iloc[0] == '07/15/2023'
        assert cleaned_br.iloc[1] == '15/07/2023'
    
    def test_empty_series(self):
        """Testa limpeza de série vazia."""
        # Arrange
        cleaner = DateCleaner()
        empty_series = pd.Series([], dtype=object)
        
        # Act
        cleaned_series, stats = cleaner.clean(empty_series)
        
        # Assert
        assert len(cleaned_series) == 0
        assert stats['rows_removed'] == 0
    
    @pytest.mark.parametrize("input_date,expected_output", [
        ('15/07/2023', '15/07/2023'),
        ('2023-07-15', '15/07/2023'),
        ('15-07-2023', '15/07/2023'),
        ('15.07.2023', '15/07/2023'),
        ('15072023', pd.NA),  # Sem separadores
        ('20230715', pd.NA),  # Formato compacto
    ])
    def test_specific_date_formats(self, input_date, expected_output):
        """Testa formatos de data específicos."""
        # Arrange
        cleaner = DateCleaner()
        series = pd.Series([input_date])
        
        # Act
        cleaned_series, _ = cleaner.clean(series)
        
        # Assert
        if expected_output is pd.NA:
            assert pd.isna(cleaned_series.iloc[0])
        else:
            assert cleaned_series.iloc[0] == expected_output
    
    def test_performance_large_dataset(self):
        """Testa performance com dataset grande."""
        # Arrange
        cleaner = DateCleaner()
        large_series = pd.Series(['15/07/2023'] * 10000 + ['invalid'] * 100)
        
        # Act & Assert (não deve lançar exceção)
        cleaned_series, stats = cleaner.clean(large_series)
        
        assert len(cleaned_series) == 10100
        assert stats['rows_removed'] == 100  # 'invalid' deve ser removido