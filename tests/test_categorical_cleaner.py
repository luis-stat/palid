import pytest
import pandas as pd
from src.cleaning import CategoricalCleaner
from src.cleaning import HybridCorrector

class TestCategoricalCleaner:
    """Testes para o CategoricalCleaner."""
    
    def test_title_case_conversion(self, sample_categorical_series):
        """Testa conversão para title case em português."""
        # Arrange
        cleaner = CategoricalCleaner()
        
        # Act
        cleaned_series, stats = cleaner.clean(sample_categorical_series)
        
        # Assert
        assert stats['method'] == 'categorical_cleaning'
        
        # Verifica title case aplicado
        assert cleaned_series.iloc[0] == 'São Paulo'  # Já estava correto
        assert cleaned_series.iloc[1] == 'Sao Paulo'  # minúsculo -> title case
        assert cleaned_series.iloc[3] == 'Rio de Janeiro'  # mantém "de" minúsculo
        assert cleaned_series.iloc[10] == 'Rio de Janeiro'  # remove espaços extras
        
        # Verifica tratamento de valores vazios/NA
        assert pd.isna(cleaned_series.iloc[8])  # NA permanece NA
        assert cleaned_series.iloc[9] == ''  # string vazia permanece vazia
    
    def test_hybrid_correction_with_dictionary(self, sample_categorical_series, reference_dictionary):
        """Testa correção híbrida com dicionário de referência."""
        # Arrange
        corrector = HybridCorrector()
        cleaner = CategoricalCleaner(corrector=corrector)
        
        # Act
        cleaned_series, stats = cleaner.clean(
            sample_categorical_series,
            reference_values=reference_dictionary,
            similarity_threshold=80
        )
        
        # Assert
        assert stats['method'] == 'hybrid_dictionary'
        assert stats['rows_corrected'] > 0
        
        # Verifica correções específicas
        assert cleaned_series.iloc[7] == 'Porto Alegre'  # "PORTO ALEGRE" -> title case
        assert 'corrections_made' in stats
    
    def test_frequency_fallback(self):
        """Testa fallback baseado em frequência."""
        # Arrange
        cleaner = CategoricalCleaner()
        
        # Série com alguns valores frequentes e outros raros
        series = pd.Series(
            ['A'] * 10 + ['B'] * 10 + ['C'] * 5 + ['D'] * 2 + ['E'] * 1
        )
        
        # Act (sem dicionário de referência)
        cleaned_series, stats = cleaner.clean(series)
        
        # Assert
        # Como não há dicionário, deve usar fallback de frequência
        assert stats['method'] == 'frequency_fallback'
        # Valores com frequência >= 5 (A, B, C) são considerados válidos
        # D e E podem ser corrigidos se similares a A, B ou C
    
    def test_can_handle_categorical_types(self):
        """Testa detecção de tipos categóricos."""
        # Arrange
        cleaner = CategoricalCleaner()
        sample_data = pd.Series(['A', 'B', 'C'])
        
        # Act & Assert
        assert cleaner.can_handle('CATEGORICO_NOMINAL', sample_data) == True
        assert cleaner.can_handle('CATEGORICO_ORDINAL', sample_data) == True
        assert cleaner.can_handle('CATEGORICO_ESTADO', sample_data) == True
        assert cleaner.can_handle('TEXTO_LIVRE', sample_data) == False
        assert cleaner.can_handle('DATA_HORA', sample_data) == False
    
    @pytest.mark.parametrize("input_text,expected_output", [
        ('são paulo', 'São Paulo'),
        ('SAO PAULO', 'Sao Paulo'),
        ('rio de janeiro', 'Rio de Janeiro'),
        ('DE JANEIRO', 'De Janeiro'),
        ('para o futuro', 'Para o Futuro'),
        ('', ''),
    ])
    def test_specific_title_cases(self, input_text, expected_output):
        """Testa casos específicos de title case."""
        # Arrange
        cleaner = CategoricalCleaner()
        series = pd.Series([input_text])
        
        # Act
        cleaned_series, _ = cleaner.clean(series)
        
        # Assert
        assert cleaned_series.iloc[0] == expected_output
    
    def test_edge_cases(self):
        """Testa casos extremos e borda."""
        # Arrange
        cleaner = CategoricalCleaner()
        
        test_cases = [
            ('all_upper', 'ALL UPPER', 'All Upper'),
            ('all_lower', 'all lower', 'All Lower'),
            ('mixed_CASE', 'mixed CASE', 'Mixed Case'),
            ('with_123', 'with 123', 'With 123'),
            ('with-special', 'with-special', 'With-Special'),
        ]
        
        for test_name, input_val, expected in test_cases:
            # Act
            cleaned_series, _ = cleaner.clean(pd.Series([input_val]))
            
            # Assert
            assert cleaned_series.iloc[0] == expected, f"Failed for {test_name}"