import pytest
import pandas as pd
from src.cleaning import HybridCorrector

class TestHybridCorrector:
    """Testes para o HybridCorrector."""
    
    def test_exact_match(self):
        """Testa correspondência exata."""
        # Arrange
        corrector = HybridCorrector()
        
        series = pd.Series(['São Paulo', 'Rio de Janeiro', 'Curitiba'])
        reference = ['São Paulo', 'Rio de Janeiro', 'Curitiba', 'Porto Alegre']
        
        # Act
        corrected_series, stats = corrector.correct(series, reference, threshold=100)
        
        # Assert
        assert stats['rows_corrected'] == 0  # Todos já estão corretos
        assert list(corrected_series) == list(series)  # Nada muda
    
    def test_fuzzy_correction(self):
        """Testa correção fuzzy."""
        # Arrange
        corrector = HybridCorrector()
        
        series = pd.Series(['sao paulo', 'rio janero', 'curitiba'])
        reference = ['São Paulo', 'Rio de Janeiro', 'Curitiba']
        
        # Act
        corrected_series, stats = corrector.correct(series, reference, threshold=70)
        
        # Assert
        assert stats['rows_corrected'] == 3
        assert 'corrections_made' in stats
        
        # Verifica correções (pode variar com threshold)
        corrections = stats['corrections_made']
        assert len(corrections) == 3
    
    def test_token_overlap(self):
        """Testa correção por sobreposição de tokens."""
        # Arrange
        corrector = HybridCorrector()
        
        series = pd.Series(['São Paulo Capital', 'Rio Janeiro'])
        reference = ['São Paulo', 'Rio de Janeiro']
        
        # Act
        corrected_series, stats = corrector.correct(series, reference, threshold=85)
        
        # Assert
        # "São Paulo Capital" tem overlap com "São Paulo"
        # "Rio Janeiro" tem overlap com "Rio de Janeiro"
        assert stats['rows_corrected'] >= 1
    
    def test_threshold_sensitivity(self):
        """Testa sensibilidade ao threshold."""
        # Arrange
        corrector = HybridCorrector()
        
        series = pd.Series(['sao paulo'])
        reference = ['São Paulo']
        
        # Act - Threshold alto (exigente)
        corrected_high, stats_high = corrector.correct(series, reference, threshold=95)
        
        # Act - Threshold baixo (permissivo)
        corrected_low, stats_low = corrector.correct(series, reference, threshold=70)
        
        # Assert
        # Com threshold alto, pode não corrigir
        # Com threshold baixo, deve corrigir
        # (comportamento específico depende da similaridade)
        assert stats_high['rows_corrected'] <= stats_low['rows_corrected']
    
    def test_cache_mechanism(self):
        """Testa mecanismo de cache."""
        # Arrange
        corrector = HybridCorrector()
        
        series = pd.Series(['sao paulo', 'sao paulo', 'rio'])  # Duplicados
        reference = ['São Paulo', 'Rio de Janeiro']
        
        # Act
        corrected_series, stats = corrector.correct(series, reference, threshold=80)
        
        # Assert
        # Cache deve evitar recálculo para valores duplicados
        assert 'sao paulo' in corrector.cache
        # Ambas as ocorrências de 'sao paulo' devem ter a mesma correção
    
    def test_empty_reference(self):
        """Testa com lista de referência vazia."""
        # Arrange
        corrector = HybridCorrector()
        
        series = pd.Series(['São Paulo', 'Rio'])
        reference = []
        
        # Act
        corrected_series, stats = corrector.correct(series, reference, threshold=80)
        
        # Assert
        assert stats['rows_corrected'] == 0  # Nada para corrigir
        assert list(corrected_series) == list(series)  # Nada muda
    
    def test_normalization_consistency(self):
        """Testa consistência da normalização."""
        # Arrange
        corrector = HybridCorrector()
        
        # Valores que devem normalizar para a mesma string
        test_cases = [
            ('São Paulo', 'sao paulo'),
            ('SÃO PAULO', 'sao paulo'),
            ('São  Paulo', 'sao paulo'),  # Espaço duplo
            ('São-Paulo', 'sao paulo'),   # Hífen
        ]
        
        for original, expected_normalized in test_cases:
            # Act
            normalized = corrector.normalize_text(original)
            
            # Assert
            assert normalized == expected_normalized
    
    def test_performance_large_dataset(self):
        """Testa performance com dataset grande."""
        # Arrange
        corrector = HybridCorrector()
        
        # Dataset grande
        n = 1000
        series = pd.Series([f'city_{i % 100}' for i in range(n)])
        reference = [f'CITY_{i}' for i in range(50)]
        
        # Act e Assert (não deve lançar exceção)
        corrected_series, stats = corrector.correct(series, reference, threshold=80)
        
        assert len(corrected_series) == n
        # Cache deve ajudar com valores duplicados
        assert len(corrector.cache) <= 100  # No máximo 100 valores únicos