import pytest
import pandas as pd
from abc import ABC
from src.cleaning import BaseCleaner

class TestBaseCleaner:
    """Testes para a interface base."""
    
    def test_is_abstract(self):
        """Verifica que BaseCleaner é uma classe abstrata."""
        # Arrange & Act & Assert
        with pytest.raises(TypeError):
            # Não deve poder instanciar classe abstrata
            cleaner = BaseCleaner()
    
    def test_concrete_implementation(self):
        """Testa que uma implementação concreta funciona."""
        # Arrange
        class ConcreteCleaner(BaseCleaner):
            def clean(self, series, **kwargs):
                return series, {'method': 'test'}
            
            def can_handle(self, dtype, sample_data):
                return True
        
        # Act
        cleaner = ConcreteCleaner()
        series = pd.Series([1, 2, 3])
        cleaned, stats = cleaner.clean(series)
        
        # Assert
        assert isinstance(cleaner, BaseCleaner)
        assert isinstance(cleaner, ABC)
        assert stats['method'] == 'test'
        assert cleaner.can_handle('any', series) == True
    
    def test_missing_implementation(self):
        """Testa que falta de implementação gera erro."""
        # Arrange
        class IncompleteCleaner(BaseCleaner):
            # Falta implementar clean()
            def can_handle(self, dtype, sample_data):
                return True
        
        # Act & Assert
        with pytest.raises(TypeError):
            cleaner = IncompleteCleaner()