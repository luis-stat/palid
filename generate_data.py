#!/usr/bin/env python
"""
Gerador rápido de dados mínimos para teste.
"""
import pandas as pd
import numpy as np
import os

# Dataset simples de clientes
clientes = pd.DataFrame({
    'id': ['C001', 'C002', 'C003', 'C004', 'C005'],
    'nome': ['João Silva', 'MARIA SANTOS', 'carlos oliveira', 'Ana  Costa ', 'Pedro'],
    'email': ['joao@email.com', 'maria@email', 'carlos@email.com', None, 'pedro@email.com'],
    'cpf': ['123.456.789-00', '98765432100', '111.222.333-44', '000.000.000-00', '555.666.777-88'],
    'data_nascimento': ['15/07/1990', '1990-07-16', '16/07/1990', '17-07-1990', '1990/07/18'],
    'cidade': ['São Paulo', 'sao paulo', 'SAO PAULO', 'Rio de Janeiro', 'rio'],
    'estado': ['SP', 'sp', 'SP', 'RJ', 'rj'],
    'salario': [5000.50, 'R$ 3.500,00', 4200, 'alto', 3800.75],
    'ativo': [True, 'S', False, 'N', 1]
})

# Dataset simples de produtos
produtos = pd.DataFrame({
    'sku': ['SKU001', 'SKU002', 'SKU003', 'sku004', 'SKU005'],
    'nome': ['Notebook Dell', 'IPHONE 14', 'tv samsung', 'Geladeira', 'Mouse Gamer'],
    'categoria': ['Eletrônicos', 'Celular', 'Eletrônicos', 'Eletrodomésticos', 'Acessório'],
    'preco': [3500.00, 'R$ 4.999,99', 2500, '2.800,50', 199.90],
    'estoque': [10, 5, 0, 8, 25],
    'data_cadastro': ['2024-01-15', '15/01/2024', '2024-01-16', '16-01-2024', '17/01/2024']
})

# Salvar arquivos
clientes.to_csv('data/raw/clientes_simples.csv', index=False, encoding='utf-8')
produtos.to_csv('data/raw/produtos_simples.csv', index=False, encoding='utf-8')

# Criar gabarito master
gabarito = pd.DataFrame({
    'nome_arquivo': ['clientes_simples.csv', 'clientes_simples.csv', 'clientes_simples.csv',
                     'clientes_simples.csv', 'clientes_simples.csv', 'clientes_simples.csv',
                     'clientes_simples.csv', 'clientes_simples.csv', 'clientes_simples.csv',
                     'produtos_simples.csv', 'produtos_simples.csv', 'produtos_simples.csv',
                     'produtos_simples.csv', 'produtos_simples.csv', 'produtos_simples.csv'],
    'nome_coluna': ['id', 'nome', 'email', 'cpf', 'data_nascimento', 'cidade', 
                    'estado', 'salario', 'ativo', 'sku', 'nome', 'categoria',
                    'preco', 'estoque', 'data_cadastro'],
    'tipo_real': ['ID', 'TEXTO_LIVRE', 'TEXTO_LIVRE', 'ID', 'DATA_HORA', 
                  'CATEGORICO_NOMINAL', 'CATEGORICO_ESTADO', 'NUMERICO', 
                  'CATEGORICO_NOMINAL', 'ID', 'TEXTO_LIVRE', 'CATEGORICO_NOMINAL',
                  'NUMERICO', 'NUMERICO', 'DATA_HORA']
})

gabarito.to_csv('data/external/gabarito_master.csv', index=False, encoding='utf-8')