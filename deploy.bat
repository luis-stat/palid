@echo off
echo 🚀 Implantando arquitetura modular...

echo 📁 Criando estrutura de diretórios...
mkdir src\cleaning 2>nul
mkdir tests 2>nul

echo 📦 Instalando dependências...
pip install -r requirements.txt

echo 📝 Criando arquivos essenciais...
python setup.py

echo 🤖 Criando modelo dummy...
python create_dummy_model.py

echo ✅ Implantação completa!
echo 🎯 Para iniciar: streamlit run main.py
pause