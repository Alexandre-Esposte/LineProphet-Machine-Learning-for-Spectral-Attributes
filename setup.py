from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()



setup(
    name='SpectraHitran',
    version='0.1.0',  # Use versão semântica (ex: 0.1.0)
    author='Alexandre Esposte',
    author_email='alexandreesposte@id.uff.br',
    description='Geração, processamento e ajuste de espectros de absorção utilizando a Hitran API',
    packages=find_packages(),  # Encontra pacotes automaticamente
    install_requires=requirements,  # Usa as libs do requirements.txt
    include_package_data=True,  # Inclui arquivos não-Python (opcional)
    python_requires='>=3.10.2',  # Versão mínima do Python
)