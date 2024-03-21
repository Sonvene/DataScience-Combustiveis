import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

sns.set_style('darkgrid')
warnings.filterwarnings('ignore')

data = pd.read_csv('arquivo.csv')

# Limpeza dos dados
data['DATA INICIAL'] = pd.to_datetime(data['DATA INICIAL'])
data['DATA FINAL'] = pd.to_datetime(data['DATA FINAL'])

atributos_numericos = ['MARGEM MÉDIA REVENDA', 'PREÇO MÉDIO DISTRIBUIÇÃO', 'DESVIO PADRÃO DISTRIBUIÇÃO',
                       'PREÇO MÍNIMO DISTRIBUIÇÃO', 'PREÇO MÁXIMO DISTRIBUIÇÃO', 'COEF DE VARIAÇÃO DISTRIBUIÇÃO']

for atributo in atributos_numericos:
    data[atributo] = pd.to_numeric(data[atributo], errors='coerce')

# Preenchimento de valores nulos e remoção das linhas com valores nulos
data.fillna(0)
data.dropna(inplace=True)

for feature in data.columns:
    print('{} \t {:.1f}% valores ausentes'.format(feature, (data[feature].isnull().sum() / len(data)) * 100))


# Visualização da distribuição do número de postos pesquisados por região
plt.figure(figsize=(10, 6))
sns.barplot(x='REGIÃO', y='NÚMERO DE POSTOS PESQUISADOS', data=data)
plt.title('Distribuição do Número de Postos Pesquisados por Região')
plt.xlabel('Região')
plt.ylabel('Número de Postos Pesquisados')
plt.show()


numero_de_postos = data['NÚMERO DE POSTOS PESQUISADOS']
regioes = ['NORDESTE', 'NORTE', 'SUDESTE', 'CENTRO OESTE', 'SUL']
produtos = ['GASOLINA COMUM', 'ETANOL HIDRATADO', 'GLP', 'ÓLEO DIESEL',
            'GNV', 'ÓLEO DIESEL S10', 'OLEO DIESEL S10', 'OLEO DIESEL', 'GASOLINA ADITIVADA']

# há um valor de -99999.000 pode indicar um erro de registro ou uma condição especial que precisa ser investigada.
# porém irei substituir os valores -99999.000 por NaN por via das duvidas
data.replace(-99999.000, np.nan, inplace=True)

for regiao in regioes:
    for produto in produtos:
        # Filtrando os dados para a região e produto atual
        dados_filtrados = data[(data['REGIÃO'] == regiao) & (data['PRODUTO'] == produto)]

        if not dados_filtrados.empty:

            # Boxplot para distribuição de preços médios de revenda
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='REGIÃO', y='PREÇO MÉDIO REVENDA', data=dados_filtrados)
            plt.title(f'Distribuição de Preços Médios de Revenda de {produto} na Região {regiao}')
            plt.xlabel('Região')
            plt.ylabel('Preço Médio de Revenda (R$/l)')
            plt.show()


            # Gráficos para distribuição de preços mínimos e máximos de revenda
            for extremo in ['MÍNIMO', 'MÁXIMO']:
                plt.figure(figsize=(10, 6))
                sns.barplot(x='ESTADO', y=f'PREÇO {extremo} REVENDA', data=dados_filtrados)
                plt.title(f'Distribuição de Preços {extremo.capitalize()}s de Revenda de {produto} na Região {regiao}')
                plt.xlabel('Estados')
                plt.ylabel(f'Preço {extremo.capitalize()} de Revenda')
                plt.xticks(rotation=90)
                plt.show()


            # Verificando a distribuição dos preços médios de revenda
            plt.figure(figsize=(10, 6))
            sns.histplot(dados_filtrados['PREÇO MÉDIO REVENDA'], bins=20, kde=True)
            plt.title(f'Distribuição dos Preços Médios de Revenda sobre {produto}')
            plt.xlabel('Preço Médio de Revenda (R$/l)')
            plt.ylabel('Frequência')
            plt.show()


            # Verificando a relação entre preço médio de revenda e margem média de revenda
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='PREÇO MÉDIO REVENDA', y='MARGEM MÉDIA REVENDA', data=dados_filtrados)
            plt.title(f'Relação entre Preço Médio de Revenda e Margem Média de Revenda sobre {produto}')
            plt.xlabel('Preço Médio de Revenda (R$/l)')
            plt.ylabel('Margem Média de Revenda (R$/l)')
            plt.show()


            # Analisando a variação dos preços médios de revenda ao longo do tempo
            plt.figure(figsize=(12, 6))
            sns.lineplot(x='DATA INICIAL', y='PREÇO MÉDIO REVENDA', data=dados_filtrados, hue='ESTADO')
            plt.title(f'Variação dos Preços Médios de Revenda ao Longo do Tempo sobre {produto}')
            plt.xlabel('Data')
            plt.ylabel('Preço Médio de Revenda (R$/l)')
            plt.xticks(rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()

        else:
            print(f"Não há dados para {produto} na região {regiao}")


# Comparação da margem média de revenda por região
plt.figure(figsize=(10, 6))
sns.barplot(x='REGIÃO', y='MARGEM MÉDIA REVENDA', data=data)
plt.title('Margem Média de Revenda por Região')
plt.xlabel('Região')
plt.ylabel('Margem Média de Revenda')
plt.show()


# Matriz de correlação entre as variáveis numéricas.
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
corr_matrix = data[numeric_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlação')
plt.show()




