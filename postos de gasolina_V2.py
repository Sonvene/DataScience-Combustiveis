import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def numero_postos_por_regiao(df, ano):
    dt1 = df[df['ANO'] == ano][['REGIÃO', 'NÚMERO DE POSTOS PESQUISADOS']].groupby(['REGIÃO']).sum().sort_values(by='NÚMERO DE POSTOS PESQUISADOS', ascending=False)
    dt1.plot(kind='bar', figsize=(11, 8))
    plt.show()


def numero_postos_por_estado(df, ano):
    dt2 = df[df['ANO'] == ano][['ESTADO', 'NÚMERO DE POSTOS PESQUISADOS']].groupby(['ESTADO']).sum().sort_values(by='NÚMERO DE POSTOS PESQUISADOS', ascending=False)
    dt2.plot(kind='bar', figsize=(11, 8))
    plt.show()


def estado_preco_medio_de_distribuicao(df):
    # Indicação de quando o preço varia no Estado em relação ao Preço Médio de Distribuição.
    dt0 = df[df['ANO'] == 2019][['ESTADO', 'DESVIO PADRÃO DISTRIBUIÇÃO']]
    # Converter a coluna 'DESVIO PADRÃO DISTRIBUIÇÃO' para float64, tratando valores inválidos
    dt0['DESVIO PADRÃO DISTRIBUIÇÃO'] = pd.to_numeric(dt0['DESVIO PADRÃO DISTRIBUIÇÃO'], errors='coerce')
    # Remover valores extremos (-99999) da coluna 'DESVIO PADRÃO DISTRIBUIÇÃO'
    dt0 = dt0[dt0['DESVIO PADRÃO DISTRIBUIÇÃO'] != -99999]
    # Verificar estatísticas descritivas para identificar outliers após remoção
    print(dt0['DESVIO PADRÃO DISTRIBUIÇÃO'].describe())
    # Calcular a média do desvio padrão por estado e plotar o gráfico de barras
    dt1 = dt0.groupby('ESTADO')['DESVIO PADRÃO DISTRIBUIÇÃO'].mean().sort_values(ascending=True)
    dt1.plot(kind='bar', figsize=(11, 7))
    plt.title('Média do Desvio Padrão de Distribuição por Estado em 2020 (sem outliers)')
    plt.xlabel('Estado')
    plt.ylabel('Média do Desvio Padrão de Distribuição')
    plt.show()


def estado_em_relacao_preco_medio_final(df):
    # Indicação de quando o preço varia no Estado em relação ao Preço Médio Final de Venda.
    dt0 = df.query("ANO==2019")[['ESTADO', 'DESVIO PADRÃO REVENDA']]
    dt0 = dt0[pd.to_numeric(dt0['DESVIO PADRÃO REVENDA'], errors='coerce').notnull()]
    dt0['DESVIO_PADRÃO_REVENDA'] = dt0['DESVIO PADRÃO REVENDA'].astype('float64')
    dt1 = dt0[['ESTADO', 'DESVIO PADRÃO REVENDA']].groupby('ESTADO').mean().sort_values(by='DESVIO PADRÃO REVENDA', ascending=True)
    dt1.plot(kind='bar', figsize=(11, 7))
    plt.show()


def variacao_preco_medio_por_estado(df):
    for regiao in df['REGIÃO'].unique():
        for produto in df['PRODUTO'].unique():

            # Variação do Preço Médio de Revenda do PRODUTO por Estado (x) ao Longo dos Anos

            dt1 = df.query("PRODUTO == @produto & REGIÃO == @regiao")
            dt1 = dt1[['ESTADO', 'ANO', 'PREÇO MÉDIO REVENDA', 'DESVIO PADRÃO REVENDA', 'DESVIO PADRÃO DISTRIBUIÇÃO']].groupby(['ESTADO', 'ANO', 'DESVIO PADRÃO REVENDA', 'DESVIO PADRÃO DISTRIBUIÇÃO']).mean().reset_index()

            # Usando pivot_table para organizar os dados para plotagem
            dt2 = dt1.pivot_table(index='ANO', columns='ESTADO', values='PREÇO MÉDIO REVENDA')

            # Plotando o gráfico de barras agrupadas por ano
            plt.figure(figsize=(12, 8))
            dt2.plot(kind='bar', ax=plt.gca(), width=0.8, cmap='tab20')
            plt.title(f'Variação do Preço Médio de Revenda do(a) {produto} por Estado ({regiao}) ao Longo dos Anos')
            plt.xlabel('Ano')
            plt.ylabel('Preço Médio de Revenda')
            plt.legend(title='Estado', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()

            regressao_desvio_de_revenda(dt1, produto)
            regressao_desvio_de_distribuicao(df, produto)

def regressao_desvio_de_revenda(df, produto):
    # Ajuste da Regressão Linear para Desvio Padrão de Revenda ao longo do tempo
    x_revenda = df['ANO'].values.reshape(-1, 1)  # Variável independente (tempo)
    y_revenda = df['DESVIO PADRÃO REVENDA']  # Variável dependente (desvio padrão de revenda)

    # Instanciação e treinamento do modelo de regressão linear para desvio padrão de revenda
    modelo_revenda = LinearRegression()
    modelo_revenda.fit(x_revenda, y_revenda)

    # Coeficientes da regressão para desvio padrão de revenda
    a_coeff_revenda = modelo_revenda.coef_  # Coeficiente angular
    l_coeff_revenda = modelo_revenda.intercept_  # Coeficiente linear

    # Plotagem do gráfico de dispersão com a linha de regressão para desvio padrão de revenda
    plt.scatter(x_revenda, y_revenda)
    plt.xlabel('Ano')
    plt.ylabel('Desvio Padrão Revenda')
    plt.plot(x_revenda, l_coeff_revenda + a_coeff_revenda * x_revenda, color='red', label='Regressão')
    plt.title(f'Regressão Linear - Desvio Padrão de Revenda do(a) {produto} ao Longo do Tempo')
    plt.legend()
    plt.show()


def regressao_desvio_de_distribuicao(df, produto):
    # Ajuste da Regressão Linear para Desvio Padrão de Distribuição ao longo do tempo
    x_distribuicao = df['ANO'].values.reshape(-1, 1)  # Variável independente (tempo)
    y_distribuicao = df['DESVIO PADRÃO DISTRIBUIÇÃO']  # Variável dependente (desvio padrão de distribuição)

    # Instanciação e treinamento do modelo de regressão linear para desvio padrão de distribuição
    modelo_distribuicao = LinearRegression()
    modelo_distribuicao.fit(x_distribuicao, y_distribuicao)

    # Coeficientes da regressão para desvio padrão de distribuição
    a_coeff_distribuicao = modelo_distribuicao.coef_  # Coeficiente angular
    l_coeff_distribuicao = modelo_distribuicao.intercept_  # Coeficiente linear

    # Plotagem do gráfico de dispersão com a linha de regressão para desvio padrão de distribuição
    plt.scatter(x_distribuicao, y_distribuicao)
    plt.xlabel('Ano')
    plt.ylabel('Desvio Padrão Distribuição')
    plt.plot(x_distribuicao, l_coeff_distribuicao + a_coeff_distribuicao * x_distribuicao, color='red',
             label='Regressão')
    plt.title(f'Regressão Linear - Desvio Padrão de Distribuição do(a) {produto} ao Longo do Tempo')
    plt.legend()
    plt.show()


def main():
    df = pd.read_csv('arquivo.csv')

    # tratamento
    df['DATA INICIAL'] = pd.to_datetime(df['DATA INICIAL'])
    df['DATA FINAL'] = pd.to_datetime(df['DATA FINAL'])
    df['DESVIO PADRÃO DISTRIBUIÇÃO'] = pd.to_numeric(df['DESVIO PADRÃO DISTRIBUIÇÃO'], errors='coerce').notnull()
    df['DESVIO PADRÃO REVENDA'] = df['DESVIO PADRÃO REVENDA'].astype('float64')
    df.replace(-99999.000, np.nan, inplace=True)
    df['ANO'] = df['DATA INICIAL'].dt.year

    numero_postos_por_regiao(df, 2018)
    numero_postos_por_estado(df, 2018)

    estado_preco_medio_de_distribuicao(df)
    estado_preco_medio_de_distribuicao(df)

    variacao_preco_medio_por_estado(df)


if __name__ == '__main__':
    main()
