import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output

# Carregar os dados das cidades
column_types = {
    'City': str,
    'Latitude': float,
    'Longitude': float,
    'Population': float  # Ajuste esta chave conforme as colunas reais do seu CSV
}
cities_df = pd.read_csv('worldcitiespop.csv', dtype=column_types, low_memory=False)

# Coordenadas aproximadas de Rio Grande
lat_rio_grande, lon_rio_grande = -32.035, -52.0986
is_rio_grande = (cities_df['Latitude'].between(lat_rio_grande - 0.05, lat_rio_grande + 0.05) &
                 cities_df['Longitude'].between(lon_rio_grande - 0.05, lon_rio_grande + 0.05))
rio_grande = cities_df[is_rio_grande].copy()

if not rio_grande.empty:
    rio_grande.loc[:, 'risco'] = 100  # Supondo alto risco para Rio Grande
    rio_grande.loc[:, 'Classificação de Risco'] = 'Alerta'
    cities_df = pd.concat([cities_df, rio_grande])

# Simular riscos das cidades com uma distribuição normal (realista)
np.random.seed(0)
cities_df['risco'] = np.random.normal(loc=50, scale=20, size=len(cities_df))
cities_df['risco'] = cities_df['risco'].clip(0, 100)  # Limitar os valores de risco entre 0 e 100

# Adicionar uma coluna de classificação de risco com base nos níveis de risco
cities_df['Classificação de Risco'] = pd.cut(cities_df['risco'],
                                             bins=[-np.inf, 25, 50, 75, np.inf],
                                             labels=['Baixo', 'Moderado', 'Alto', 'Alerta'])

# Simular agrupamento de risco usando KMeans
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(cities_df[['Latitude', 'Longitude', 'risco']])
cities_df['Cluster'] = clusters

# Ordenar as cidades dentro de cada cluster por risco
cities_df_sorted = cities_df.sort_values(by=['Cluster', 'risco'], ascending=[True, False])

# Criar a figura 3D interativa
fig = go.Figure()
for cluster_id in range(5):
    cluster_cities = cities_df_sorted[cities_df_sorted['Cluster'] == cluster_id]
    fig.add_trace(go.Scatter3d(
        x=cluster_cities['Longitude'], y=cluster_cities['Latitude'], z=cluster_cities['risco'],
        mode='markers',
        marker=dict(
            size=12 if 'Rio Grande' in cluster_cities['City'].values else 5,
            color=cluster_id,  # color points by cluster they belong to
            colorscale='Viridis',  # choose a color scale
            opacity=0.8
        ),
        text=cluster_cities['City'],  # add city names as labels
        name=f'Cluster {cluster_id}'
    ))

# Configurar layout do gráfico
fig.update_layout(
    title='Simulação de Risco de Rompimento de Barragem com Direção do Vento',
    scene=dict(
        xaxis_title='Longitude',
        yaxis_title='Latitude',
        zaxis_title='Nível de Risco (%)'
    ),
    legend_title="Legenda",
    autosize=False,
    width=800,
    height=600
)

# Criar a aplicação Dash
app = dash.Dash(__name__)

# Layout da aplicação Dash
app.layout = html.Div([
    html.H1("Tabela de Clusters de Risco e Probabilidade de Progressão"),
    html.Div([
        dcc.Graph(id='3d-map', figure=fig),
    ]),
    html.Div([
        html.H3("Tabela de Todas as Cidades"),
        dash_table.DataTable(
            id='all-cities-table',
            columns=[
                {"name": "Cidade", "id": "City"},
                {"name": "Nível de Risco", "id": "risco"},
                {"name": "Classificação de Risco", "id": "Classificação de Risco"}
            ],
            data=cities_df_sorted.to_dict('records'),
            sort_action='native',
            style_table={'overflowX': 'scroll'}
        )
    ])
])

# Executar a aplicação Dash
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
