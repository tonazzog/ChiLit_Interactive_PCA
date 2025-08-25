import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
import pickle
import pandas as pd
import json


chunk_size = 200
data_folder="./data/"

# ChiLit books
df_metadata = pd.read_csv(f"{data_folder}ChiLit_metadata.csv", encoding="utf-8")
df_chilit = pd.read_csv(f"{data_folder}ChiLit_Chunks_{chunk_size}.csv")
df_chilit = df_chilit.fillna("")

# ProdLDA model optimized by OPTUNA
final_model = pickle.load(open(f"{data_folder}Optuna_ProdLDA_output.pkl", "rb"))

# PCA from topic document matris
topic_doc_matrix = final_model['topic-document-matrix'].T
chunk_to_text_map = df_chilit['book_id'].to_list()
pca = PCA(n_components=2)
coords = pca.fit_transform(topic_doc_matrix)

# Create a DataFrame for easier handling with Plotly
df_plot = pd.DataFrame({
    'PC1': coords[:, 0],
    'PC2': coords[:, 1],
    'book_id': chunk_to_text_map,
    'point_id': range(len(chunk_to_text_map))
})

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Interactive PCA Scatter Plot", style={'text-align': 'center'}),

    html.Div([
        html.Label("Select Books to Highlight:"),
        dcc.Dropdown(
            id='book-dropdown',
            options=[{'label': book, 'value': book} for book in sorted(set(chunk_to_text_map))],
            value=['squirrel', 'bunny', 'mice', 'rabbit', 'jemima', 'flopsy'],  # Default to Beatrice Potter's books
            multi=True,
            style={'margin-bottom': '20px'}
        )
    ], style={'width': '48%', 'display': 'inline-block'}),

    dcc.Graph(id='scatter-plot', style={'width': '1000px', 'height': '1200px'})
])

@app.callback(
    Output('scatter-plot', 'figure'),
    Input('book-dropdown', 'value')
)
def update_scatter_plot(selected_books):
    fig = go.Figure()

    unique_books = sorted(set(chunk_to_text_map))
    colors = px.colors.qualitative.Set3

    for i, book in enumerate(sorted(unique_books)):
        book_mask = df_plot['book_id'] == book
        book_data = df_plot[book_mask]

        # Determine opacity and size based on selection
        if book in selected_books:
            opacity = 0.8
            size = 10
            line_width = 2
        else:
            opacity = 0.2
            size = 6
            line_width = 0

        fig.add_trace(go.Scatter(
            x=book_data['PC1'],
            y=book_data['PC2'],
            mode='markers',
            name=book,
            marker=dict(
                color=colors[i % len(colors)],
                size=size,
                opacity=opacity,
                line=dict(width=line_width, color='darkslategray')
            ),
            hovertemplate=f'<b>{book}</b><br>' +
                         'PC1: %{x:.3f}<br>' +
                         'PC2: %{y:.3f}<br>' +
                         '<extra></extra>'
        ))

    fig.update_layout(
        title='Documents in Topic Space (PCA)',
        xaxis_title='PC1',
        yaxis_title='PC2',
        hovermode='closest',
        legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5)
    )

    return fig

if __name__ == '__main__':
    app.run(debug=True)