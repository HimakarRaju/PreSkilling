import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_table
import pandas as pd
import plotly.express as px
import base64
import io

# Initialize Dash app
app = dash.Dash(__name__)

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1("Data Analytics Dashboard", style={'text-align': 'center'}),

    # File upload component
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '50%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px auto'
        },
        multiple=False
    ),

    # Div to display the uploaded file's data
    html.Div(id='output-data-upload'),

    # Dropdown to select columns for visualization
    html.Div([
        dcc.Dropdown(id='xaxis-column', placeholder="Select X-Axis"),
        dcc.Dropdown(id='yaxis-column', placeholder="Select Y-Axis"),
    ], style={'width': '50%', 'margin': '10px auto'}),

    # Graph to display the analytics
    dcc.Graph(id='data-graph'),
])

# Helper function to parse CSV or Excel
def parse_data(contents, filename):
    content_type, content_string = contents.split(',')
    
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return None
    except Exception as e:
        print(e)
        return None

    return df

# Update the table and dropdowns after file upload
@app.callback(
    [Output('output-data-upload', 'children'),
     Output('xaxis-column', 'options'),
     Output('yaxis-column', 'options')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_output(contents, filename):
    if contents is None:
        return '', [], []

    # Parse the uploaded file
    df = parse_data(contents, filename)

    if df is None:
        return "There was an error processing the file.", [], []

    # Create options for dropdowns
    options = [{'label': col, 'value': col} for col in df.columns]

    return dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in df.columns],
        page_size=10
    ), options, options

# Update the graph based on selected columns
@app.callback(
    Output('data-graph', 'figure'),
    [Input('xaxis-column', 'value'),
     Input('yaxis-column', 'value')],
    [State('upload-data', 'contents'),
     State('upload-data', 'filename')]
)
def update_graph(xaxis_column, yaxis_column, contents, filename):
    if contents is None or xaxis_column is None or yaxis_column is None:
        return {}
    
    df = parse_data(contents, filename)

    if df is None:
        return {}

    # Create the plotly express figure
    fig = px.scatter(df, x=xaxis_column, y=yaxis_column)
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
