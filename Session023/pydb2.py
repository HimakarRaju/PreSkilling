import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash import dash_table
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
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

    # Row for Graph 1 and Graph 2
    html.Div([
        html.Div([
            html.H3("Graph 1"),
            dcc.Dropdown(id='xaxis-column1', placeholder="Select X-Axis"),
            dcc.Dropdown(id='yaxis-column1', placeholder="Select Y-Axis"),
            dcc.Dropdown(
                id='graph-type1',
                options=[
                    {'label': 'Scatter', 'value': 'scatter'},
                    {'label': 'Bar', 'value': 'bar'},
                    {'label': 'Line', 'value': 'line'},
                    {'label': 'Histogram', 'value': 'histogram'},
                    {'label': 'Boxplot', 'value': 'box'},
                    {'label': 'Pie Chart', 'value': 'pie'},
                    {'label': 'Heatmap', 'value': 'heatmap'}
                ],
                placeholder="Select Graph Type",
            ),
            dcc.Input(id='color-picker1', type='text', placeholder='Enter Color', style={'margin-top': '10px'}),
            dcc.Graph(id='data-graph1', style={'height': '300px'}),
        ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}),
        
        html.Div([
            html.H3("Graph 2"),
            dcc.Dropdown(id='xaxis-column2', placeholder="Select X-Axis"),
            dcc.Dropdown(id='yaxis-column2', placeholder="Select Y-Axis"),
            dcc.Dropdown(
                id='graph-type2',
                options=[
                    {'label': 'Scatter', 'value': 'scatter'},
                    {'label': 'Bar', 'value': 'bar'},
                    {'label': 'Line', 'value': 'line'},
                    {'label': 'Histogram', 'value': 'histogram'},
                    {'label': 'Boxplot', 'value': 'box'},
                    {'label': 'Pie Chart', 'value': 'pie'},
                    {'label': 'Heatmap', 'value': 'heatmap'}
                ],
                placeholder="Select Graph Type",
            ),
            dcc.Input(id='color-picker2', type='text', placeholder='Enter Color', style={'margin-top': '10px'}),
            dcc.Graph(id='data-graph2', style={'height': '300px'}),
        ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}),
    ]),

    # Row for Graph 3 and Graph 4
    html.Div([
        html.Div([
            html.H3("Graph 3"),
            dcc.Dropdown(id='xaxis-column3', placeholder="Select X-Axis"),
            dcc.Dropdown(id='yaxis-column3', placeholder="Select Y-Axis"),
            dcc.Dropdown(
                id='graph-type3',
                options=[
                    {'label': 'Scatter', 'value': 'scatter'},
                    {'label': 'Bar', 'value': 'bar'},
                    {'label': 'Line', 'value': 'line'},
                    {'label': 'Histogram', 'value': 'histogram'},
                    {'label': 'Boxplot', 'value': 'box'},
                    {'label': 'Pie Chart', 'value': 'pie'},
                    {'label': 'Heatmap', 'value': 'heatmap'}
                ],
                placeholder="Select Graph Type",
            ),
            dcc.Input(id='color-picker3', type='text', placeholder='Enter Color', style={'margin-top': '10px'}),
            dcc.Graph(id='data-graph3', style={'height': '300px'}),
        ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}),
        
        html.Div([
            html.H3("Graph 4"),
            dcc.Dropdown(id='xaxis-column4', placeholder="Select X-Axis"),
            dcc.Dropdown(id='yaxis-column4', placeholder="Select Y-Axis"),
            dcc.Dropdown(
                id='graph-type4',
                options=[
                    {'label': 'Scatter', 'value': 'scatter'},
                    {'label': 'Bar', 'value': 'bar'},
                    {'label': 'Line', 'value': 'line'},
                    {'label': 'Histogram', 'value': 'histogram'},
                    {'label': 'Boxplot', 'value': 'box'},
                    {'label': 'Pie Chart', 'value': 'pie'},
                    {'label': 'Heatmap', 'value': 'heatmap'}
                ],
                placeholder="Select Graph Type",
            ),
            dcc.Input(id='color-picker4', type='text', placeholder='Enter Color', style={'margin-top': '10px'}),
            dcc.Graph(id='data-graph4', style={'height': '300px'}),
        ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}),
    ]),
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
     Output('xaxis-column1', 'options'),
     Output('yaxis-column1', 'options'),
     Output('xaxis-column2', 'options'),
     Output('yaxis-column2', 'options'),
     Output('xaxis-column3', 'options'),
     Output('yaxis-column3', 'options'),
     Output('xaxis-column4', 'options'),
     Output('yaxis-column4', 'options')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_output(contents, filename):
    if contents is None:
        return '', [], [], [], [], [], [], [], []

    # Parse the uploaded file
    df = parse_data(contents, filename)

    if df is None:
        return "There was an error processing the file.", [], [], [], [], [], [], [], []

    # Create options for dropdowns
    options = [{'label': col, 'value': col} for col in df.columns]

    return dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in df.columns],
        page_size=10
    ), options, options, options, options, options, options, options, options

# Helper function to create different graph types
def create_graph(graph_type, df, x, y, color):
    if graph_type == 'scatter':
        return px.scatter(df, x=x, y=y, color_discrete_sequence=[color])
    elif graph_type == 'bar':
        return px.bar(df, x=x, y=y, color_discrete_sequence=[color])
    elif graph_type == 'line':
        return px.line(df, x=x, y=y, color_discrete_sequence=[color])
    elif graph_type == 'histogram':
        return px.histogram(df, x=x, color_discrete_sequence=[color])
    elif graph_type == 'box':
        return px.box(df, x=x, y=y, color_discrete_sequence=[color])
    elif graph_type == 'pie':
        return px.pie(df, names=x, values=y, color_discrete_sequence=[color])
    elif graph_type == 'heatmap':
        corr = df.corr()
        return ff.create_annotated_heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            colorscale='Viridis'
        )
    else:
        return {}

# Update the graphs based on selected columns, graph type, and color
@app.callback(
    [Output('data-graph1', 'figure'),
     Output('data-graph2', 'figure'),
     Output('data-graph3', 'figure'),
     Output('data-graph4', 'figure')],
    [Input('xaxis-column1', 'value'), Input('yaxis-column1', 'value'), Input('graph-type1', 'value'), Input('color-picker1', 'value'),
     Input('xaxis-column2', 'value'), Input('yaxis-column2', 'value'), Input('graph-type2', 'value'), Input('color-picker2', 'value'),
     Input('xaxis-column3', 'value'), Input('yaxis-column3', 'value'), Input('graph-type3', 'value'), Input('color-picker3', 'value'),
     Input('xaxis-column4', 'value'), Input('yaxis-column4', 'value'), Input('graph-type4', 'value'), Input('color-picker4', 'value')],
    [State('upload-data', 'contents'), State('upload-data', 'filename')]
)
def update_graph(x1, y1, type1, color1, x2, y2, type2, color2, x3, y3, type3, color3, x4, y4, type4, color4, contents, filename):
    if contents is None:
        return {}, {}, {}, {}

    df = parse_data(contents, filename)

    if df is None:
        return {}, {}, {}, {}

    fig1 = create_graph(type1, df, x1, y1, color1 or 'blue')
    fig2 = create_graph(type2, df, x2, y2, color2 or 'green')
    fig3 = create_graph(type3, df, x3, y3, color3 or 'red')
    fig4 = create_graph(type4, df, x4, y4, color4 or 'purple')

    return fig1, fig2, fig3, fig4

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
