import gradio as gr
import pandas as pd
from io import StringIO
import numpy as np
import scipy.optimize
import random

import plotly.graph_objects as go
import plotly
import plotly.express as px
import plotly.io as pio
pio.templates.default = 'simple_white'

def get_retention(a, b, c, d, periods):
    return  a + 1./(b + c*periods ** d)

def get_retention_same_event(a, c, d, periods):
    b = 1./(1 - a)
    return get_retention(a, b, c, d, periods)

def get_mse_for_retention(params, df):
    tmp_df = df.copy()
    tmp_df['retention_pred'] = tmp_df.index.map(
        lambda x: get_retention_same_event(params[0], params[1], params[2], x)
    )
    
    tmp_df['se'] = (tmp_df.retention_fact - tmp_df.retention_pred)
    tmp_df['se'] = tmp_df['se']**2
    
    return tmp_df.se.mean() ** 0.5

def parse_file(temp_file, num_periods):
    if isinstance(temp_file, str):
        df = pd.read_csv(StringIO(temp_file), sep = '\t')
    else:
        df = pd.read_csv(temp_file.name, sep = '\t')
    return df

def show_graph_for_df(df, num_periods):
    print(df)
    df['period'] = df.period.map(int)
    df['retention_fact'] = df.retention_fact.map(float)
    result = scipy.optimize.minimize(lambda x: get_mse_for_retention(x, df), [random.random(), random.random(), random.random()])
    a, c, d = result.x
    print(a, c, d)

    pred_df = pd.DataFrame({'period': range(num_periods + 1)})
    pred_df['retention_pred'] = pred_df.period.map(lambda x: get_retention_same_event(a, c, d, x))
    pred_df = pred_df.merge(df, how = 'left')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred_df.period, y=pred_df.retention_fact, name='fact',
                             line=dict(color=plotly.colors.qualitative.Prism[0], width=3)))
    
    fig.add_trace(go.Scatter(x=pred_df.period, y=pred_df.retention_pred, name='prediction',
                             line=dict(color=plotly.colors.qualitative.Prism[0], width=3, dash='dot')))
    
    fig.update_layout(title='Daily retention model (a = %.2f, c = %.2f, d = %.2f)' % (a, c, d),
                       yaxis_title='retention',
                       xaxis_title='period')
    return fig

def show_graph_for_file(temp_file, num_periods):
    df = parse_file(temp_file, num_periods)
    return show_graph_for_df(df, num_periods)

default_csv = 'period\tretention_fact\n0\t1\n1\t0.55\n2\t0.4\n3\t0.35\n4\t0.3\n'

with gr.Blocks() as demo:
    gr.Markdown('# Predicting retention curve 📊')
    periods = gr.Dropdown([10, 30, 90, 180], label="Number of Periods", value = 30)
    gr.Markdown('Upload .csv file with data, use default data as an example or put in numbers manually in the Uploaded data section.')
    gr.Markdown('''__File format:__ 2 columns (`period` and `retention_fact`)''')
    with gr.Row():
        upload_button = gr.UploadButton(label="Upload file", file_types = ['.csv'], live=True, file_count = "single")
        default_button = gr.Button('Show example')
    
    with gr.Row():
        with gr.Accordion("Uploaded data", open=False):
            gr.Markdown('You can change values in the table')
            table = gr.Dataframe(type="pandas", col_count=2, interactive = True, headers = ['period', 'retention_fact'])
            
    with gr.Row():    
        image = gr.Plot()    

    upload_button.upload(fn=show_graph_for_file, inputs=[upload_button, periods], outputs=image, api_name="upload_graph")
    upload_button.upload(fn=parse_file, inputs=[upload_button, periods], outputs=table, api_name="upload_csv")
    default_button.click(fn=lambda x: show_graph_for_file(default_csv, x), inputs=[periods], outputs=image, api_name="upload_example_graph")
    default_button.click(fn=lambda x: parse_file(default_csv, x), inputs=[periods], outputs=table, api_name="upload_example_csv")
    table.change(fn=show_graph_for_df, inputs=[table, periods], outputs=image, api_name="upload_table_graph")
    periods.change(fn=show_graph_for_df, inputs=[table, periods], outputs=image, api_name="upload_table_graph")

demo.launch(debug=True)