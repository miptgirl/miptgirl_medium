import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = 'simple_white'

def get_retention(a, b, c, d, periods):
    return  a + 1./(b + c*periods ** d)

def get_retention_same_event(a, c, d, periods):
    b = 1./(1 - a)
    return get_retention(a, b, c, d, periods)

def get_retention_plots(a, c, d, num_periods, cohort_size):
    ret_df = pd.DataFrame({'x': range(num_periods + 1)})
    ret_df['retention'] = ret_df.x.map(lambda x: get_retention_same_event(a, c, d, x))
    
    ret_fig = px.line(ret_df.iloc[1:], x = 'x', y = 'retention', color_discrete_sequence = px.colors.qualitative.Prism, 
                      title = 'Retention curve', labels = {'x': 'period'})

    # simulation

    tmp_data = []

    for cohort in range(num_periods + 1):
        for cohort_period in range(num_periods + 1):
            period = cohort_period + cohort
            if period > num_periods:
                continue
            retention = get_retention_same_event(a, c, d, cohort_period)
            tmp_data.append(
                {
                    'cohort': 'cohort %s' % str(cohort).rjust(3, '0'),
                    'cohort_period': cohort_period,
                    'period': period,
                    'retention': retention,
                    'users': int(round(retention * cohort_size))
                }
            )
    users_df = pd.DataFrame(tmp_data)

    users_fig = px.area(users_df.groupby('period').users.sum(),
                    color_discrete_sequence = px.colors.qualitative.Prism, 
                      title = 'Active users', labels = {'value': 'active users'})

    cohorts_fig = px.area(users_df.pivot_table(index = 'period', columns = 'cohort', values = 'users',
                    aggfunc = 'sum'),
                    color_discrete_sequence = px.colors.qualitative.Prism, 
                      title = 'Active users by cohorts', labels = {'value': 'active users'})
                        

    return ret_fig, users_fig, cohorts_fig

with gr.Blocks() as demo:
    gr.Markdown("# Understanding Growth 🚀")
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Retention curve parameters 📈")
            gr.Markdown(r"$\textbf{retention}(\textsf{x}) = \textsf{a} + \frac{\textsf{1}}{\textsf{b} + \textsf{c} * \textsf{x}^{\textsf{d}}}\ where\ \textsf{b} = \frac{\textsf{1}}{\textsf{1}-\textsf{a}}$")
            with gr.Row():
                a = gr.Slider(0, 1, 0.03, label="a")
                c = gr.Slider(0, 5, 0.55, label="c")
                d = gr.Slider(0, 5, 1.5, label="d")
            with gr.Accordion("More options", open=False):
                with gr.Row():
                    num_periods = gr.Dropdown([10, 30, 60, 90], value = 30, label="Number of Periods")
                    cohort_size = gr.Dropdown([10, 100, 1000, 10000], value = 10000, label="Number of new users each period")
            btn_caption = gr.Button("Submit")
        with gr.Column():
            plot1 = gr.Plot()
    with gr.Row():
        plot2 = gr.Plot()
        plot3 = gr.Plot()
    btn_caption.click(fn=get_retention_plots, inputs=[a, c, d, num_periods, cohort_size], outputs=[plot1, plot2, plot3])

gr.close_all()
demo.launch(share = True)