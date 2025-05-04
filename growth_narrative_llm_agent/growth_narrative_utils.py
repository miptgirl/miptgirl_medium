import pandas as pd 
import numpy as np 
import random
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"

# visualisations

def create_share_vs_impact_chart(df, dimension, share_field, impact_field):
    """
    Creates an interactive scatter plot chart using Plotly to compare the size of the segments vs its impact on the metric change
    
    Parameters:
    -----------
    df : pandas DataFrame
        Data containing share 
    dimension : str
        Column name of the parameter/segment variable
    share_field : str
        Column name for the share of segment values
    impact_field : str
        Column name for the impact on the metric change values
    """
    plot_df = df.copy() # cloning df for manipulations
    plot_df['impact_coef'] = plot_df[impact_field]/plot_df[share_field]

    colorscale = px.colors.qualitative.D3
    fig = go.Figure()
    
    # Add scatter plot with conditional colors
    for i, row in plot_df.iterrows():
        marker_color = colorscale[2] if row['impact_coef'] > 1.5 else (colorscale[3] if row['impact_coef'] < -0.5 else colorscale[0]) 
        
        fig.add_trace(go.Scatter(
            x=[row[share_field]],
            y=[row[impact_field]],
            mode='markers',
            marker=dict(
                size=10,  # Larger marker size
                color=marker_color,
                opacity=0.75
            ),
            showlegend=False
        ))
    
    
    # Add the x=y reference line
    max_val = max(max(plot_df[share_field].values), max(plot_df[impact_field].values))
    min_val = min(min(plot_df[share_field].values), min(plot_df[impact_field].values))
    line_range = [min_val, max_val]
    
    fig.add_trace(
        go.Scatter(
            x=line_range,
            y=line_range,
            mode="lines",
            line=dict(dash="dash", color="gray", width=1.5),
        )
    )

    # Add annotations to the chart
    for i, row in plot_df.iterrows():
        fig.add_annotation(
            x=row[share_field],
            y=row[impact_field],
            text=f"{row[dimension]}",
            showarrow=False,
            xanchor='center',   
            yanchor='bottom',  
            font=dict(size=10),
            yshift=5,  
        )
    
    # Show the plot
    fig.update_layout(
        title="<b>Metric change explained:</b> correlation between segment size and impact on the change",
        xaxis_title="Share of segment before, %",
        yaxis_title="Share in difference, %",
        template="plotly_white",
        height=600,
        width=800, 
        showlegend = False)
    fig.show()

def create_parallel_coordinates_chart(df, dimension, before_field='before', 
                                      after_field='after', impact_coef_field = 'impact_coef'):
    """
    Creates an interactive parallel coordinates chart using Plotly
    
    Parameters:
    -----------
    df : pandas DataFrame
        Data containing before/after values per segment
    dimension : str
        Column name of the parameter/segment variable
    before_field : str
        Column name for the 'before' values
    after_field : str
        Column name for the 'after' values
    impact_coef_field : str
        Column name for the impact coefficient values (the ratio of impact to the segment size)
    """
    # Create a copy of the dataframe for manipulation
    plot_df = df.copy()
    
    # Define color mapping for params
    dimensions = plot_df[dimension].unique()
    colorscale = px.colors.qualitative.D3
    colors = [colorscale[i % len(colorscale)] for i in range(len(dimensions))]
    color_map = dict(zip(dimensions, colors))
    plot_df['color'] = plot_df[dimension].map(color_map)
    
    # Create accents on meaningful changes using line width and opacity
    plot_df['line_width'] = plot_df.impact_coef.map(
        lambda x: 4 if (x > 1.5) or (x < -0.5) else 2
    )
    plot_df['opacity'] = plot_df.impact_coef.map(
        lambda x: 1 if (x > 1.5) or (x < -0.5) else 0.6
    )
    
    # Create the figure
    fig = go.Figure()
    
    # Calculate mean values for reference line
    mean_before = plot_df[before_field].mean()
    mean_after = plot_df[after_field].mean()
    
    # Add mean reference line
    fig.add_trace(
        go.Scatter(
            x=['BEFORE', 'AFTER'],
            y=[mean_before, mean_after],
            mode='lines',
            line=dict(color='gray', width=1.5, dash='dash'),
            opacity=0.7,
            name='Average',
            showlegend=False # remove from legend
        )
    )
    
    # Add lines for each parameter value
    for idx, row in plot_df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=['BEFORE', 'AFTER'],
                y=[row[before_field], row[after_field]],
                mode='lines+markers',
                line=dict(
                    color=row['color'],
                    width=row['line_width']
                ),
                opacity=row['opacity'],
                name=f"{row[dimension]}",
                marker=dict(size=8),
            )
        )
    
    # Update layout
    fig.update_layout(
        title= '<b>Metric change explained:</b> before vs after',
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=12, weight='bold')
        ),
        yaxis=dict(
            title='Value',
            showgrid=True,
            gridcolor='rgba(211, 211, 211, 0.7)',
            gridwidth=1,
            tickformat='.0s'  # Automatically format large numbers (K, M)
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.25,
            xanchor='center',
            x=0.5,
            font=dict(size=10)
        ),
        plot_bgcolor='white',
        width=800,
        height=600,
        margin=dict(l=60, r=30, t=80, b=120)
    )
    
    fig.show()

def hex_to_rgba(hex_color, alpha=None):
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    if alpha is not None:
        return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})"
    else:
        return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"

def plot_conversion_waterfall(rate_before, rate_after, df):
    """
    Creates a waterfall chart showing contribution of dimension effects to conversion change.

    Parameters:
    -----------
    rate_before : float
        Initial conversion rate
    rate_after : float
        Final conversion rate
    df : pandas DataFrame
        DataFrame indexed by dimensions, with a column "effect" showing contribution
    """

    plot_df = df.copy()
    plot_df = plot_df[plot_df.effect != 0] # filtering out segments without effect
    colorscale = px.colors.qualitative.D3

    # Build the waterfall steps
    dimensions = plot_df.index.tolist()
    effects = plot_df["effect"].tolist()
    

    base = rate_before
    total_effect = sum(effects)
    remaining_effect = (rate_after - rate_before) - total_effect
    
    if remaining_effect >= 0.1: 
        dimensions.append('remaining effects')
        effects.append(remaining_effect) 
    total_effect = sum(effects)

    # Add starting point
    x = ["Before"] + dimensions + ["After"]
    y = [rate_before] + effects + [rate_after - (rate_before + total_effect)]

    measures = ["absolute"] + ["relative"] * len(dimensions) + ["total"]

    fig = go.Figure(go.Waterfall(
        name="Conversion",
        orientation="v",
        measure=measures,
        textposition="outside",
        x=x,
        y=y, 
        text= ['%.1f%%' % rate_before] + list(map(lambda x: '+%.1f%%' % x if x > 0 else '%.1f%%' % x, effects)) + ['%.1f%%' % rate_after],
        connector={"line": {"color": "gray"}},
        increasing={"marker": {"color": hex_to_rgba(colorscale[2], 0.7)}},
        decreasing={"marker": {"color": hex_to_rgba(colorscale[3], 0.7)}},
        totals={"marker": {"color": hex_to_rgba(colorscale[0], 0.7)}}
    ))

    fig.update_layout(
        title="<b>Conversion rate change explained</b>",
        yaxis_title="conversion, %",
        waterfallgap=0.4,
        margin=dict(t=100)
    )
    
    plot_df = plot_df.sort_values('effect', ascending = False)
    plot_df['cum_effect'] = plot_df.effect.cumsum()
    max_val = max([rate_before, rate_after, rate_before + plot_df['cum_effect'].max()]) + 10

    # max_val = rate_before + plot_df['cum_effect'].max() + 10  # adjust buffer
    fig.update_yaxes(range=[0, max_val])

    fig.show()

# analysis

def calculate_simple_growth_metrics(stats_df):
    """
    Analyses the change of simple metrics before and after
    
    Parameters:
    -----------
    df : pandas DataFrame
        Data containing before/after values per segment:
            - dimension is in index; 
            - data frame has columns "before" and "after"
    """

    # Calculating overall stats
    before = stats_df.before.sum()
    after = stats_df.after.sum()
    print('Metric change: %.2f -> %.2f (%.2f%%)' % (before, after, 100*(after - before)/before))

    # Estimating impact of each segment
    stats_df['diff_abs'] = stats_df.after - stats_df.before
    stats_df['diff_rate'] = (100*stats_df.diff_abs/stats_df.before).map(lambda x: round(x, 2))
    stats_df['share_in_diff'] = (100*stats_df.diff_abs / stats_df.diff_abs.sum()).map(lambda x: round(x, 2))
    stats_df['share_before'] = (100* stats_df.before / stats_df.before.sum()).map(lambda x: round(x, 2))
    stats_df['impact_coef'] = (stats_df.share_in_diff/stats_df.share_before).map(lambda x: round(x, 2))
    stats_df['abs_impact_coef'] = stats_df.impact_coef.map(abs)

    # Sorting based on the impact normed on the size
    stats_df = stats_df.sort_values('abs_impact_coef', ascending = False)
    stats_df = stats_df.drop('abs_impact_coef', axis = 1)

    # Creating visualisations
    create_parallel_coordinates_chart(stats_df.reset_index(), stats_df.index.name)
    create_share_vs_impact_chart(stats_df.reset_index(), stats_df.index.name, 'share_before', 'share_in_diff', )
    
    return stats_df.sort_values('impact_coef', ascending = False)

def calculate_conversion_effects(df, dimension, numerator_field1, denominator_field1, 
                       numerator_field2, denominator_field2):
    """
    Analyses the change of conversion metrics before and after
    
    Parameters:
    -----------
    df : pandas DataFrame
        Data containing before/after values per segment
    dimension : str
        Column name of the parameter/segment variable
    numerator_field1: str
        Column name of the numerator field before
    denominator_field1: str
        Column name of the denominator field before
    numerator_field2: str
        Column name of the numerator field after
    denominator_field2: str
        Column name of the denominator field after
    """
    
    cmp_df = df.groupby(dimension)[[numerator_field1, denominator_field1, numerator_field2, denominator_field2]].sum()
    cmp_df = cmp_df.rename(columns = {
        numerator_field1: 'j1', 
        numerator_field2: 'j2',
        denominator_field1: 'f1', 
        denominator_field2: 'f2'
    })
    
    cmp_df['prev_rate'] = cmp_df['j1']/cmp_df['f1']
    cmp_df['curr_rate'] = cmp_df['j2']/cmp_df['f2']
    
    y1 = cmp_df['j1'].sum()
    a1 = cmp_df['f1'].sum()
    y2 = cmp_df['j2'].sum()
    a2 = cmp_df['f2'].sum()

    print('previous success rate = %.2f' % (100*y1/a1))
    print('current success rate = %.2f' % (100*y2/a2))
    print('total success rate change = %.2f' % (100*(y2/a2 - y1/a1)))
    
    cmp_df['df'] = (a1*cmp_df.f2 - a2*cmp_df.f1)/(a2 - cmp_df.f2)
    cmp_df['total_effect'] = (y1 - cmp_df.j1 + (cmp_df.f1 + cmp_df.df)*cmp_df.curr_rate)/(a1 + cmp_df.df) - y1/a1
    cmp_df['mix_change_effect'] = (y1 + cmp_df.df*cmp_df.prev_rate)/(a1 + cmp_df.df) - y1/a1
    cmp_df['conversion_change_effect'] = (cmp_df.f1*cmp_df.j2 - cmp_df.f2*cmp_df.j1)/(a1 * cmp_df.f2)
    
    for col in ['total_effect', 'mix_change_effect', 'conversion_change_effect', 'curr_rate', 'prev_rate']:
        cmp_df[col] = 100*cmp_df[col]
        
    cmp_df['success_rate_diff'] = cmp_df.curr_rate - cmp_df.prev_rate
    cmp_df['prev_dim_share'] = 100*cmp_df.f1/a1
    cmp_df['curr_dim_share'] = 100*cmp_df.f2/a2
    for p in ['prev_dim_share', 'curr_dim_share', 'prev_rate', 'curr_rate', 'success_rate_diff',
                     'total_effect', 'mix_change_effect', 'conversion_change_effect']:
        cmp_df[p] = cmp_df[p].map(lambda x: round(x, 2))
    cmp_df['total_effect_share'] = 100*cmp_df.total_effect/(100*(y2/a2 - y1/a1))
    cmp_df['impact_coef'] = cmp_df.total_effect_share/cmp_df.prev_dim_share

    # creating visualisations
    create_share_vs_impact_chart(cmp_df.reset_index(), dimension, 'prev_dim_share', 'total_effect_share')
    cmp_df = cmp_df[['f1', 'f2', 'prev_dim_share', 'curr_dim_share', 'prev_rate', 'curr_rate', 'success_rate_diff',
                     'total_effect', 'mix_change_effect', 'conversion_change_effect', 'total_effect_share']]

    # return cmp_df[['total_effect']].rename(columns = {'total_effect': 'effect'})
    plot_conversion_waterfall(
        100*y1/a1, 100*y2/a2, cmp_df[['total_effect']].rename(columns = {'total_effect': 'effect'})
    )

    # putting together effects split by change of mix and conversion change
    tmp = []
    for rec in cmp_df.reset_index().to_dict('records'): 
        tmp.append(
            {
                'segment': rec[dimension] + ' - change of mix',
                'effect': rec['mix_change_effect']
            }
        )
        tmp.append(
            {
                'segment': rec[dimension] + ' - conversion change',
                'effect': rec['conversion_change_effect']
            }
        )
    effects_det_df = pd.DataFrame(tmp)
    effects_det_df['effect_abs'] = effects_det_df.effect.map(lambda x: abs(x))
    effects_det_df = effects_det_df.sort_values('effect_abs', ascending = False) 
    top_effects_det_df = effects_det_df.head(5).drop('effect_abs', axis = 1)
    plot_conversion_waterfall(
        100*y1/a1, 100*y2/a2, top_effects_det_df.set_index('segment')
    )
    return cmp_df

