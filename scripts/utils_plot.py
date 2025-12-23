import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import plotly.io as pio
from utils_processing import add_country_codes, format_days_to_ymwd

def plot_bar_time_series(df, select_type='travel', time_period='year', title='Countries Been', save_path=None):

    color_bar = 'royalblue'
    color_line = 'darkorange'
    
    fig = go.Figure()

    if select_type == 'travel':
        # Melt to long format
        year_cols = [col for col in df.columns if col != 'country' and str(col).isdigit()]
        df_long = df.melt(id_vars='country', value_vars=year_cols, var_name=time_period, value_name='count')
        df_visits = df_long[df_long['count'] > 0].copy()
        df_visits[time_period] = df_visits[time_period].astype(int)

        # Identify first visit year
        first_visit = df_visits.groupby('country')[time_period].min().reset_index().rename(columns={time_period: 'first_year'})
        df_visits = df_visits.merge(first_visit, on='country')
        df_visits['visit_type'] = df_visits.apply(lambda row: 'new' if row[time_period] == row['first_year'] else 'repeat', axis=1)

        # Numbered country list
        def make_numbered_list(country_series):
            countries = sorted(country_series.unique())
            return '<br>'.join(f"{i + 1}. {c}" for i, c in enumerate(countries))

        # Aggregate data
        new_visits = df_visits[df_visits['visit_type'] == 'new'].groupby(time_period).agg(
            new_count=('country', 'nunique'),
            new_countries=('country', make_numbered_list)
        ).reset_index()

        repeat_visits = df_visits[df_visits['visit_type'] == 'repeat'].groupby(time_period).agg(
            repeat_count=('country', 'nunique'),
            repeat_countries=('country', make_numbered_list)
        ).reset_index()

        summary = pd.merge(new_visits, repeat_visits, on=time_period, how='outer').fillna({'new_count': 0, 'repeat_count': 0})
        summary = summary.sort_values(time_period)
        summary['cumulative'] = summary['new_count'].cumsum()
        summary = summary[~((summary['new_count'] == 0) & (summary['repeat_count'] == 1))].copy()

        # Bars
        fig.add_trace(go.Bar(
            x=summary[time_period].astype(str),
            y=summary['new_count'],
            name='First visits',
            marker_color=color_bar,
            offsetgroup=0,
            customdata=summary['new_countries'].values,
            hovertemplate='%{customdata}<extra></extra>'
        ))

        fig.add_trace(go.Bar(
            x=summary[time_period].astype(str),
            y=summary['repeat_count'],
            name='Repeat visits',
            marker_color='forestgreen',
            offsetgroup=1,
            customdata=summary['repeat_countries'].values,
            hovertemplate='%{customdata}<extra></extra>'
        ))

        # Cumulative line
        fig.add_trace(go.Scatter(
            x=summary[time_period].astype(str),
            y=summary['cumulative'],
            name='Cumulative count',
            yaxis='y2',
            mode='lines+markers',
            line=dict(color=color_line, width=3),
            marker=dict(size=8, color=color_line),
            hovertemplate='Year: %{x}<br>Cumulative: %{y}<extra></extra>'
        ))

        title = f'{title} ({int(summary["year"].min())}–{int(summary["year"].max())})'

    elif select_type == 'book':
        count_per_year = df.groupby(time_period).size().reset_index(name='count')
        count_per_year['cumulative'] = count_per_year['count'].cumsum()

        fig.add_trace(go.Bar(
            x=count_per_year[time_period].astype(str),
            y=count_per_year['count'],
            name='Count',
            marker=dict(color=color_bar, line=dict(width=0)),
            customdata=count_per_year['count'],
            hovertemplate='Year: %{x}<br>Count: %{y}<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=count_per_year[time_period].astype(str),
            y=count_per_year['cumulative'],
            name='Cumulative count',
            yaxis='y2',
            mode='lines+markers',
            line=dict(color=color_line, width=3),
            marker=dict(size=8, color=color_line),
            hovertemplate='Year: %{x}<br>Cumulative: %{y}<extra></extra>'
        ))

        title = f'{title} ({int(count_per_year[time_period].min())}–{int(count_per_year[time_period].max())})'

    else:
        raise ValueError("select_type must be 'travel' or 'book'")

    # Layout
    fig.update_layout(
        title=title,
        xaxis=dict(title='Year', showgrid=False, zeroline=False),
        yaxis=dict(
            title='Count' if select_type == 'book' else 'Country count',
            showgrid=False,
            zeroline=False
        ),
        yaxis2=dict(
            title=dict(
                text='Cumulative count' if select_type == 'book' else 'Cumulative country count',
                font=dict(color=color_line)
            ),
            overlaying='y',
            side='right',
            showgrid=False,
            zeroline=False,
            tickfont=dict(color=color_line)
        ),
        barmode='group',
        legend=dict(x=0.03, y=0.98),
        bargap=0.3,
        template='plotly_dark'
    )

    config = {
        "displayModeBar": True,        # show mode bar
        "displaylogo": False,          # remove Plotly logo
        "modeBarButtonsToRemove": [    # buttons to remove
        "toImage",                     # download as PNG
        #"zoom2d",                      # optional: keep zoom if you want
        "pan2d",                       # optional: keep pan
        "lasso2d", "select2d",
        "autoScale2d", 
        "hoverCompareCartesian",
        "hoverClosestCartesian",
        "toggleSpikelines",
        "resetScale2d",                # remove default reset if you will keep your own
        ],
        "modeBarButtonsToAdd": [       # optional: add back only desired buttons
        "zoomIn2d", "zoomOut2d", "resetScale2d"
        ]
    }

    if save_path:
        fig.write_html(save_path, include_plotlyjs="cdn", config=config)

    return fig.show(config=config)

def create_map(
    df_,
    var="total_days",
    code_convention="code3",
    bins=None,
    color=None,
    projection_type="orthographic",
    tooltip_mode='calendar',  # 'calendar' or 'raw'
    save_path=None
):

    # ---- Ensure ISO codes ----
    df = add_country_codes(df_.copy())
    df = df.dropna(subset=[code_convention, var])

    # ---- Binning ----
    if bins is None:
        bins = np.linspace(df[var].min(), df[var].max(), 5)

    n_bins = len(bins) - 1
    df["bin"] = pd.cut(
        df[var],
        bins=bins,
        include_lowest=True,
        labels=False
    ).astype(int)

    # ---- Colors ----
    if color is None:
        # Default discrete Set1
        colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"][:n_bins]
    elif isinstance(color, list):
        # Use provided discrete colors
        colors = color[:n_bins]
    else:
        # Single color: create a log-scaled gradient
        # Convert base color to RGB
        base_rgb = np.array(mcolors.to_rgb(color))
        # Log scale based on bin midpoints
        bin_mids = np.array([bins[i]+(bins[i+1]-bins[i])/2 for i in range(n_bins)])
        log_norm = (np.log(bin_mids) - np.log(bin_mids.min())) / (np.log(bin_mids.max()) - np.log(bin_mids.min()))
        # Darker = larger values
        colors = [mcolors.to_hex(base_rgb * (0.3 + 0.7*(1-log_val))) for log_val in log_norm]

    # ---- Discrete colorscale ----
    colorscale = []
    for i, c in enumerate(colors):
        colorscale.append([i / n_bins, c])
        colorscale.append([(i + 1) / n_bins, c])

    # ---- Tooltip ----
    def _tooltip(row):
        country_name = row.get('country', row[code_convention])
        value = row[var]
        if tooltip_mode == 'raw':
            return f"{country_name}: {value}"
        else:  # 'calendar' mode
            return f"{country_name}: {format_days_to_ymwd(value)}"

    hover_text = df.apply(_tooltip, axis=1)

    # ---- Choropleth ----
    fig = go.Figure(
        go.Choropleth(
            locations=df[code_convention],
            z=df["bin"],
            locationmode="ISO-3",
            text=hover_text,
            hoverinfo="text",
            colorscale=colorscale,
            zmin=0,
            zmax=n_bins,
            marker_line_width=0,
            showscale=False
        )
    )

    fig.update_traces(
        hoverlabel=dict(
            bgcolor='black',
            )
        )

    # ---- Geo config ----
    fig.update_geos(
        projection_type=projection_type,
        #resolution=50,
        showland=True,
        landcolor="white",
        showocean=True,
        oceancolor="skyblue",
        showcountries=False,
        showcoastlines=False,
        showframe=False,
        bgcolor="black",
    )

    # ---- Layout ----
    fig.update_layout(
        paper_bgcolor="black",
        plot_bgcolor="black",
        margin=dict(l=0, r=0, t=0, b=0),
        height=600,
        dragmode="pan",
        hovermode="closest",
    )

    config = {
        "displayModeBar": True,        # show mode bar
        "displaylogo": False,          # remove Plotly logo
        "modeBarButtonsToRemove": [    # buttons to remove
        "toImage",                     # download as PNG
        "zoom2d",                      # optional: keep zoom if you want
        "pan2d",                       # optional: keep pan
        "lasso2d", "select2d",
        "autoScale2d", 
        "hoverCompareCartesian",
        "hoverClosestCartesian",
        "toggleSpikelines",
        "resetScale2d",                # remove default reset if you will keep your own
        ],
        "modeBarButtonsToAdd": [       # optional: add back only desired buttons
        "zoomIn2d", "zoomOut2d", "resetScale2d"
        ]
    }

    if save_path:
        fig.write_html(save_path, include_plotlyjs="cdn", config=config)

    return fig.show(config=config)

def create_tree(
        df,
        feat='genre',
        var='count',
        flag='flag',
        threshold=0,
        threshold_global=False,
        color_dict={'Movie': 'royalblue', 'TV': 'gold', 'Other': 'white'},
        height=560,
        save_path=None
        ):
    
    df_orig = df.copy()
    df_copy = df.copy()
    
    # Step 0: aggregate low-value entries into "Other"
    if threshold_global:
        # Aggregate globally
        large_entries = df_copy[df_copy[var] >= threshold]
        small_entries_sum = df_copy[df_copy[var] < threshold][var].sum()
        if small_entries_sum > 0:
            other_row = {feat: 'Other', flag: 'Other', var: small_entries_sum}
            df_copy = pd.concat([large_entries, pd.DataFrame([other_row])], ignore_index=True)
    else:
        # Aggregate per flag
        aggregated_rows = []
        for f, group in df_copy.groupby(flag):
            large_entries = group[group[var] >= threshold]
            small_entries_sum = group[group[var] < threshold][var].sum()
            aggregated_rows.append(large_entries)
            if small_entries_sum > 0:
                other_row = {feat: 'Other', flag: f, var: small_entries_sum}
                aggregated_rows.append(pd.DataFrame([other_row]))
        df_copy = pd.concat(aggregated_rows, ignore_index=True)

    if 'Other' not in color_dict:
        color_dict['Other'] = 'gray'
    
    # Define min/max line length
    min_line_length = 10
    max_line_length = 30

    # Normalize count to 0-1
    count_min = df_copy[var].min()
    count_max = df_copy[var].max()
    norm_var = f'norm_{var}'
    df_copy[norm_var] = (df_copy[var] - count_min) / (count_max - count_min)

    # Compute dynamic line length
    df_copy['line_length'] = df_copy[norm_var] * (max_line_length - min_line_length) + min_line_length
    df_copy['line_length'] = df_copy['line_length'].astype(int)

    def wrap_feat(row):
        genre = row[feat]
        line_length = row['line_length']
        
        # If the whole genre fits, return as is
        if len(genre) <= line_length:
            return genre
        
        # Split into words
        words = genre.split(' ')
        lines = []
        current_line = ''
        
        for word in words:
            # If adding this word exceeds line_length
            if len(current_line) + len(word) + (1 if current_line else 0) > line_length:
                # Push current line to lines
                if current_line:
                    lines.append(current_line)
                current_line = word
            else:
                # Append word to current line
                current_line = f"{current_line} {word}" if current_line else word
        
        # Add last line
        if current_line:
            lines.append(current_line)
        
        return '<br>'.join(lines)
    
    feat_wrapped = f'{feat}_wrapped'
    df_copy[feat_wrapped] = df_copy.apply(wrap_feat, axis=1)
    df_copy['id'] = df_copy[flag] + ' | ' + df_copy[feat_wrapped]  # unique per flag+genre
    df_copy['parent'] = ""  # flat hierarchy
    df_copy['label'] = df_copy[feat_wrapped]  # show only genre text

    # Step 6: create treemap
    fig = go.Figure(go.Treemap(
        ids=df_copy['id'],
        labels=df_copy['label'],
        parents=df_copy['parent'],
        values=df_copy[var],
        marker=dict(
            colors=[color_dict.get(f, 'gray') for f in df_copy[flag]],
            line=dict(color='black', width=1)
        ),
        textinfo="label+value+percent parent",
        texttemplate="%{label}<br>(%{value:.0f}, %{percentParent:.0%})",
        textposition="middle center",
        hoverinfo='none',  # <-- disables hover
        branchvalues='total'
    ))

    fig.update_layout(
        margin=dict(l=1, r=1, t=1, b=1),
        template='plotly_dark',
        height=500,
        uniformtext=dict(minsize=8, mode='show')
    )

    # Step 7: compute total counts per flag, sort descending for title
    flag_counts_orig = df_orig.groupby(flag, as_index=False)[var].sum().sort_values(var, ascending=False)
    total_count_orig = flag_counts_orig[var].sum()

    flag_summaries = []
    for _, row in flag_counts_orig.iterrows():
        flag_name = row[flag]
        flag_count = row[var]
        flag_frac = flag_count / total_count_orig
        color = color_dict.get(flag_name, 'gray')
        flag_summaries.append(f"<span style='color:{color}'>{flag_name} ({int(flag_count)}, {flag_frac:.0%})</span>")

    title_text = ", ".join(flag_summaries)

    fig.update_layout(
        margin=dict(l=1, r=1, t=20, b=1),  # top margin increased for title
        template='plotly_dark',
        font=dict(color='white', size=11),
        height=height,
        uniformtext=dict(minsize=8, mode='show'),
        title=dict(
            text=title_text,
            x=0.5,
            xanchor='center',
            yanchor='top',
            y=0.99,  # stick to the very top inside the margin
            font=dict(size=16)
        )
    )

    config = {'displayModeBar': False, 'responsive': True}
    if save_path:    
        pio.write_html(fig, file=save_path, config=config, include_plotlyjs='cdn')

    return fig.show(config=config)