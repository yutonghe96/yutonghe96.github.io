import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from utils_processing import add_country_codes, format_days_to_ymwd

def plot_bar(df, cumulative=False, rotation=0, ystep=1, title=None):
    x_val = df.columns[0]
    y_val = df.columns[1]

    fig, ax1 = plt.subplots(figsize=(12, 4), dpi=200)

    if title:
        ax1.set_title(title)

    ax1.bar(df[x_val], df[y_val], color='royalblue', edgecolor='black', alpha=0.6)
    ax1.set_xlim(df[x_val].min() - 0.5, df[x_val].max() + 0.5)
    ax1.set_xticks(df[x_val])
    ax1.set_xticklabels(df[x_val], rotation=rotation)
    ax1.set_yticks(np.arange(0, df[y_val].max() * 1.05, ystep))
    ax1.set_ylabel('New countries visited', color='royalblue')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.tick_params(axis='y', labelcolor='royalblue')
    if cumulative:
        ax2 = ax1.twinx()
        cumulative = df[y_val].cumsum()
        ax2.plot(df[x_val], cumulative, color='magenta', marker='o', linewidth=2)
        ax2.set_ylabel('Cumulative countries visited', color='magenta')
        ax2.tick_params(axis='y', labelcolor='magenta')

    fig.tight_layout()
    plt.show()

def plot_countries_been_over_years(df: pd.DataFrame) -> None:
    # Melt to long format
    year_cols = [col for col in df.columns if col != 'country' and str(col).isdigit()]
    df_long = df.melt(id_vars='country', value_vars=year_cols, var_name='year', value_name='count')
    df_visits = df_long[df_long['count'] > 0].copy()
    df_visits['year'] = df_visits['year'].astype(int)

    # Identify first visit year
    first_visit = df_visits.groupby('country')['year'].min().reset_index().rename(columns={'year': 'first_year'})
    df_visits = df_visits.merge(first_visit, on='country')
    df_visits['visit_type'] = df_visits.apply(
        lambda row: 'new' if row['year'] == row['first_year'] else 'repeat',
        axis=1
    )

    # Function to create numbered country list
    def make_numbered_list(country_series):
        countries = sorted(country_series.unique())
        return '<br>'.join(f"{i + 1}. {c}" for i, c in enumerate(countries))

    # Group for new visits
    new_visits = df_visits[df_visits['visit_type'] == 'new'].groupby('year').agg(
        new_count=('country', 'nunique'),
        new_countries=('country', make_numbered_list)
    ).reset_index()

    # Group for repeat visits
    repeat_visits = df_visits[df_visits['visit_type'] == 'repeat'].groupby('year').agg(
        repeat_count=('country', 'nunique'),
        repeat_countries=('country', make_numbered_list)
    ).reset_index()

    # Merge summaries
    summary = pd.merge(new_visits, repeat_visits, on='year', how='outer').fillna({'new_count': 0, 'repeat_count': 0})
    summary = summary.sort_values('year')
    summary['cumulative'] = summary['new_count'].cumsum()

    # Remove years with only a repeat visit to one country
    summary = summary[~((summary['new_count'] == 0) & (summary['repeat_count'] == 1))].copy()

    color_bar_new = 'royalblue'
    color_bar_repeat = 'forestgreen'
    color_line = 'darkorange'

    fig = go.Figure()

    # New visit bar
    fig.add_trace(go.Bar(
        x=summary['year'].astype(str),
        y=summary['new_count'],
        name='First visits',
        marker_color=color_bar_new,
        offsetgroup=0,
        customdata=summary['new_countries'].values,
        hovertemplate='%{customdata}<extra></extra>'
    ))

    # Repeat visit bar
    fig.add_trace(go.Bar(
        x=summary['year'].astype(str),
        y=summary['repeat_count'],
        name='Repeat visits',
        marker_color=color_bar_repeat,
        offsetgroup=1,
        customdata=summary['repeat_countries'].values,
        hovertemplate='%{customdata}<extra></extra>'
    ))

    # Cumulative line
    fig.add_trace(go.Scatter(
        x=summary['year'].astype(str),
        y=summary['cumulative'],
        name='Cumulative',
        yaxis='y2',
        mode='lines+markers',
        line=dict(color=color_line, width=3),
        marker=dict(size=8, color=color_line),
        hovertemplate='Year: %{x}<br>Cumulative: %{y}<extra></extra>'
    ))

    # Layout using the plotly_dark template
    fig.update_layout(
        title=f'Countries Been ({int(summary["year"].min())}–{int(summary["year"].max())})',
        xaxis=dict(title='Year'),
        yaxis=dict(
            title='Country count',
            showgrid=False,
            zeroline=False
        ),
        yaxis2=dict(
            title=dict(
                text='Cumulative country count',
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

    fig.show()

def plot_book_count(df):
    # Prepare data
    count_per_year = df.groupby('dates_read').size().reset_index(name='count')
    count_per_year['cumulative'] = count_per_year['count'].cumsum()

    color_bar = 'royalblue'
    color_line = 'darkorange'

    fig = go.Figure()

    # Yearly count bars
    fig.add_trace(go.Bar(
        x=count_per_year['dates_read'].astype(str),
        y=count_per_year['count'],
        name='Book count',
        marker=dict(color=color_bar, line=dict(width=0)),
        customdata=count_per_year['count'],
        hovertemplate='Year: %{x}<br>Books: %{y}<extra></extra>'
    ))

    # Cumulative line
    fig.add_trace(go.Scatter(
        x=count_per_year['dates_read'].astype(str),
        y=count_per_year['cumulative'],
        name='Cumulative book count',
        yaxis='y2',
        mode='lines+markers',
        line=dict(color=color_line, width=3),
        marker=dict(size=8, color=color_line),
        hovertemplate='Year: %{x}<br>Cumulative: %{y}<extra></extra>'
    ))

    # Layout
    fig.update_layout(
        title=f'Books Read ({int(count_per_year["dates_read"].min())}–{int(count_per_year["dates_read"].max())})',
        xaxis=dict(title='Year', showgrid=False, zeroline=False),
        yaxis=dict(
            title=dict(text='Book count', font=dict(color=color_bar)),
            tickfont=dict(color=color_bar),
            showgrid=False,
            zeroline=False
        ),
        yaxis2=dict(
            title=dict(text='Cumulative book count', font=dict(color=color_line)),
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

    fig.show()

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