import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import folium
import pycountry
import requests
import geopandas as gpd
import copy
from branca.element import Element
from folium.features import GeoJsonTooltip
from utils_processing import error_matrix_2x2, add_country_codes, format_days_to_ymwd

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
        marker=dict(color=color_bar, line=dict(width=0)),  # solid color, no edge
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
        yaxis=dict(title='Book count', showgrid=False, zeroline=False),
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

def create_travel_map(df_, var='total_days', code_convention='code3', bins=None, colors='Set1', save_path=None):

    # Add ISO country codes
    df_0 = df_.copy() # for counts
    df_ = add_country_codes(df_)
    df_ = df_.dropna(subset=[code_convention])

    # Define bins automatically if not given
    if bins is None:
        bins = np.linspace(df_[var].min(), df_[var].max(), 5)  # 4 intervals → 4 legend categories

    # Define fixed 4 colors (Set1)
    set1_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    legend_labels = ["days", "weeks", "months", "years"]

    # Load GeoJSON for world countries
    url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/world-countries.json'
    world_geo = requests.get(url).json()
    for f in world_geo['features']:
        f['properties']['id'] = f['id']

    # Duplicate features to handle wrapping across ±180° longitude
    def duplicate_features_with_wrap(features, offsets=[-360, 0, 360]):
        duplicated = []
        for offset in offsets:
            for feature in features:
                new_feature = copy.deepcopy(feature)
                geom = new_feature['geometry']
                if geom['type'] == 'Polygon':
                    geom['coordinates'] = [
                        [[lon + offset, lat] for lon, lat in ring]
                        for ring in geom['coordinates']
                    ]
                elif geom['type'] == 'MultiPolygon':
                    geom['coordinates'] = [
                        [[[lon + offset, lat] for lon, lat in ring]
                        for ring in polygon]
                        for polygon in geom['coordinates']
                    ]
                duplicated.append(new_feature)
        return duplicated
    world_geo['features'] = duplicate_features_with_wrap(world_geo['features'])

    # Create folium base map
    m = folium.Map(location=[0, 0], zoom_start=2, min_zoom=2, max_zoom=6, max_bounds=True)

    # Add choropleth layer
    folium.Choropleth(
        geo_data=world_geo,
        data=df_,
        columns=[code_convention, var],
        key_on='feature.properties.id',
        bins=bins,
        fill_color=colors,
        nan_fill_color='lightgray',
        fill_opacity=0.9,
        line_opacity=0.1,
        legend_name=None
    ).add_to(m)

    # Merge data for tooltips
    gdf = gpd.GeoDataFrame.from_features(world_geo['features']).set_crs("EPSG:4326")
    bounds = gdf.total_bounds
    gdf = gdf.merge(df_[[code_convention, var, 'country']], left_on='id', right_on=code_convention, how='left')

    gdf['tooltip_text'] = gdf.apply(
        lambda r: f"{r['country']}: {format_days_to_ymwd(r[var])}" if pd.notnull(r[var]) else f"{r['country']}: No data",
        axis=1
    )

    folium.GeoJson(
        gdf,
        name="Hover Info",
        style_function=lambda f: {'fillOpacity': 0, 'color': 'transparent', 'weight': 0},
        tooltip=folium.GeoJsonTooltip(fields=['tooltip_text'], labels=False, sticky=True)
    ).add_to(m)

    # Fit to world bounds
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    # Hide default folium legend
    hide_legend = Element("""
    <script>
    function hideLegend(){
        var legend = document.querySelector('.leaflet-control.legend');
        if(legend){ legend.style.display = 'none'; }
        else { setTimeout(hideLegend, 500); }
    }
    hideLegend();
    </script>
    """)
    m.get_root().html.add_child(hide_legend)

    # ---- Determine counts directly from bins and df_ ----
    counts = []
    for i in range(len(bins) - 1):
        lower, upper = bins[i], bins[i + 1]
        # count values in [lower, upper) range
        count = df_0[(df_0[var] >= lower) & (df_0[var] < upper)].shape[0] if i < len(bins) - 2 else df_0[(df_0[var] >= lower) & (df_0[var] <= upper)].shape[0]
        counts.append(count)

    # If user provides more or fewer than 4 bins, truncate or pad labels
    n_bins = len(bins) - 1
    legend_labels_used = legend_labels[:n_bins] + ["extra"] * max(0, n_bins - len(legend_labels))

    # ---- Build custom legend ----
    legend_entries = ""
    for color, label, count in zip(set1_colors[:n_bins], legend_labels_used, counts):
        legend_entries += f"""
        <i style="background:{color}; width:16px; height:16px; float:left; margin-right:4px;"></i>
        {label} ({count})<br>
        """

    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 15px; left: 20px; width: 90px; height:auto;
        background-color: lightgray;
        border:0px solid grey; z-index:9999; font-size:12px;
        padding: 5px;
    ">
        {legend_entries}
    </div>
    """
    m.get_root().html.add_child(Element(legend_html))

    # Remove mobile tap artifacts
    remove_mobile_artifacts = Element("""
    <style>
        .leaflet-interactive:focus { outline: none !important; stroke: none !important; }
        .leaflet-container * {
            -webkit-tap-highlight-color: rgba(0,0,0,0);
            -webkit-touch-callout: none;
            -webkit-user-select: none;
            user-select: none;
        }
    </style>
    """)
    m.get_root().html.add_child(remove_mobile_artifacts)

    # Save if path provided
    if save_path:
        m.save(save_path)

    return m

def create_book_map(df_, var='total_days', code_convention='code3', bins=None, colors='Set1', save_path=None):
    # Add ISO country codes
    df_ = add_country_codes(df_)

    # Drop rows with missing codes
    df_ = df_.dropna(subset=[code_convention])

    if bins is None:
        bins = np.linspace(df_[var].min(), df_[var].max(), 4)

    # 4 colors from Set1 (fixed colors as in original)
    set1_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

    # Load GeoJSON of world countries
    url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/world-countries.json'
    world_geo = requests.get(url).json()

    # Move feature.id to properties.id (needed for matching)
    for f in world_geo['features']:
        f['properties']['id'] = f['id']

    # Create GeoDataFrame and filter
    gdf = gpd.GeoDataFrame.from_features(world_geo['features']).set_crs("EPSG:4326")
    gdf = gdf[gdf['id'].isin(df_[code_convention])]
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    
    # Create base map
    m = folium.Map(
        location=[0, 0],
        zoom_start=2,
        min_zoom=2,
        max_zoom=6,
        max_bounds=True
    )
    import copy
    def duplicate_features_with_wrap(features, offsets=[-360, 0, 360]):
        duplicated = []
        for offset in offsets:
            for feature in features:
                new_feature = copy.deepcopy(feature)
                geometry = new_feature['geometry']
                if geometry['type'] == 'Polygon':
                    geometry['coordinates'] = [
                        [[lon + offset, lat] for lon, lat in ring]
                        for ring in geometry['coordinates']
                    ]
                elif geometry['type'] == 'MultiPolygon':
                    geometry['coordinates'] = [
                        [[[lon + offset, lat] for lon, lat in ring]
                        for ring in polygon]
                        for polygon in geometry['coordinates']
                    ]
                duplicated.append(new_feature)
        return duplicated
    world_geo['features'] = duplicate_features_with_wrap(world_geo['features'])

    # Add choropleth layer
    folium.Choropleth(
        geo_data=world_geo,
        data=df_,
        columns=[code_convention, var],
        key_on='feature.properties.id',
        bins=bins,
        fill_color=colors,
        nan_fill_color='lightgray',
        fill_opacity=0.9,
        line_opacity=0.1,
        legend_name=None
    ).add_to(m)

    # Merge data into gdf for tooltip
    gdf = gdf.merge(df_[[code_convention, var, 'country']], left_on='id', right_on=code_convention, how='left')

    # Add formatted tooltip text using country name instead of code
    gdf['tooltip_text'] = gdf.apply(
        lambda row: f"{row['country']}: {row[var]}" if pd.notnull(row[var]) else f"{row['country']}: No data",
        axis=1
    )

    # Add GeoJson with tooltip showing formatted time string
    folium.GeoJson(
        gdf,
        name="Hover Info",
        style_function=lambda feature: {
            'fillColor': 'transparent',
            'color': 'transparent',
            'weight': 0,
            'fillOpacity': 0,
        },
        highlight_function=lambda x: {'weight': 0, 'color': 'transparent'},
        tooltip=folium.GeoJsonTooltip(
            fields=['tooltip_text'],
            labels=False,
            sticky=True
        )
    ).add_to(m)

    # Fit to bounds
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    # Add JS to hide default legend
    hide_legend = Element("""
    <script>
    function hideLegend(){
        var legend = document.querySelector('.leaflet-control.legend');
        if(legend){
            legend.style.display = 'none';
        } else {
            setTimeout(hideLegend, 500);
        }
    }
    hideLegend();
    </script>
    """)
    m.get_root().html.add_child(hide_legend)

    # Manually define legend labels
    legend_html = f"""
    <div style="
        position: fixed; 
        bottom: 15px; left: 20px; width: 72px; height: 80px; 
        background-color: lightgray; 
        border:0px solid grey; z-index:9999; font-size:12px;
        padding: 5px;
    ">
        <i style="background:{set1_colors[0]}; width: 16px; height: 16px; float: left; margin-right: 4px;"></i> >0 <br>
        <i style="background:{set1_colors[1]}; width: 16px; height: 16px; float: left; margin-right: 4px;"></i> >1 <br>
        <i style="background:{set1_colors[2]}; width: 16px; height: 16px; float: left; margin-right: 4px;"></i> >10 <br>
        <i style="background:{set1_colors[3]}; width: 16px; height: 16px; float: left; margin-right: 4px;"></i> >30
    </div>
    """
    m.get_root().html.add_child(Element(legend_html))

    remove_mobile_artifacts = Element("""
    <style>
        .leaflet-interactive:focus {
            outline: none !important;
            stroke: none !important;
        }

        /* Remove tap highlight on mobile (for iOS/Android) */
        .leaflet-container * {
            -webkit-tap-highlight-color: rgba(0,0,0,0);
            -webkit-touch-callout: none;
            -webkit-user-select: none;
            user-select: none;
        }
    </style>
    """)
    m.get_root().html.add_child(remove_mobile_artifacts)

    # Save map if path is provided
    if save_path:
        m.save(save_path)

    return m