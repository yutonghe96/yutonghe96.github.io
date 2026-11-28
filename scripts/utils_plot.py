import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import plotly.io as pio
import re
from utils_processing import add_country_codes, format_days_to_ymwd

def plot_bar_time_series(
        df,
        select_type='travel',
        time_period='year',
        title='Countries Been',
        font_size_dict=None,
        trip_counts=None,
        save_path=None
        ):
    
    if font_size_dict is not None:
        BASE_FONT_SIZE = font_size_dict['base']
        TITLE_FONT_SIZE = font_size_dict['title']
        AXIS_TITLE_SIZE = font_size_dict['axis_title']
        LEGEND_FONT_SIZE = font_size_dict['legend']
        HOVER_FONT_SIZE = font_size_dict['hover']
    else:
        BASE_FONT_SIZE = 20
        TITLE_FONT_SIZE = 26
        AXIS_TITLE_SIZE = 22
        LEGEND_FONT_SIZE = 20
        HOVER_FONT_SIZE = 20

    color_bar = 'royalblue'
    color_line = 'darkorange'
    color_cum_countries = 'mediumpurple'   # travel: running total of first visits
    MAX_BAR_WIDTH = 0.4    # travel: width (in years) of the widest bar
    MAX_DEPTH = 0.35       # travel: depth offset (in years) of the most travel days
    PLOT_ASPECT = 2.0      # travel: assumed plot width / height, for the depth angle
    MAX_BALL_SIZE = 40     # travel: diameter (px) of the largest cumulative ball
    BALL_MIN_SIZE = 6      # travel: diameter (px) of the smallest cumulative ball
    PLOT_HEIGHT_PX = 550   # travel: assumed (lower bound) plot height, to sit balls above the blocks
    BALL_GAP_PX = 3        # travel: gap between a ball and the block below it
    BAR_GAP = 0.3          # gap between years' bar clusters (fraction of a year)
    BAR_PAIR_GAP = 0.02    # travel: gap between the two bars of a year (in years)
    
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

        # Numbered country list, with number of days in that country that year
        def make_numbered_list(group):
            group = group.sort_values('country')
            return '<br>'.join(
                f"{i + 1}. {row['country']} ({row['count']:.1f})"
                for i, (_, row) in enumerate(group.iterrows())
            )

        # Aggregate data
        new_group = df_visits[df_visits['visit_type'] == 'new'].groupby(time_period)
        new_visits = new_group.agg(new_count=('country', 'nunique'), new_days=('count', 'sum')).reset_index()
        new_visits['new_countries'] = new_group.apply(make_numbered_list).values

        repeat_group = df_visits[df_visits['visit_type'] == 'repeat'].groupby(time_period)
        repeat_visits = repeat_group.agg(repeat_count=('country', 'nunique'), repeat_days=('count', 'sum')).reset_index()
        repeat_visits['repeat_countries'] = repeat_group.apply(make_numbered_list).values

        summary = pd.merge(new_visits, repeat_visits, on=time_period, how='outer').fillna({
            'new_count': 0, 'repeat_count': 0, 'new_days': 0, 'repeat_days': 0,
            'new_countries': '', 'repeat_countries': ''})
        summary = summary.sort_values(time_period)
        summary['cumulative'] = summary['new_count'].cumsum()
        summary = summary[~((summary['new_count'] == 0) & (summary['repeat_count'] == 1))].copy()

        x_years = summary[time_period].astype(str)
        years = [int(y) for y in summary[time_period]]

        # Countries: first (blue) and repeat (green) visits side by side, drawn
        # as pseudo-3D blocks. Height = number of countries; width = number of
        # trips that visited them that year; depth (an oblique projection going
        # back into the page) = travel days spent in them that year, i.e. days
        # on trips, not days living there.
        countries_by = df_visits.groupby([time_period, 'visit_type'])['country'].apply(set).to_dict()
        trips = (trip_counts or {}).get('trips')

        def trips_visiting(year, visit_type):
            """(number of trips, travel days) in that year's first/repeat-visit
            countries: trips with a stay in one of them during the year, and the
            days of those stays falling in the year."""
            countries = countries_by.get((year, visit_type), set())
            y_start, y_end = pd.Timestamp(year=year, month=1, day=1), pd.Timestamp(year=year + 1, month=1, day=1)
            n_trips, days = 0, 0
            for t in trips:
                in_group = [(start, end) for c, start, end in t['stays']
                            if c in countries and start < y_end and end >= y_start]
                if in_group:
                    n_trips += 1
                    days += sum(max((min(end, y_end) - max(start, y_start)).days, 0) for start, end in in_group)
            return n_trips, days

        if trips is not None:
            new_trips, new_travel_days = zip(*[trips_visiting(y, 'new') for y in years])
            repeat_trips, repeat_travel_days = zip(*[trips_visiting(y, 'repeat') for y in years])
            new_w, new_off, rep_w, rep_off = _paired_bar_geometry(
                new_trips, repeat_trips, MAX_BAR_WIDTH, BAR_PAIR_GAP)
        else:
            new_trips = repeat_trips = new_travel_days = repeat_travel_days = None
            new_w = rep_w = [MAX_BAR_WIDTH] * len(years)
            new_off, rep_off = -MAX_BAR_WIDTH - BAR_PAIR_GAP / 2, BAR_PAIR_GAP / 2

        def country_hover(counts, lists, n_trips, travel_days):
            n_trips = n_trips if n_trips is not None else [None] * len(counts)
            travel_days = travel_days if travel_days is not None else [None] * len(counts)
            return [
                '<br>'.join(part for part in (
                    f'Depth: {_count(d, "travel day")}' if d is not None else '',
                    f'Width: {_count(t, "trip")}' if t is not None else '',
                    f'Height: {n:g} {"countries" if n > 1 else "country"}',
                    lst,
                ) if part)
                for n, lst, t, d in zip(counts, lists, n_trips, travel_days)
            ]

        new_hover = country_hover(summary['new_count'], summary['new_countries'], new_trips, new_travel_days)
        repeat_hover = country_hover(summary['repeat_count'], summary['repeat_countries'], repeat_trips, repeat_travel_days)

        # Numeric x positions (one per year, labelled with the year) so the
        # depth faces can be placed in the same coordinates as the bars
        x_pos = list(range(len(years)))
        fig.add_trace(go.Bar(
            x=x_pos,
            y=summary['new_count'],
            width=new_w, offset=new_off,
            name='First visits',
            meta=dict(series='new'),   # legend toggle also hides its depth faces
            marker_color=color_bar,
            marker_line_width=0,
            customdata=new_hover,
            hovertemplate='%{customdata}<extra></extra>'
        ))
        fig.add_trace(go.Bar(
            x=x_pos,
            y=summary['repeat_count'],
            width=rep_w, offset=rep_off,
            name='Repeat visits',
            meta=dict(series='repeat'),
            marker_color='forestgreen',
            marker_line_width=0,
            customdata=repeat_hover,
            hovertemplate='%{customdata}<extra></extra>'
        ))

        # Depth faces (top and right side of each block), drawn as shapes
        # below the bars so each bar's front face covers what's behind it.
        # Shapes can't be hovered, so each block also gets an invisible hover
        # area covering exactly the part of its faces left visible (not hidden
        # behind a neighbouring bar or face), so hovering always picks the
        # block that's actually seen there.
        n_years = len(years)
        new_depth = new_travel_days or [0] * n_years
        repeat_depth = repeat_travel_days or [0] * n_years
        max_days = max(list(new_depth) + list(repeat_depth) + [1])
        max_height = max(summary['new_count'].max(), summary['repeat_count'].max(), 1)
        # Turn an x depth (in years) into the matching rise (in counts) so the
        # depth recedes at about 45 degrees on a plot ~PLOT_ASPECT times wider
        # than tall
        y_span = max_height * 1.15
        rise_per_year = y_span * PLOT_ASPECT / n_years
        shapes = []
        faces = []    # (polygon, block, series) in drawing order, bottom first
        fronts = []   # (polygon, series) bar front faces, drawn above every shape
        blocks = []   # (hover text, colour, series) per block with depth
        for series, heights, widths, offsets, days, color, hovers in (
            ('new', summary['new_count'], new_w, new_off, new_depth, color_bar, new_hover),
            ('repeat', summary['repeat_count'], rep_w, rep_off, repeat_depth, 'forestgreen', repeat_hover),
        ):
            offsets = offsets if isinstance(offsets, list) else [offsets] * n_years
            top_color = _shade(color, 0.35)    # lit from above: lighter
            side_color = _shade(color, -0.45)  # in shadow: darker
            for i, (h, w, off, d, hover) in enumerate(zip(heights, widths, offsets, days, hovers)):
                if h <= 0 or w <= 0:
                    continue
                x0, x1 = i + off, i + off + w
                fronts.append(([(x0, 0), (x1, 0), (x1, h), (x0, h)], series))
                if d <= 0:
                    continue
                dx = MAX_DEPTH * d / max_days
                dy = dx * rise_per_year
                top = [(x0, h), (x1, h), (x1 + dx, h + dy), (x0 + dx, h + dy)]
                side = [(x1, 0), (x1 + dx, dy), (x1 + dx, h + dy), (x1, h)]
                for poly, fill in ((top, top_color), (side, side_color)):
                    shapes.append(dict(
                        type='path', xref='x', yref='y', layer='below',
                        name=f'depth-{series}',   # hidden with its series' bars
                        path='M ' + ' L '.join(f'{x},{y}' for x, y in poly) + ' Z',
                        fillcolor=fill, line=dict(width=0)))
                    faces.append((poly, len(blocks), series))
                blocks.append((hover, color, series))

        # Visible part of each face: remove everything drawn above it (later
        # faces, then all bar fronts). Hiding a series in the legend uncovers
        # parts of the other one, so the hover areas are built for each
        # combination of shown series ('both', or one alone); the post script
        # shows the set matching the legend.
        for scenario, shown in (('both', {'new', 'repeat'}), ('new', {'new'}), ('repeat', {'repeat'})):
            visible = [[] for _ in blocks]
            s_faces = [(poly, block) for poly, block, ser in faces if ser in shown]
            s_fronts = [poly for poly, ser in fronts if ser in shown]
            for k, (poly, block) in enumerate(s_faces):
                pieces = [poly]
                for above in [f for f, _ in s_faces[k + 1:]] + s_fronts:
                    pieces = [q for p in pieces for q in _subtract_convex(p, above)]
                    if not pieces:
                        break
                visible[block].extend(pieces)

            for (hover, color, _), pieces in zip(blocks, visible):
                if not pieces:
                    continue
                xs, ys = [], []
                for piece in pieces:
                    xs += [x for x, _ in piece] + [piece[0][0], None]
                    ys += [y for _, y in piece] + [piece[0][1], None]
                fig.add_trace(go.Scatter(
                    x=xs, y=ys,
                    mode='lines', fill='toself',
                    fillcolor='rgba(0,0,0,0)', line=dict(width=0),
                    hoveron='fills', hoverinfo='text', text=hover,
                    hoverlabel=dict(bgcolor=color),
                    showlegend=False,
                    visible=scenario == 'both',
                    meta=dict(depth_hover=scenario),
                ))
        top_y = max_height + MAX_DEPTH * rise_per_year
        fig.update_layout(shapes=shapes)

        # Cumulative countries as shaded balls sitting just above the top face
        # of each year's tallest block; ball area is proportional to the
        # running total (no axis needed). A radial gradient (light centre,
        # darker rim) makes each marker read as a sphere.
        cum = summary['cumulative'].tolist()
        cum_max = max(max(cum), 1)
        ball_px = [max(MAX_BALL_SIZE * np.sqrt(c / cum_max), BALL_MIN_SIZE) for c in cum]

        def block_top(heights, widths, offsets, depth_days, i):
            """(height, x of the top face's centre, y of its back edge) of a year's block."""
            off = offsets[i] if isinstance(offsets, list) else offsets
            dx = MAX_DEPTH * depth_days[i] / max_days
            dy = dx * rise_per_year
            return heights[i], i + off + widths[i] / 2 + dx / 2, heights[i] + dy

        new_h, rep_h = summary['new_count'].tolist(), summary['repeat_count'].tolist()
        # Where each ball rests when only one series is shown (the post script
        # moves the balls there when the other is hidden in the legend)
        rests_new = [block_top(new_h, new_w, new_off, new_depth, i) for i in range(n_years)]
        rests_rep = [block_top(rep_h, rep_w, rep_off, repeat_depth, i) for i in range(n_years)]
        rests = [
            max(block_top(new_h, new_w, new_off, new_depth, i),
                block_top(rep_h, rep_w, rep_off, repeat_depth, i))
            for i in range(n_years)
        ]
        # Ball size is in pixels: convert to counts with the y range, which in
        # turn has to fit the balls. Each ball's bottom sits BALL_GAP_PX above
        # the back edge of the block's top face, so it never cuts into the
        # block. PLOT_HEIGHT_PX errs on the short side: in a taller plot the
        # balls float a little higher rather than sinking in.
        # (a ball of d px above y fits under R if y + (gap + d) * R / H <= R)
        y_max = max([top_y * 1.03] + [
            y / (1 - (BALL_GAP_PX + d) / PLOT_HEIGHT_PX) * 1.02
            for (_, _, y), d in zip(rests, ball_px)
        ])
        units_per_px = y_max / PLOT_HEIGHT_PX
        fig.add_trace(go.Scatter(
            x=[x for _, x, _ in rests],
            y=[y + (BALL_GAP_PX + d / 2) * units_per_px for (_, _, y), d in zip(rests, ball_px)],
            name='Cumulative',
            mode='markers',
            cliponaxis=False,   # never crop a ball at the plot edge
            marker=dict(
                size=ball_px,
                opacity=1,
                color=_shade(color_cum_countries, -0.15),    # rim
                gradient=dict(type='radial', color=_shade(color_cum_countries, 0.75)),  # centre
                line=dict(width=0),
            ),
            customdata=list(zip(x_years, cum)),
            hovertemplate='Year: %{customdata[0]}<br>Cumulative: %{customdata[1]}<extra></extra>',
            # Read by _TRAVEL_BARS_SCRIPT to grow the balls with the zoom and
            # move them onto whichever blocks are shown
            meta=dict(ball_zoom=dict(
                size=[float(d) for d in ball_px],
                rest={key: [[float(x), float(y)] for _, x, y in r]
                      for key, r in (('both', rests), ('new', rests_new), ('repeat', rests_rep))},
                gap=BALL_GAP_PX,
            )),
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
            line=dict(color=color_line, width=4),
            marker=dict(size=10, color=color_line),
            hovertemplate='Year: %{x}<br>Cumulative: %{y}<extra></extra>'
        ))

        title = f'{title} ({int(count_per_year[time_period].min())}–{int(count_per_year[time_period].max())})'

    else:
        raise ValueError("select_type must be 'travel' or 'book'")

    # Layout
    def y_axis(text, color=None, **kw):
        return dict(
            title=dict(text=text, font=dict(size=AXIS_TITLE_SIZE, color=color)),
            tickfont=dict(size=BASE_FONT_SIZE, color=color),
            showgrid=False,
            zeroline=False,
            **kw
        )

    x_title = dict(text='Year', font=dict(size=AXIS_TITLE_SIZE))
    x_common = dict(tickfont=dict(size=BASE_FONT_SIZE), showgrid=False, zeroline=False)
    legend_common = dict(x=0.01, xanchor='left', yanchor='top', font=dict(size=LEGEND_FONT_SIZE))
    # Travel: one-row legend just above the plot, clear of the bars
    legend_above = dict(orientation='h', x=0, xanchor='left', yanchor='bottom',
                        bgcolor='rgba(0,0,0,0)', font=dict(size=LEGEND_FONT_SIZE))

    top_domain = [0, 0.92 if select_type == 'travel' else 1]   # room for the legend row above
    layout = dict(
        title=dict(text=title, font=dict(size=TITLE_FONT_SIZE)),
        font=dict(size=BASE_FONT_SIZE),
        xaxis=dict(
            title=x_title,
            **x_common
        ),
        yaxis=y_axis('Count', domain=top_domain),
        yaxis2=y_axis('Cumulative count', color_line if select_type == 'book' else color_cum_countries, overlaying='y', side='right'),
        legend=dict(y=top_domain[1], **legend_common),
        hoverlabel=dict(font=dict(size=HOVER_FONT_SIZE)),
        barmode='group',
        bargap=BAR_GAP,
        template='plotly_dark'
    )

    if select_type == 'travel':
        layout['legend'] = dict(y=top_domain[1] + 0.01, **legend_above)
        # Fixed range so the depth faces and balls fit (the axis doesn't
        # autorange to shapes reliably)
        layout['yaxis'].update(range=[0, y_max])
        layout['margin'] = dict(r=30)   # no right axis any more
        layout['xaxis'].update(tickvals=x_pos, ticktext=list(x_years),
                               range=[-0.6, len(x_pos) - 0.4])
        del layout['yaxis2']   # ball size shows the cumulative count

    fig.update_layout(**layout)

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
        fig.write_html(
            save_path,
            include_plotlyjs="cdn",
            config=config,
            # Travel has one y axis, shown or hidden by _TRAVEL_BARS_SCRIPT
            post_script=_TRAVEL_BARS_SCRIPT if select_type == 'travel' else _AXIS_TOGGLE_SCRIPT,
        )

    return fig.show(config=config)

def _paired_bar_geometry(left_values, right_values, max_width, gap):
    """Widths and offsets for two bars per x category whose widths encode
    ``left_values``/``right_values`` (one shared scale, widest = ``max_width``),
    placed to meet at the category centre with ``gap`` between them. Plotly's
    group mode can't lay out bars of varying width, hence explicit offsets.

    Returns ``(left_widths, left_offsets, right_widths, right_offsets)``.
    """
    biggest = max(list(left_values) + list(right_values) + [1e-9])
    left_w = [max_width * v / biggest for v in left_values]
    right_w = [max_width * v / biggest for v in right_values]
    return left_w, [-w - gap / 2 for w in left_w], right_w, gap / 2

def _shade(color, amount):
    """``color`` blended towards white (``amount`` > 0) or black (< 0) by ``|amount|``, as rgb()."""
    r, g, b = mcolors.to_rgb(color)
    target = 1.0 if amount > 0 else 0.0
    a = abs(amount)
    r, g, b = (c + (target - c) * a for c in (r, g, b))
    return f'rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})'

def _clip_half_plane(poly, a, b, keep_left):
    """Part of convex polygon ``poly`` left (or right) of the line a -> b."""
    def side(p):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    out = []
    for i, p in enumerate(poly):
        q = poly[(i + 1) % len(poly)]
        sp, sq = side(p), side(q)
        if not keep_left:
            sp, sq = -sp, -sq
        if sp >= 0:
            out.append(p)
        if (sp >= 0) != (sq >= 0):
            t = sp / (sp - sq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out

def _polygon_area(poly):
    return 0.5 * sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(poly, poly[1:] + poly[:1]))

def _subtract_convex(poly, cutter, eps=1e-9):
    """Convex polygon ``poly`` minus convex polygon ``cutter`` (both
    counter-clockwise), as a list of convex pieces."""
    pieces, rest = [], poly
    for i, a in enumerate(cutter):
        b = cutter[(i + 1) % len(cutter)]
        outside = _clip_half_plane(rest, a, b, keep_left=False)
        if len(outside) >= 3 and abs(_polygon_area(outside)) > eps:
            pieces.append(outside)
        rest = _clip_half_plane(rest, a, b, keep_left=True)
        if len(rest) < 3 or abs(_polygon_area(rest)) <= eps:
            return pieces
    return pieces   # whatever is left in ``rest`` lies inside the cutter

def _count(value, noun):
    """``value`` with ``noun``, plural only above 1 (e.g. '0.3 trip', '5 days')."""
    return f"{value:g} {noun}{'s' if value > 1 else ''}"

def _parse_coords(c):
    if c is None or pd.isna(c):
        return pd.Series([np.nan, np.nan])

    # (lat, lon) tuple or list
    if isinstance(c, (list, tuple)) and len(c) == 2:
        return pd.Series([float(c[0]), float(c[1])])

    # dict form
    if isinstance(c, dict):
        return pd.Series([c.get("lat"), c.get("lon")])

    # string formats
    if isinstance(c, str):
        parts = [p.strip() for p in c.split(",")]
        if len(parts) != 2:
            return pd.Series([np.nan, np.nan])

        def _parse_part(p):
            # extract numeric value
            val = float(re.findall(r"[-+]?\d*\.\d+|\d+", p)[0])
            # hemisphere
            if re.search(r"[SW]", p, re.IGNORECASE):
                val = -val
            return val

        try:
            lat = _parse_part(parts[0])
            lon = _parse_part(parts[1])
            return pd.Series([lat, lon])
        except Exception:
            return pd.Series([np.nan, np.nan])

    return pd.Series([np.nan, np.nan])

_PLANNED_COLOR = "#9467bd"


def _build_country_traces(df_, var, code_convention, bins, color, tooltip_mode, df_planned):
    NUM_COLS = df_.select_dtypes(include='number').columns.tolist()
    df_ = df_.groupby('country')[NUM_COLS].sum().reset_index()
    df = add_country_codes(df_.copy())
    df = df.dropna(subset=[code_convention, var])

    if bins is None:
        bins = np.linspace(df[var].min(), df[var].max(), 5)

    n_bins = len(bins) - 1
    df = df[df[var].notna()].copy()
    df["bin"] = pd.cut(df[var], bins=bins, include_lowest=True, labels=False)
    df["bin"] = df["bin"].astype("Int64")

    if color is None:
        colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"][:n_bins]
    elif isinstance(color, list):
        colors = color[:n_bins]
    else:
        base_rgb = np.array(mcolors.to_rgb(color))
        bin_mids = np.array([bins[i] + (bins[i+1] - bins[i]) / 2 for i in range(n_bins)])
        log_norm = ((np.log(bin_mids) - np.log(bin_mids.min())) / (np.log(bin_mids.max()) - np.log(bin_mids.min())))
        colors = [mcolors.to_hex(base_rgb * (0.3 + 0.7*(1 - log_val))) for log_val in log_norm]

    colorscale = []
    for i, c in enumerate(colors):
        colorscale.append([i / n_bins, c])
        colorscale.append([(i + 1) / n_bins, c])

    # ---- Aggregate planned data per country (if any) for tooltip lookup ----
    planned_map = {}
    dfp_full = None
    if df_planned is not None and not df_planned.empty:
        P_NUM = df_planned.select_dtypes(include='number').columns.tolist()
        dfp_full = df_planned.groupby('country')[P_NUM].sum().reset_index()
        dfp_full = add_country_codes(dfp_full)
        dfp_full = dfp_full.dropna(subset=[code_convention, var])
        planned_map = dict(zip(dfp_full[code_convention], dfp_full[var]))

    def _fmt(v):
        return v if tooltip_mode == 'raw' else format_days_to_ymwd(v)

    def _tooltip(row):
        country_name = row.get('country', row[code_convention])
        code = row[code_convention]
        been_val = row[var]
        planned_val = planned_map.get(code, 0) or 0
        if planned_val > 0:
            total = been_val + planned_val
            return (f"{country_name} (been): {_fmt(been_val)}<br>"
                    f"{country_name} (planned): {_fmt(planned_val)}<br>"
                    f"{country_name} (total): {_fmt(total)}")
        return f"{country_name}: {_fmt(been_val)}"

    hover_text = df.apply(_tooltip, axis=1)

    traces = [
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
            showscale=False,
        )
    ]

    if dfp_full is not None:
        been_codes = set(df[code_convention].dropna())
        dfp = dfp_full[~dfp_full[code_convention].isin(been_codes)]
        if not dfp.empty:
            hover_planned = dfp.apply(
                lambda r: f"{r['country']} (planned): {_fmt(r[var])}",
                axis=1
            )
            traces.append(
                go.Choropleth(
                    locations=dfp[code_convention],
                    z=[1] * len(dfp),
                    locationmode="ISO-3",
                    text=hover_planned,
                    hoverinfo="text",
                    colorscale=[[0, _PLANNED_COLOR], [1, _PLANNED_COLOR]],
                    zmin=0,
                    zmax=1,
                    marker_line_width=0,
                    showscale=False,
                )
            )
    return traces


def _build_city_traces(df_, var, color, tooltip_mode, df_planned):
    df = df_.copy()
    df = df.dropna(subset=["city", "coordinates", var])
    df[["lat", "lon"]] = df["coordinates"].apply(_parse_coords)
    df = df.dropna(subset=["lat", "lon"])
    df_b = df[['city', 'coordinates', 'lat', 'lon', var]].rename(columns={var: 'been'})

    if df_planned is not None and not df_planned.empty:
        dfp = df_planned.copy().dropna(subset=["city", "coordinates", var])
        dfp[["lat", "lon"]] = dfp["coordinates"].apply(_parse_coords)
        dfp = dfp.dropna(subset=["lat", "lon"])
        df_p = dfp[['city', 'coordinates', 'lat', 'lon', var]].rename(columns={var: 'planned'})
        merged = pd.merge(df_b, df_p, on=['city', 'coordinates', 'lat', 'lon'], how='outer')
    else:
        merged = df_b.copy()
        merged['planned'] = 0.0

    merged['been'] = merged['been'].fillna(0.0)
    merged['planned'] = merged['planned'].fillna(0.0)
    merged['total'] = merged['been'] + merged['planned']

    size_raw = np.log1p(merged['total'])
    sr_min, sr_max = float(size_raw.min()), float(size_raw.max())
    def _size(v):
        if v is None or pd.isna(v) or v <= 0:
            return 0.0
        r = np.log1p(v)
        if sr_max > sr_min:
            return float(4 + 30 * (r - sr_min) / (sr_max - sr_min))
        return 10.0
    merged['size_been'] = merged['been'].apply(_size)
    merged['size_total'] = merged['total'].apply(_size)

    solid = merged[merged['been'] > 0].copy()
    rings_both = merged[(merged['been'] > 0) & (merged['planned'] > 0)].copy()
    planned_only = merged[(merged['been'] == 0) & (merged['planned'] > 0)].copy()

    def _tt_solid(row):
        v = row['been']
        return f"{row['city']}: {v}" if tooltip_mode == 'raw' else f"{row['city']}: {format_days_to_ymwd(v)}"

    def _tt_ring(row):
        if tooltip_mode == 'raw':
            if row['been'] > 0:
                return (f"{row['city']} (been): {row['been']}<br>"
                        f"{row['city']} (planned): {row['planned']}<br>"
                        f"{row['city']} (total): {row['total']}")
            return f"{row['city']} (planned): {row['planned']}"
        if row['been'] > 0:
            return (f"{row['city']} (been): {format_days_to_ymwd(row['been'])}<br>"
                    f"{row['city']} (planned): {format_days_to_ymwd(row['planned'])}<br>"
                    f"{row['city']} (total): {format_days_to_ymwd(row['total'])}")
        return f"{row['city']} (planned): {format_days_to_ymwd(row['planned'])}"

    marker_color = color or "#e41a1c"
    traces = []
    if not planned_only.empty:
        traces.append(
            go.Scattergeo(
                lon=planned_only['lon'], lat=planned_only['lat'],
                text=planned_only.apply(_tt_ring, axis=1),
                hoverinfo="text",
                mode="markers",
                marker=dict(
                    size=planned_only['size_total'],
                    color=_PLANNED_COLOR,
                    line=dict(width=0),
                    opacity=0.75,
                ),
                name='planned',
            )
        )
    if not rings_both.empty:
        ring_width = (rings_both['size_total'] - rings_both['size_been']) / 2.0
        traces.append(
            go.Scattergeo(
                lon=rings_both['lon'], lat=rings_both['lat'],
                text=rings_both.apply(_tt_ring, axis=1),
                hoverinfo="text",
                mode="markers",
                marker=dict(
                    size=rings_both['size_total'],
                    color='rgba(0,0,0,0)',
                    line=dict(color=_PLANNED_COLOR, width=ring_width),
                    opacity=1.0,
                ),
                name='both',
            )
        )
    if not solid.empty:
        traces.append(
            go.Scattergeo(
                lon=solid['lon'], lat=solid['lat'],
                text=solid.apply(_tt_solid, axis=1),
                hoverinfo="text",
                mode="markers",
                marker=dict(
                    size=solid['size_been'],
                    color=marker_color,
                    opacity=0.75,
                    line=dict(width=0),
                ),
                name='been',
            )
        )
    return traces


_PAN_LOCK_SCRIPT = r"""
(function(){
  var gd = document.getElementById('{plot_id}');
  if (!gd) return;
  var resetting = false;
  function isFlat() {
    try {
      var t = gd.layout && gd.layout.geo && gd.layout.geo.projection && gd.layout.geo.projection.type;
      return t && t !== 'orthographic';
    } catch(e) { return false; }
  }
  function onRelayout(ed) {
    if (resetting || !isFlat() || !ed) return;
    var fixes = {};
    var needsFix = false;
    Object.keys(ed).forEach(function(k){
      if ((k === 'geo.center.lat' || k === 'geo.center.lon') && ed[k] !== 0) {
        fixes[k] = 0;
        needsFix = true;
      } else if ((k === 'geo.projection.rotation.lat' || k === 'geo.projection.rotation.lon') && ed[k] !== 0) {
        fixes[k] = 0;
        needsFix = true;
      }
    });
    if (needsFix && window.Plotly) {
      resetting = true;
      window.Plotly.relayout(gd, fixes).then(function(){ resetting = false; }).catch(function(){ resetting = false; });
    }
  }
  if (gd.on) { gd.on('plotly_relayout', onRelayout); }
})();
"""


_RESIZE_POST_SCRIPT = r"""
(function(){
  var gd = document.getElementById('{plot_id}');
  if (!gd) return;
  function forceResize() {
    if (window.Plotly && window.Plotly.Plots && window.Plotly.Plots.resize) {
      try { window.Plotly.Plots.resize(gd); } catch(e) {}
    }
  }
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(forceResize).catch(function(){});
  }
  setTimeout(forceResize, 300);
  setTimeout(forceResize, 1200);
})();
"""


_AXIS_TOGGLE_SCRIPT = r"""
(function(){
  var gd = document.getElementById('{plot_id}');
  if (!gd) return;
  function isOn(t){ return t.visible === true || t.visible === undefined; }
  // Hide each y axis whose traces are all toggled off in the legend
  function refresh(){
    if (!gd.data || !window.Plotly) return;
    var shown = {};
    gd.data.forEach(function(t){
      var ax = t.yaxis || 'y';
      if (!(ax in shown)) shown[ax] = false;
      if (isOn(t)) shown[ax] = true;
    });
    var update = {};
    Object.keys(shown).forEach(function(ax){
      var key = 'yaxis' + ax.slice(1);
      var cur = gd.layout && gd.layout[key] && gd.layout[key].visible !== false;
      if (cur !== shown[ax]) update[key + '.visible'] = shown[ax];
    });
    if (Object.keys(update).length) window.Plotly.relayout(gd, update);
  }
  if (gd.on) {
    gd.on('plotly_restyle', function(){ setTimeout(refresh, 0); });
    gd.on('plotly_legenddoubleclick', function(){ setTimeout(refresh, 50); });
  }
  setTimeout(refresh, 300);
})();
"""


_TRAVEL_BARS_SCRIPT = r"""
(function(){
  // Keeps the travel bar chart's extras in step with the legend and zoom:
  // - Legend: the depth faces (layout shapes named 'depth-new' /
  //   'depth-repeat') and the invisible hover areas aren't legend items, so
  //   hide them with their series' bars, and show the hover areas built for
  //   the series still shown. The balls move onto the shown blocks.
  // - Zoom: marker sizes are in pixels, so on their own the balls keep their
  //   size while the bars grow. Scale each ball by how much the axes are
  //   zoomed in (the smaller of the x and y factors, so a ball never outgrows
  //   its block), and re-seat it the same pixel gap above its block.
  // - Cumulative only (both bar series hidden): the count axis means nothing,
  //   so hide it and line the balls up across the middle of the plot, one
  //   per year.
  var gd = document.getElementById('{plot_id}');
  if (!gd || !gd.on) return;
  var ball = -1, base = null, x0 = null, y0 = null;
  function span(r){ return Math.abs(r[1] - r[0]); }
  function isOn(t){ return t.visible === true || t.visible === undefined; }
  function init(){
    if (!gd.data || !gd._fullLayout) return false;
    for (var i = 0; i < gd.data.length; i++) {
      var m = gd.data[i].meta;
      if (m && m.ball_zoom) { ball = i; base = m.ball_zoom; break; }
    }
    if (ball < 0) return false;
    x0 = span(gd._fullLayout.xaxis.range);
    y0 = span(gd._fullLayout.yaxis.range);
    return true;
  }
  function shownSeries(){
    var shown = {};
    gd.data.forEach(function(t){
      if (t.meta && t.meta.series) shown[t.meta.series] = isOn(t);
    });
    return shown;
  }
  function scenario(shown){
    return shown['new'] && shown['repeat'] ? 'both'
         : shown['new'] ? 'new' : shown['repeat'] ? 'repeat' : null;
  }
  var busy = false;
  function sync(){
    if (busy || !window.Plotly || (ball < 0 && !init())) return;
    var shown = shownSeries(), sc = scenario(shown);
    var calls = [];
    // Depth faces
    var shapeUpdate = {};
    (gd.layout.shapes || []).forEach(function(sh, i){
      var m = /^depth-(.+)$/.exec(sh.name || '');
      if (!m) return;
      var want = !!shown[m[1]];
      if ((sh.visible !== false) !== want) shapeUpdate['shapes[' + i + '].visible'] = want;
    });
    if (Object.keys(shapeUpdate).length) calls.push(function(){ return window.Plotly.relayout(gd, shapeUpdate); });
    // Count axis: only while some bars are shown
    var axisWanted = !!sc;
    if ((gd.layout.yaxis.visible !== false) !== axisWanted)
      calls.push(function(){ return window.Plotly.relayout(gd, {'yaxis.visible': axisWanted}); });
    // Hover areas
    var hoverIdx = [], hoverVis = [];
    gd.data.forEach(function(t, i){
      if (!t.meta || !t.meta.depth_hover) return;
      var want = t.meta.depth_hover === sc;
      if ((t.visible !== false) !== want) { hoverIdx.push(i); hoverVis.push(want); }
    });
    if (hoverIdx.length) calls.push(function(){ return window.Plotly.restyle(gd, {visible: hoverVis}, hoverIdx); });
    // Balls: size from the zoom, position on top of the shown blocks
    var fl = gd._fullLayout;
    var k = Math.max(Math.min(x0 / span(fl.xaxis.range), y0 / span(fl.yaxis.range)), 1);
    var perPx = span(fl.yaxis.range) / fl._size.h;   // y units per pixel
    var rest = base.rest[sc || 'both'];
    var sizes = base.size.map(function(d){ return d * k; });
    var mid = (fl.yaxis.range[0] + fl.yaxis.range[1]) / 2;
    var xs = rest.map(function(r, i){ return sc ? r[0] : i; });
    var ys = rest.map(function(r, i){
      return sc ? r[1] + (base.gap + sizes[i] / 2) * perPx : mid;
    });
    var cur = gd.data[ball];
    function same(a, b){
      if (!a || !b || a.length !== b.length) return false;
      for (var i = 0; i < a.length; i++) if (Math.abs(a[i] - b[i]) > 1e-9) return false;
      return true;
    }
    if (!(same(cur.marker.size, sizes) && same(cur.x, xs) && same(cur.y, ys)))
      calls.push(function(){ return window.Plotly.restyle(gd, {'marker.size': [sizes], x: [xs], y: [ys]}, [ball]); });
    if (!calls.length) return;
    busy = true;
    calls.reduce(function(p, f){ return p.then(f); }, Promise.resolve())
      .then(function(){ busy = false; }, function(){ busy = false; });
  }
  function later(){ setTimeout(sync, 0); }
  gd.on('plotly_relayout', later);
  gd.on('plotly_restyle', later);
  gd.on('plotly_legenddoubleclick', function(){ setTimeout(sync, 50); });
  setTimeout(sync, 300);
})();
"""


_RESET_HOOK_SCRIPT = r"""
(function(){
  var gd = document.getElementById('{plot_id}');
  if (!gd) return;
  function fullReset(){
    if (!window.Plotly) return;
    return window.Plotly.relayout(gd, {
      'geo.center.lon': 0,
      'geo.center.lat': 0,
      'geo.projection.rotation.lon': 0,
      'geo.projection.rotation.lat': 0,
      'geo.projection.rotation.roll': 0,
      'geo.projection.scale': 1,
      'geo.lonaxis.range': null,
      'geo.lataxis.range': null
    });
  }
  function attach(){
    var selectors = [
      '[data-attr="resetDefault"]',
      '[data-attr="resetGeo"]',
      '[data-attr="resetScale"]',
      '[data-attr="zoomDefault"]',
      '[data-title="Reset"]',
      '[data-title="Reset axes"]',
      '[data-title="Reset view"]',
      '[data-val="reset"]'
    ];
    var buttons = gd.querySelectorAll(selectors.join(','));
    buttons.forEach(function(btn){
      if (btn.__fullResetHooked) return;
      btn.__fullResetHooked = true;
      btn.addEventListener('click', function(e){
        e.preventDefault();
        e.stopImmediatePropagation();
        fullReset();
      }, true);
    });
  }
  setTimeout(attach, 300);
  setTimeout(attach, 1200);
  if (gd.on) {
    gd.on('plotly_afterplot', function(){ setTimeout(attach, 50); });
  }
})();
"""


def _dropdown_menu(buttons, x, y):
    return dict(
        type='dropdown',
        direction='down',
        buttons=buttons,
        x=x,
        y=y,
        xanchor='left',
        yanchor='top',
        pad=dict(l=8, r=8, t=6, b=6),
        showactive=True,
        active=0,
        bgcolor='rgba(0,0,0,0)',
        borderwidth=0,
        font=dict(family='Arial, sans-serif', color='royalblue', size=14),
    )


def create_map(
    df_,
    var="total_days",
    code_convention="code3",
    bins=None,
    color=None,
    projection_type="orthographic",
    tooltip_mode='calendar',
    level="country",
    df_planned=None,
    save_path=None
):

    # ---- Normalize projection_type and level ----
    if isinstance(projection_type, dict):
        proj_items = list(projection_type.items())
    else:
        proj_items = [(str(projection_type), projection_type)]
    initial_projection = proj_items[0][1]

    if isinstance(level, dict):
        level_items = list(level.items())
    else:
        level_items = [(str(level), level)]

    # ---- Build traces for each level; track index ranges ----
    fig = go.Figure()
    level_ranges = []  # list of (label, start_idx, end_idx)

    for lvl_label, lvl_value in level_items:
        start_idx = len(fig.data)
        if lvl_value == "country":
            traces = _build_country_traces(
                df_, var, code_convention, bins, color, tooltip_mode, df_planned
            )
        elif lvl_value == "city":
            traces = _build_city_traces(
                df_, var, color, tooltip_mode, df_planned
            )
        else:
            raise ValueError("level values must be 'country' or 'city'")
        for t in traces:
            fig.add_trace(t)
        level_ranges.append((lvl_label, start_idx, len(fig.data)))

    # ---- Initial visibility: only first level visible ----
    initial_label = level_items[0][0]
    n_total = len(fig.data)
    for label, start, end in level_ranges:
        is_initial = (label == initial_label)
        for i in range(start, end):
            fig.data[i].visible = is_initial

    fig.update_traces(hoverlabel=dict(bgcolor="black"))

    fig.update_geos(
        projection_type=initial_projection,
        showland=True,
        landcolor="white",
        showocean=True,
        oceancolor="skyblue",
        showcountries=False,
        showcoastlines=False,
        showframe=False,
        bgcolor="black",
    )

    # ---- Build dropdown menus (only if multiple options) ----
    menus = []
    next_y = 0.99
    if len(level_items) > 1:
        level_buttons = []
        for label, start, end in level_ranges:
            visible = [(start <= i < end) for i in range(n_total)]
            level_buttons.append(
                dict(label=label, method='update', args=[{'visible': visible}])
            )
        menus.append(_dropdown_menu(level_buttons, x=0.01, y=next_y))
        next_y -= 0.10
    if len(proj_items) > 1:
        proj_buttons = [
            dict(
                label=label,
                method='relayout',
                args=[{
                    'geo.projection.type': value,
                    'geo.projection.rotation.lon': 0,
                    'geo.projection.rotation.lat': 0,
                    'geo.projection.rotation.roll': 0,
                    'geo.projection.scale': 1,
                    'geo.center.lon': 0,
                    'geo.center.lat': 0,
                    'geo.lonaxis.range': None,
                    'geo.lataxis.range': None,
                }],
            )
            for label, value in proj_items
        ]
        menus.append(_dropdown_menu(proj_buttons, x=0.01, y=next_y))

    layout_kwargs = dict(
        paper_bgcolor="black",
        plot_bgcolor="black",
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode="pan",
        hovermode="closest",
        showlegend=False,
    )
    if menus:
        layout_kwargs['updatemenus'] = menus
    fig.update_layout(**layout_kwargs)

    config = {
        "displayModeBar": True,
        "displaylogo": False,
        "scrollZoom": True,
        "modeBarButtonsToRemove": [
            "toImage", "zoom2d", "pan2d",
            "lasso2d", "select2d",
            "autoScale2d",
            "hoverCompareCartesian",
            "hoverClosestCartesian",
            "toggleSpikelines",
            "resetScale2d",
        ],
        "modeBarButtonsToAdd": [
            "zoomIn2d", "zoomOut2d"
        ]
    }

    if save_path:
        write_kwargs = dict(include_plotlyjs="cdn", config=config)
        post_scripts = []
        if menus:
            post_scripts.append(_RESIZE_POST_SCRIPT)
        post_scripts.append(_RESET_HOOK_SCRIPT)
        if post_scripts:
            write_kwargs['post_script'] = post_scripts
        fig.write_html(save_path, **write_kwargs)

    return fig.show(config=config)

# Treemap drill-down: cancel the click (no zoom) on leaf tiles, i.e. single
# countries, so only tiles that contain other tiles (e.g. 'Others') zoom in.
# Group labels read 'Name<br>(value, pct)' on their own tiles, but the tile
# clicked into becomes a thin header with room for one line only: give that
# tile 'Name (value, pct)' just before the zoom animation draws it.
_TREE_LEAF_NOZOOM_SCRIPT = r"""
(function(){
  var gd = document.getElementById('{plot_id}');
  if (!gd || !gd.on) return;
  var trace = gd.data[0];
  var labels = (trace.labels || []).slice();   // original (two-line) labels
  var hasChildren = {};
  (trace.parents || []).forEach(function(p){ if (p) hasChildren[p] = true; });
  function headerLabels(entryId){
    var i = trace.ids.indexOf(entryId);
    var out = labels.slice();
    if (i >= 0) out[i] = String(out[i]).replace(/<br>\(/, ' (');
    return out;
  }
  gd.on('plotly_treemapclick', function(e){
    var pt = e && e.points && e.points[0];
    if (!pt) return;
    var id = pt.id !== undefined ? pt.id : trace.ids[pt.pointNumber];
    if (!hasChildren[id]) return false;
    var next = e.nextLevel !== undefined ? e.nextLevel : id;
    gd.data[0].labels = headerLabels(next);
  });
})();
"""

def create_tree(
        df,
        feat='genre',
        var='count',
        flag='flag',
        threshold=0,
        threshold_global=False,
        group_flag=False,
        title=None,
        color_dict={'Movie': 'royalblue', 'TV': 'gold', 'Others': 'white'},
        font_size_dict=None,
        expand_others=False,
        start_collapsed=False,
        log_scale=False,
        value_fmt=None,
        pct_fmt=None,
        name_dict=None,
        wrap_labels=True,
        save_path=None
        ):
    # expand_others: keep the countries merged into each group's 'Others' tile
    # as hidden children, so clicking 'Others' zooms in to show them (only for
    # group_flag=True with threshold_global=False).
    # start_collapsed: put all groups under one top tile labelled with the title
    # and open on the groups only; clicking a group zooms into its tiles (only
    # for group_flag=True).
    start_collapsed = start_collapsed and group_flag
    # log_scale: size each tile by log(1 + value) instead of value: groups by
    # log of their own total, and tiles within a group split the group's area
    # in proportion to their own log(1 + value). Labels still show the real
    # values and percentages.
    # value_fmt: optional function turning a value into the text shown on its
    # tile (e.g. days -> '45 days' / '2.3 years'); default prints the number.
    # pct_fmt: optional function (value, share of total) -> percentage text
    # (e.g. one decimal for small values), or ''/None to leave it out; default
    # is a whole-number percent.
    # name_dict: optional {original name: display name} for the feat column
    # (e.g. {'Russian Federation': 'Russia'}); names not in it are kept.
    # wrap_labels: break long tile names over several lines; False keeps every
    # name on one line.

    # ------------------ Font sizes ------------------
    if font_size_dict is not None:
        BASE_FONT_SIZE = font_size_dict['base']
        TITLE_FONT_SIZE = font_size_dict['title']
        TREE_TEXT_SIZE = font_size_dict['tree_text']
    else:
        BASE_FONT_SIZE = 20
        TITLE_FONT_SIZE = 22
        TREE_TEXT_SIZE = 20

    TREE_TEXT_TEMPLATE = "<br>%{label}<br>(%{value:.0f}, %{percentRoot:.0%})"

    df = df.copy()
    if name_dict:
        df[feat] = df[feat].replace(name_dict)
    df_orig = df.copy()
    df_copy = df.copy()

    # ------------------ Flat treemap: one tile per category ------------------
    # When the tile key and the color key are the same column (feat == flag)
    # the treemap is single-level. Collapse any duplicate rows (e.g. the same
    # genre split across Movie/TV) so each category becomes exactly one tile,
    # otherwise duplicate ids drop out and leave empty space.
    flat = feat == flag
    if flat:
        df_copy = df_copy.groupby(feat, as_index=False)[var].sum()
        df_orig = df_copy.copy()

    # ------------------ Threshold aggregation ------------------
    if threshold_global:
        large_entries = df_copy[df_copy[var] >= threshold]
        small_entries = df_copy[df_copy[var] < threshold]
        if small_entries[var].sum() > 0:
            if flat:
                # Single-level: collapse all small tiles into one 'Others' tile,
                # unless a single tile would end up in it (then just show it).
                if len(small_entries) == 1:
                    other_rows = small_entries.copy()
                else:
                    other_rows = pd.DataFrame([{feat: 'Others', var: small_entries[var].sum()}])
            else:
                # Nested: keep the feat breakdown (e.g. Movie/TV) so the 'Others'
                # block still splits into sub-blocks like the real blocks.
                other_rows = small_entries.groupby(feat, as_index=False)[var].sum()
                # An 'Others' block holding a single item is just that item.
                other_rows[flag] = other_rows[feat] if len(other_rows) == 1 else 'Others'
            df_copy = pd.concat([large_entries, other_rows], ignore_index=True)
    else:
        aggregated_rows = []
        for f, group in df_copy.groupby(flag):
            large_entries = group[group[var] >= threshold]
            small_entries = group[group[var] < threshold]
            aggregated_rows.append(large_entries)
            if small_entries[var].sum() > 0:
                if len(small_entries) == 1:
                    # An 'Others' tile holding a single item is just that item.
                    aggregated_rows.append(small_entries)
                else:
                    other_row = {feat: 'Others', flag: f, var: small_entries[var].sum()}
                    aggregated_rows.append(pd.DataFrame([other_row]))
                    if expand_others and group_flag:
                        aggregated_rows.append(small_entries.assign(_in_others=True))
        df_copy = pd.concat(aggregated_rows, ignore_index=True)
    if '_in_others' not in df_copy.columns:
        df_copy['_in_others'] = False
    df_copy['_in_others'] = df_copy['_in_others'].eq(True)
    drilldown = bool(df_copy['_in_others'].any()) or start_collapsed

    if 'Others' not in color_dict:
        color_dict['Others'] = 'gray'

    # ------------------ Dynamic text wrapping ------------------
    min_line_length = 10
    max_line_length = 30
    count_min = df_copy[var].min()
    count_max = df_copy[var].max()
    norm_var = f'norm_{var}'
    if count_max > count_min:
        df_copy[norm_var] = (df_copy[var] - count_min) / (count_max - count_min)
    else:
        df_copy[norm_var] = 1.0
    df_copy['line_length'] = (df_copy[norm_var] * (max_line_length - min_line_length) + min_line_length).astype(int)

    def wrap_feat(row):
        text = row[feat]
        max_len = row['line_length']
        if not wrap_labels or len(text) <= max_len:
            return text
        words = text.split(' ')
        lines = []
        current = ''
        for word in words:
            if len(current) + len(word) + (1 if current else 0) > max_len:
                if current:
                    lines.append(current)
                current = word
            else:
                current = f"{current} {word}" if current else word
        if current:
            lines.append(current)
        return '<br>'.join(lines)

    feat_wrapped = f'{feat}_wrapped'
    df_copy[feat_wrapped] = df_copy.apply(wrap_feat, axis=1)

    # ------------------ Hierarchy ------------------
    if group_flag:
        # ----- children -----
        df_copy['flag_id'] = df_copy[flag].astype(str)
        df_copy['id'] = df_copy['flag_id'] + ' | ' + df_copy[feat_wrapped]
        df_copy['parent'] = df_copy['flag_id']
        df_copy['label'] = df_copy[feat_wrapped]

        # Countries folded into 'Others' hang under that group's 'Others' tile
        in_others = df_copy['_in_others']
        df_copy.loc[in_others, 'parent'] = df_copy.loc[in_others, 'flag_id'] + ' | Others'
        df_copy.loc[in_others, 'id'] = df_copy.loc[in_others, 'parent'] + ' | ' + df_copy.loc[in_others, feat_wrapped]

        # ----- parent blocks -----
        # A block's numbers are built from what its sub-blocks actually display,
        # not from rounding the block's own total: rounding the sum and summing
        # the roundings disagree (e.g. 4.6 + 1.6 shows as 5 + 2 = 7, while the
        # 6.2 total would show as 6).
        # Countries inside 'Others' are already counted in the 'Others' tile
        root_total = df_copy.loc[~df_copy['_in_others'], var].sum()

        def round_display(x):
            # Match plotly's ':.0f' / ':.0%' (round halves away from zero)
            return int(np.floor(np.abs(x) + 0.5)) * (-1 if x < 0 else 1)

        df_copy['_shown_var'] = df_copy[var].apply(round_display)
        df_copy['_shown_pct'] = df_copy[var].apply(
            lambda v: round_display(100 * v / root_total) if root_total > 0 else 0
        )

        # Children of 'Others' are already counted in the 'Others' tile itself
        parent_df = df_copy[~df_copy['_in_others']].groupby('flag_id', as_index=False).agg(
            **{var: (var, 'sum'), '_shown_var': ('_shown_var', 'sum'), '_shown_pct': ('_shown_pct', 'sum')}
        )
        if start_collapsed:
            # Groups are shown on their own first (tiles hidden), so round each
            # group's own total; summing many rounded-down tiles can show 0%
            parent_df['_shown_var'] = parent_df[var].apply(round_display)
            # Under 1%: one decimal, so a small group doesn't read as 0%
            parent_df['_shown_pct'] = parent_df[var].apply(
                lambda v: (0 if root_total <= 0
                           else f"{100 * v / root_total:.1f}" if 100 * v / root_total < 1
                           else round_display(100 * v / root_total))
            )
        parent_df[flag] = parent_df['flag_id']
        parent_df['id'] = parent_df['flag_id']
        parent_df['parent'] = ""
        def group_label(name, v, shown_var, shown_pct):
            # 'Name' over '(value, pct)' for a tile holding other tiles;
            # value_fmt / pct_fmt override the defaults, and an empty pct is
            # left out. Leading breaks push it toward the top so it doesn't
            # overlap the tiles inside when zoomed in.
            if value_fmt is not None:
                shown_var = value_fmt(v)
            if pct_fmt is not None:
                shown_pct = pct_fmt(v, v / root_total if root_total > 0 else 0)
            else:
                shown_pct = f"{shown_pct}%"
            pct_part = f", {shown_pct}" if shown_pct else ""
            return f"<br><br>{name}<br>({shown_var}{pct_part})"

        parent_df['label'] = parent_df.apply(
            lambda r: group_label(r['flag_id'], r[var], r['_shown_var'], r['_shown_pct']),
            axis=1
        )

        # Parents print their pre-computed label (numbers already summed from the
        # children); children get the value/percent straight from the treemap.
        parent_df['_texttemplate'] = '<br>%{label}'
        df_copy['_texttemplate'] = TREE_TEXT_TEMPLATE
        # Countries inside 'Others' are small by definition: keep one decimal
        # (trailing zeros trimmed) so sub-1 values don't all read as 0
        df_copy.loc[df_copy['_in_others'], '_texttemplate'] = (
            "<br>%{label}<br>(%{value:.1~f}, %{percentRoot:.1~%})"
        )
        # An expanded 'Others' tile is a group too: label it like the groups,
        # so its total stays visible in the header once clicked into
        expanded = df_copy['id'].isin(df_copy.loc[df_copy['_in_others'], 'parent'])
        df_copy.loc[expanded, 'label'] = df_copy.loc[expanded, var].apply(
            lambda v: group_label(
                'Others', v, round_display(v),
                round_display(100 * v / root_total) if root_total > 0 else 0,
            )
        )
        df_copy.loc[expanded, '_texttemplate'] = '<br>%{label}'

        df_copy = pd.concat([parent_df, df_copy], ignore_index=True)

        if start_collapsed:
            # One explicit top tile above the groups: with maxdepth=2 the first
            # view is top tile + groups, and zooming into a group shows its tiles
            ROOT_ID = '__root__'
            df_copy.loc[df_copy['parent'] == "", 'parent'] = ROOT_ID
            root_row = {
                'id': ROOT_ID, 'parent': "", flag: ROOT_ID,
                var: parent_df[var].sum(),
                'label': title if title is not None else '',
                '_texttemplate': '%{label}',
                '_in_others': False,
            }
            df_copy = pd.concat([pd.DataFrame([root_row]), df_copy], ignore_index=True)
    else:
        df_copy['id'] = df_copy[flag] + ' | ' + df_copy[feat_wrapped]
        df_copy['parent'] = ""
        df_copy['label'] = df_copy[feat_wrapped]
        df_copy['_texttemplate'] = TREE_TEXT_TEMPLATE

    # ------------------ Colors ------------------
    df_copy['_color'] = df_copy[flag].map(color_dict).fillna('gray')
    if start_collapsed:
        df_copy.loc[df_copy['id'] == '__root__', '_color'] = 'rgb(17,17,17)'

    # ------------------ Tile sizes ------------------
    if log_scale:
        # Every tile's weight is log(1 + its own real value). Built top-down:
        # top-level tiles take their weight as size; the children of a tile
        # are scaled so they fill it exactly, in proportion to their weights.
        # (Summing children's logs bottom-up instead would make a group with
        # many small tiles outgrow one with a few large tiles.)
        ids = df_copy['id'].tolist()
        parent_of = dict(zip(ids, df_copy['parent']))
        weight = dict(zip(ids, np.log1p(df_copy[var].astype(float))))
        children = {}
        for i in ids:
            if parent_of[i]:
                children.setdefault(parent_of[i], []).append(i)

        size = {}
        queue = [i for i in ids if not parent_of[i]]
        for i in queue:
            size[i] = weight[i]
        while queue:
            i = queue.pop()
            kids = children.get(i, [])
            total_w = sum(weight[k] for k in kids)
            # Shrink a hair so float error never lets children exceed the parent
            scale = size[i] / total_w * (1 - 1e-9) if total_w > 0 else 0.0
            for k in kids:
                size[k] = weight[k] * scale
            queue.extend(kids)
        df_copy['_size'] = df_copy['id'].map(size)
    else:
        df_copy['_size'] = df_copy[var]

    # ------------------ Tile text ------------------
    # With log sizes the treemap's own %{value} / %{percentRoot} refer to the
    # log sizes, and value_fmt needs Python-side text: in both cases labels
    # read the real value / percent / formatted text from customdata instead
    customdata = None
    if log_scale or value_fmt is not None or pct_fmt is not None:
        # Top-level tiles (groups, or every tile when flat) add up to the total
        top_level = df_copy['parent'].isin(['', '__root__']) & (df_copy['id'] != '__root__')
        real_total = df_copy.loc[top_level, var].sum()
        customdata = np.column_stack([
            df_copy[var].astype(object),
            (df_copy[var] / real_total if real_total > 0 else 0 * df_copy[var]).astype(object),
            df_copy[var].apply(value_fmt) if value_fmt is not None else df_copy[var].astype(str),
        ])
        if pct_fmt is not None:
            pct_text = [(', ' + t) if t else '' for t in
                        (pct_fmt(v, f) for v, f in zip(customdata[:, 0], customdata[:, 1]))]
            customdata = np.column_stack([customdata, np.array(pct_text, dtype=object)])
        tt = df_copy['_texttemplate']
        if value_fmt is not None:
            tt = tt.str.replace(r'%\{value(:[^}]*)?\}', '%{customdata[2]}', regex=True)
        if pct_fmt is not None:
            tt = tt.str.replace(r', %\{percentRoot(:[^}]*)?\}', '%{customdata[3]}', regex=True)
        df_copy['_texttemplate'] = (
            tt.str.replace('%{value', '%{customdata[0]', regex=False)
              .str.replace('%{percentRoot', '%{customdata[1]', regex=False)
        )

    # ------------------ Treemap ------------------
    fig = go.Figure(go.Treemap(
        ids=df_copy['id'],
        labels=df_copy['label'],
        parents=df_copy['parent'],
        values=df_copy['_size'],
        customdata=customdata,
        marker=dict(
            colors=df_copy['_color'],
            line=dict(color='black', width=1)
        ),
        textinfo="label+value+percent root",
        texttemplate=df_copy['_texttemplate'],
        textposition="top left",
        textfont=dict(size=TREE_TEXT_SIZE),
        hoverinfo='none',
        branchvalues='total',
        # With drill-down, show two levels from the current tile: groups + tiles
        # (or top tile + groups when start_collapsed); deeper levels such as the
        # countries inside 'Others' appear once their parent is clicked
        maxdepth=2 if drilldown else -1,
    ))

    # ------------------ Title (flat only) ------------------
    if title is None and not group_flag:
        flag_counts_orig = (
            df_orig.groupby(flag, as_index=False)[var]
            .sum()
            .sort_values(var, ascending=False)
        )
        total_count_orig = flag_counts_orig[var].sum()
        flag_summaries = []
        for _, row in flag_counts_orig.iterrows():
            name = row[flag]
            count = row[var]
            frac = count / total_count_orig if total_count_orig > 0 else 0
            color = color_dict.get(name, 'gray')
            flag_summaries.append(
                f"<span style='color:{color}'>{name} ({int(count)}, {frac:.0%})</span>"
            )
        title = ", ".join(flag_summaries)

    else:
        title = title

    fig.update_layout(
        title=dict(
        # With start_collapsed the title is already the top tile's label
        text=None if start_collapsed else title,
        x=0.5,
        xanchor='center',
        y=0.99,
        yanchor='top',
        font=dict(size=TITLE_FONT_SIZE)
        )
    )

    t_margin=6 if not group_flag else 0
    
    # ------------------ Layout ------------------
    fig.update_layout(
        margin=dict(l=0, r=0, t=t_margin, b=0),
        template='plotly_dark',
        font=dict(color='white', size=BASE_FONT_SIZE),
        uniformtext=dict(minsize=TREE_TEXT_SIZE, mode='show'),
    )

    # ------------------ Output ------------------
    # Clicks must stay live for the 'Others' drill-down; otherwise keep it static
    config = {'displayModeBar': False} if drilldown else {'staticPlot': True}
    if save_path:
        write_kwargs = dict(file=save_path, config=config, include_plotlyjs='cdn')
        if drilldown:
            write_kwargs['post_script'] = _TREE_LEAF_NOZOOM_SCRIPT
        pio.write_html(fig, **write_kwargs)

    return fig.show(config=config)