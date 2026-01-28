import numpy as np
import pandas as pd
import pycountry
import country_converter as coco
from collections import Counter

def transform_verbose_to_calendar(df, save=False):
    # Step 1: Build date → list of (row_id, start, end, country, city, coordinates)
    date_to_entries = {}
    for i, row in df.iterrows():
        start, end = pd.to_datetime(row['start_date']), pd.to_datetime(row['end_date'])
        days = pd.date_range(start, end, freq='D', inclusive='both')
        for day in days:
            date_to_entries.setdefault(day, []).append(
                (i, start, end, row['country'], row['city'], row['coordinates'])
            )

    # Step 2: Allocate per-day fractions
    records = []
    for day, entries in date_to_entries.items():
        core_entries = []
        boundary_entries = []

        # Classify entries
        for row_id, start, end, country, city, coordinates in entries:
            if start < day < end:  # core day
                core_entries.append((row_id, country, city, coordinates))
            else:  # boundary day
                boundary_entries.append((row_id, country, city, coordinates))

        # Core days: full credit
        for row_id, country, city, coordinates in core_entries:
            records.append((day, country, city, coordinates, 1.0))

        # Boundary days: split evenly
        if boundary_entries:
            weight = 1.0 / len(boundary_entries)
            for row_id, country, city, coordinates in boundary_entries:
                records.append((day, country, city, coordinates, weight))

    # Step 3: Aggregate
    dc_df = pd.DataFrame(records, columns=['date', 'country', 'city', 'coordinates', 'day_fraction'])
    dc_df['date'] = pd.to_datetime(dc_df['date'])
    dc_df['year'] = dc_df['date'].dt.year

    grouped = (dc_df.groupby(['country', 'city', 'coordinates', 'year'])['day_fraction']
                 .sum().unstack(fill_value=0))
    grouped.columns.name = None

    # Step 4: Add total per place column
    grouped['total_days'] = grouped.sum(axis=1)

    # Step 5: Add total per year row
    all_row = grouped.drop(columns='total_days').sum()
    all_row['total_days'] = grouped['total_days'].sum()
    grouped.loc[('All', None, None)] = all_row

    # Step 6: Move 'total_days' to first column
    cols = ['total_days'] + [c for c in grouped.columns if c != 'total_days']
    final_df = grouped[cols].reset_index()

    # Step 7: Add continent column
    final_df['continent'] = final_df['country'].apply(
        lambda x: coco.convert(names=x, to='continent') if x != 'All' else None
    )

    # Step 8: Reorder columns
    fixed_cols = ['country', 'city', 'coordinates', 'continent', 'total_days']
    year_cols = [c for c in final_df.columns if c not in fixed_cols]
    final_df = final_df[fixed_cols + year_cols]

    # Step 9: Sort
    final_df = final_df.sort_values(by='total_days', ascending=False).reset_index(drop=True)

    if save:
        final_df.to_csv('trips_calendar.csv', index=False)

    return final_df.round(2)

def transform_counts_to_calendar(df):
    # Step 1: Normalize country strings
    def normalize_countries(country_str):
        # Fix spacing issues and split
        countries = [c.strip() for c in country_str.replace(' ,', ',').replace(', ', ',').split(',')]
        count = Counter(countries)
        total = sum(count.values())
        return {c: v / total for c, v in count.items()}

    # Step 2: Expand DataFrame
    rows = []
    for idx, row in df.iterrows():
        # Ensure dates_read is a list
        years = row['dates_read']
        if isinstance(years, int):
            years = [years]
        country_shares = normalize_countries(row['country'])
        for country, share in country_shares.items():
            for year in years:
                rows.append({'country': country, 'year': year, 'count': share})

    df_expanded = pd.DataFrame(rows)

    # Step 3: Pivot table
    df_format = df_expanded.pivot_table(index='country', columns='year', values='count', aggfunc='sum', fill_value=0)

    # Step 4: Add total_books column
    df_format['total_books'] = df_format.sum(axis=1)

    # Optional: reorder columns to have total_books first
    cols = ['total_books'] + [col for col in df_format.columns if col != 'total_books']
    df_format = df_format[cols]

    # Reset index to have 'country' as a column
    df_format = df_format.sort_values('total_books', ascending=False).round(2).reset_index()
    return df_format

def get_country_codes(country_name):
    try:
        country = pycountry.countries.lookup(country_name)
        return country.alpha_2, country.alpha_3
    except LookupError:
        return None, None

# Helper function to add alpha-2 and alpha-3 codes
def add_country_codes(df, name_col='country'):
    def get_alpha2(name):
        try:
            return pycountry.countries.lookup(name).alpha_2
        except LookupError:
            return None

    def get_alpha3(name):
        try:
            return pycountry.countries.lookup(name).alpha_3
        except LookupError:
            return None

    df = df.copy()
    df['code2'] = df[name_col].apply(get_alpha2)
    df['code3'] = df[name_col].apply(get_alpha3)
    return df

def format_days_to_ymwd(n):
    def fmt_days(x):
        if x >= 1:
            x = int(round(x))
        else:
            x = round(x, 2)
        return f"{x} day" if x == 1 else f"{x} days"
    if n == 1:
        return "1 day"
    # Days only
    if n < 7:
        return fmt_days(n)
    # Weeks
    elif 7 <= n < 30:
        weeks = int(n // 7)
        rem_days = n - weeks * 7
        parts = [f"{weeks} week" if weeks == 1 else f"{weeks} weeks"]
        if rem_days > 0:
            parts.append(fmt_days(rem_days))
        return " ".join(parts)
    # Months
    elif 30 <= n < 365:
        months = int(n // 30)
        rem_days = n - months * 30
        weeks = int(rem_days // 7)
        rem_days = rem_days - weeks * 7
        parts = [f"{months} month" if months == 1 else f"{months} months"]
        if weeks:
            parts.append(f"{weeks} week" if weeks == 1 else f"{weeks} weeks")
        if rem_days > 0:
            parts.append(fmt_days(rem_days))
        return " ".join(parts)
    # Years
    else:
        years = int(n // 365)
        rem_days = n - years * 365
        months = int(rem_days // 30)
        rem_days = rem_days - months * 30
        weeks = int(rem_days // 7)
        rem_days = rem_days - weeks * 7
        parts = [f"{years} year" if years == 1 else f"{years} years"]
        if months:
            parts.append(f"{months} month" if months == 1 else f"{months} months")
        if weeks:
            parts.append(f"{weeks} week" if weeks == 1 else f"{weeks} weeks")
        if rem_days > 0:
            parts.append(fmt_days(rem_days))
        return " ".join(parts)

def update_country_data(df, country_name, year_updates=None, new_days=None):
    # Locate the row for the country
    row_idx = df[df['country'] == country_name].index
    if len(row_idx) == 0:
        raise ValueError(f"Country '{country_name}' not found in DataFrame.")
    row_idx = row_idx[0]

    # Apply year updates
    if year_updates:
        for year, value in year_updates.items():
            if year not in df.columns:
                df[year] = 0
            df.at[row_idx, year] = value

    # Update days and recalculate log_days
    if new_days is not None:
        df.at[row_idx, 'days'] = new_days

    # Ensure days is numeric and calculate log_days
    days_val = df.at[row_idx, 'days']
    df.at[row_idx, 'log_days'] = np.log(days_val) if days_val > 0 else 0.0
    df.to_csv('map.csv', index=False)

    return df

def geometric_series(r, a=1, n=12):
    s = a*(r**n-1)/(r-1)
    return s