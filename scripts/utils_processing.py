import numpy as np
import pandas as pd
import pycountry
from collections import Counter
from sklearn.metrics import confusion_matrix

def transform_verbose_to_calendar(df, save=False):
    # Step 1: Build date → country map with first-claim-wins (to preserve day trips)
    date_to_country = {}
    for _, row in df.iterrows():
        days = pd.date_range(row['start_date'], row['end_date'], freq='D', inclusive='left')  # inclusive
        for day in days:
            if day not in date_to_country:
                date_to_country[day] = row['country']  # assign only if not already claimed

    # Step 2: Convert to DataFrame
    dc_df = pd.DataFrame(list(date_to_country.items()), columns=['date', 'country'])
    dc_df['year'] = dc_df['date'].dt.year

    # Step 3: Count days per (country, year)
    grouped = dc_df.groupby(['country', 'year']).size().unstack(fill_value=0)
    grouped.columns.name = None

    # Step 4: Add total per country column
    grouped['total_days'] = grouped.sum(axis=1)

    # Step 5: Add total per year row
    grouped.loc['All'] = grouped.drop(columns='total_days').sum()
    grouped.at['All', 'total_days'] = grouped['total_days'].sum()

    # Step 6: Move 'Total_days' to first column
    cols = ['total_days'] + [col for col in grouped.columns if col != 'total_days']
    final_df = grouped[cols].reset_index()
    final_df = final_df.apply(lambda col: col.astype('Int64') if pd.api.types.is_numeric_dtype(col) else col)
    final_df = final_df.sort_values(by='country', ascending=True).sort_values(by='total_days', ascending=False).reset_index(drop=True)
    if save == True:
        final_df.to_csv('trips_calendar.csv', index=False)
    return final_df

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
    if n == 1:
        return "1 day"
    elif n < 7:
        return f"{n} days"
    elif 7 <= n < 30:
        weeks = n // 7
        rem_days = n % 7
        parts = [f"{weeks} week" if weeks == 1 else f"{weeks} weeks"]
        if rem_days:
            parts.append(f"{rem_days} day" if rem_days == 1 else f"{rem_days} days")
        return " ".join(parts)
    elif 30 <= n < 365:
        months = n // 30
        rem_days = n % 30
        weeks = rem_days // 7
        days = rem_days % 7

        parts = [f"{months} month" if months == 1 else f"{months} months"]
        if weeks:
            parts.append(f"{weeks} week" if weeks == 1 else f"{weeks} weeks")
        if days:
            parts.append(f"{days} day" if days == 1 else f"{days} days")
        return " ".join(parts)
    else:
        years = n // 365
        rem_days = n % 365
        months = rem_days // 30
        rem_days %= 30
        weeks = rem_days // 7
        days = rem_days % 7

        parts = [f"{years} year" if years == 1 else f"{years} years"]
        if months:
            parts.append(f"{months} month" if months == 1 else f"{months} months")
        if weeks:
            parts.append(f"{weeks} week" if weeks == 1 else f"{weeks} weeks")
        if days:
            parts.append(f"{days} day" if days == 1 else f"{days} days")
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