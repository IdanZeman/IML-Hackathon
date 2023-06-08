from pandas import DataFrame


def preprocess_lines_training(df: DataFrame):
    df = df.drop_duplicates()
    df['cancellation_datetime'].drop()
    for c in ['request_nonesmoke', 'request_latecheckin', 'request_highfloor', 'request_largebed', 'request_twinbeds',
              'request_airport', 'request_earlycheckin']:
        df[c] = df[c].fillna(0)
        df = df[2 > df[c] >= 0]
        df[c] = df[c].astype(int)

    df['hotel_brand_code'] = df['hotel_brand_code'].fillna('No brand')
    df.dropna()

    for c in ["no_of_adults", 'no_of_room', ]:
        df = df[df[c] > 0]
    for c in ["no_of_children", 'original_selling_amount', ]:
        df = df[df[c] >= 0]

    df = df[6 > df["hotel_star_rating"] >= 0]
    return df
