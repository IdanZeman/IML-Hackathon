from pandas import DataFrame


def preprocess_lines_training(df: DataFrame):
    """
    Load agoda dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to agoda dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
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

    df = df[df["waterfront"].isin([0, 1]) &
            df["view"].isin(range(5)) &
            df["condition"].isin(range(1, 6)) &
            df["grade"].isin(range(1, 15))]

    df["recently_renovated"] = np.where(df["yr_renovated"] >= np.percentile(df.yr_renovated.unique(), 70), 1, 0)
    df = df.drop("yr_renovated", 1)

    df["decade_built"] = (df["yr_built"] / 10).astype(int)
    df = df.drop("yr_built", 1)

    df = pd.get_dummies(df, prefix='zipcode_', columns=['zipcode'])
    df = pd.get_dummies(df, prefix='decade_built_', columns=['decade_built'])

    # Removal of outliers (Notice that there exists methods for better defining outliers
    # but for this course this suffices
    df = df[df["bedrooms"] < 20]
    df = df[df["sqft_lot"] < 1250000]

    return df.drop("price", axis=1), df.price