import numpy as np
import pandas as pd
from pandas import DataFrame


# Idan and Gal function
def preprocess_lines_training(df: DataFrame):
    df = df.drop_duplicates()

    for c in ['request_latecheckin', 'request_highfloor', 'request_largebed', 'request_twinbeds',
              'request_airport', 'request_earlycheckin']:
        df[c] = df[c].fillna(0)
        df[c] = df[c].astype(int)
        if not set(df[c]).issubset({0, 1}):
            df.loc[~df[c].isin([0, 1]), c] = 0

    df['hotel_brand_code'] = df['hotel_brand_code'].fillna(-1)
    df.dropna()

    df['accommadation_type_name'] = df['accommadation_type_name'].fillna(-1)
    df['accommadation_type_name'] = pd.factorize(df['accommadation_type_name'])[0]


    for c in ["no_of_adults", 'no_of_room', ]:
        df = df[df[c] > 0]
    for c in ["no_of_children", 'original_selling_amount', ]:
        df = df[df[c] >= 0]
    df["hotel_star_rating"] = df["hotel_star_rating"].fillna(0).notnull().astype(float)
    df = df[(df["hotel_star_rating"] < 6) & (df["hotel_star_rating"] >= 0)]
    return df


def preprocess_train(X_train: DataFrame, y_train: DataFrame):
    """

    :param filename:
    :return:
    """
    drop_for_now = ['guest_nationality_country_name','hotel_live_date','original_payment_method']

    to_drop = ['hotel_chain_code', 'hotel_area_code', 'request_nonesmoke',
               'original_payment_currency', 'language', 'origin_country_code', 'no_of_extra_bed',
               'customer_nationality', 'h_customer_id']
    X_train.insert(loc=len(X_train.columns), column="cancellation_datetime", value=y_train)
    df = preprocess_lines_training(X_train)
    df = df.drop(to_drop + drop_for_now, axis=1)

    to_date_time = ['booking_datetime', 'checkin_date', 'checkout_date', 'cancellation_datetime']
    y = df['cancellation_datetime'].notnull().astype(int)

    df[to_date_time] = df[to_date_time].apply(pd.to_datetime)

    df['time_reserve_before_checking'] = (df['checkin_date'] - df['booking_datetime']).dt.days
    df['num_of_day_to_stay'] = (df['checkout_date'] - df['checkin_date']).dt.days
    df = df.drop(to_date_time, axis=1)

    df = pd.get_dummies(df, prefix='in_country_', columns=['hotel_country_code'])
    df = pd.get_dummies(df,prefix='cancellation_policy_', columns=['cancellation_policy_code'])

    df = pd.get_dummies(df, prefix='payment_type_', columns=['original_payment_type'])
    df['charge_option'] = df['charge_option'].replace({'Pay Now': 1, 'Pay Later': 0, 'Pay at Check-in': 0})
    df['is_user_logged_in'] = df['is_user_logged_in'].astype(int)
    df['is_first_booking'] = df['is_first_booking'].astype(int)
    return df, y


def get_average(vec: np.ndarray):
    return np.nanmean(vec)

