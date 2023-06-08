import numpy as np
import pandas as pd
from pandas import DataFrame
PERCENTAGE_FACTOR = 100

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


def preprocess_lines_testing(df: DataFrame):
    df = df.drop_duplicates()
    for c in ['request_latecheckin', 'request_highfloor', 'request_largebed', 'request_twinbeds',
              'request_airport', 'request_earlycheckin']:
        df[c] = df[c].fillna(0)
        df[c] = df[c].astype(int)
        if not set(df[c]).issubset({0, 1}):
            df.loc[~df[c].isin([0, 1]), c] = 0

    df['hotel_brand_code'] = df['hotel_brand_code'].fillna(-1)
    df['accommadation_type_name'] = df['accommadation_type_name'].fillna(-1)
    df['accommadation_type_name'] = pd.factorize(df['accommadation_type_name'])[0]

    for c in ["no_of_adults", 'no_of_room', ]:
        df[c] = df[c].where(df[c] > 0, other=get_average(df[c]))
    for c in ["no_of_children", 'original_selling_amount', ]:
        df[c] = df[c].where(df[c] >= 0, other=get_average(df[c]))
    df["hotel_star_rating"] = df["hotel_star_rating"].fillna(0).astype(float)
    df[c] = df[c].where((df[c] < 6) & (df[c] >= 0), other=get_average(df[c]))
    return df


def preprocess_test(X_test: DataFrame):
    """

    :param filename:
    :return:
    """
    drop_for_now = ['guest_nationality_country_name', 'hotel_live_date', 'original_payment_method']

    to_drop = ['hotel_chain_code', 'hotel_area_code', 'request_nonesmoke',
               'original_payment_currency', 'language', 'origin_country_code', 'no_of_extra_bed',
               'customer_nationality', 'h_customer_id']
    df = preprocess_lines_testing(X_test)
    df = df.drop(to_drop + drop_for_now, axis=1)

    to_date_time = ['booking_datetime', 'checkin_date', 'checkout_date', 'cancellation_datetime']
    y = df['cancellation_datetime'].notnull().astype(int)

    df[to_date_time] = df[to_date_time].apply(pd.to_datetime)

    df['time_reserve_before_checking'] = (df['checkin_date'] - df['booking_datetime']).dt.days
    df['num_of_day_to_stay'] = (df['checkout_date'] - df['checkin_date']).dt.days
    df = df.drop(to_date_time, axis=1)

    df = pd.get_dummies(df, prefix='in_country_', columns=['hotel_country_code'])
    df = pd.get_dummies(df, prefix='cancellation_policy_', columns=['cancellation_policy_code'])

    df = pd.get_dummies(df, prefix='payment_type_', columns=['original_payment_type'])
    df['charge_option'] = df['charge_option'].replace({'Pay Now': 1, 'Pay Later': 0, 'Pay at Check-in': 0})
    df['is_user_logged_in'] = df['is_user_logged_in'].astype(int)
    df['is_first_booking'] = df['is_first_booking'].astype(int)
    return df, y


def preprocess(X: DataFrame, y: DataFrame):
    """

    :param filename:
    :return:
    """
    new_y = None
    global dummies_hotel_country_code
    global dummis_cancellation_policy_code
    global dummis_original_payment_type
    global global_train_X
    if y is not None:  # In train
        X.insert(loc=len(X.columns), column="cancellation_datetime", value=y)
        X = preprocess_lines_training(X)
        new_y = X['cancellation_datetime'].notnull().astype(int)
        X = X.drop(columns=['cancellation_datetime'])
    else:  # In test
        X = preprocess_lines_testing(X)
    X = pd.get_dummies(X, prefix='in_country_', columns=['hotel_country_code'])
    X = pd.get_dummies(X, prefix='cancellation_policy_', columns=['cancellation_policy_code'])
    X = pd.get_dummies(X, prefix='payment_type_', columns=['original_payment_type'])
    drop_for_now = ['guest_nationality_country_name', 'hotel_live_date', 'original_payment_method']
    to_drop = ['hotel_id', 'hotel_brand_code', 'hotel_city_code', 'hotel_chain_code', 'hotel_area_code',
               'request_nonesmoke',
               'original_payment_currency', 'language', 'origin_country_code', 'no_of_extra_bed',
               'customer_nationality', 'h_customer_id']
    X = X.drop(to_drop + drop_for_now, axis=1)
    to_date_time = ['booking_datetime', 'checkin_date', 'checkout_date']

    X[to_date_time] = X[to_date_time].apply(pd.to_datetime)



    X['time_reserve_before_checking'] = (X['checkin_date'] - X['booking_datetime']).dt.days
    X['num_of_day_to_stay'] = (X['checkout_date'] - X['checkin_date']).dt.days
    X = X.drop(to_date_time, axis=1)
    X['charge_option'] = X['charge_option'].replace({'Pay Now': 1, 'Pay Later': 0, 'Pay at Check-in': 0})
    X['is_user_logged_in'] = X['is_user_logged_in'].astype(int)
    X['is_first_booking'] = X['is_first_booking'].astype(int)
    return X, new_y


def get_average(vec: np.ndarray):
    return np.nanmean(vec)


def split_policy_code(s: str, total_day: int):
    days = set()
    if s == 'UNKNOWN':
        return [(0, 0)], set()

    splited_polcis = s.split('_')
    res = []
    for s in splited_polcis:
        index_D = s.find('D')
        num_days = int(s[:index_D])
        days.add(num_days)
        percent, percent_id = 0, s.find('P')
        if percent_id != -1:
            percent = int(s[index_D + 1:percent_id]) / PERCENTAGE_FACTOR
        else:
            percent_id = s.find('N')
            percent = int(s[index_D + 1:percent_id]) / total_day
        res.append((num_days, percent))

    return res, days