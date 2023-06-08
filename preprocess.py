import numpy as np
import pandas as pd
from pandas import DataFrame


# Idan and Gal function
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


def preprocess(filename: str):
    """

    :param filename:
    :return:
    """
    drop_for_now = ['request_earlycheckin', 'request_airport', 'request_twinbeds', 'request_largebed',
                    'request_highfloor', 'request_latecheckin', 'guest_nationality_country_name']

    to_drop = ['hotel_chain_code', 'hotel_area_code', 'hotel_brand_code', 'request_nonesmoke',
               'original_payment_currency', 'language', 'origin_country_code', 'no_of_extra_bed',
               'customer_nationality', 'h_customer_id']
    df = pd.read_csv(filename).drop_duplicates()
    df = df.drop(to_drop + drop_for_now, axis=1)

    to_date_time = ['booking_datetime', 'checkin_date', 'checkout_date', 'cancellation_datetime']
    df[to_date_time] = df[to_date_time].apply(pd.to_datetime)

    df['time_reserve_before_checking'] = (df['checkin_date'] - df['booking_datetime']).dt.days
    df['delta_cancel_before_checking'] = (df['checkin_date'] - df['cancellation_datetime']).dt.days
    df['num_of_day_to_stay'] = (df['checkout_date'] - df['checkin_date']).dt.days
    df = df.drop(to_date_time, axis=1)

    df = pd.get_dummies(df, prefix='in_country', columns=['hotel_country_code'])
    df['charge_option'] = df['charge_option'].replace({'Pay Now': 1, 'Pay Later': 0})

    return df


df1 = preprocess('Data/agoda_cancellation_train.csv')
print(df1.head(n=1).to_csv('check.csv', index=False))
