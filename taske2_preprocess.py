import pandas as pd


PERCENTAGE_FACTOR = 100
from preprocess import  preprocess_lines_training
from preprocess import  preprocess_lines_testing

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


# def task2_preprocess(X: pd.DataFrame, y: pd.Series):
#     """
#     :param X:
#     :param filename:
#     :return:
#     """
#     if y:
#         X = pd.concat(X, y)
#
#     drop_for_now = ['guest_nationality_country_name', 'hotel_live_date', 'original_payment_method']
#
#     to_drop = ['hotel_chain_code', 'hotel_area_code', 'request_nonesmoke',
#                'original_payment_currency', 'language', 'origin_country_code',
#                'customer_nationality', 'h_customer_id', 'is_user_logged_in', 'is_first_booking']
#
#     X = preprocess_lines_training(X)
#     X = X.drop(to_drop + drop_for_now, axis=1)
#
#     to_date_time = ['booking_datetime', 'checkin_date', 'checkout_date', 'cancellation_datetime']
#     y = X['cancellation_datetime'].notnull().astype(int)
#     X[to_date_time] = X[to_date_time].apply(pd.to_datetime)
#
#     X['time_reserve_before_checking'] = (X['checkin_date'] - X['booking_datetime']).dt.days
#     X['delta_cancel_before_checking'] = (X['checkin_date'] - X['cancellation_datetime']).dt.days
#     X['delta_cancel_before_checking'] = X['delta_cancel_before_checking'].fillna(0)
#     X['num_of_day_to_stay'] = (X['checkout_date'] - X['checkin_date']).dt.days
#     X = X.drop(to_date_time, axis=1)
#
#     col_days = set()
#     for i in range(len(X)):
#         current_res, days = split_policy_code(X.loc[i, 'cancellation_policy_code'], X.loc[i, 'num_of_day_to_stay'])
#         if days:
#             col_days.update(days)
#         for day_perc in current_res:
#             X.loc[i, day_perc[0]] = day_perc[1]
#
#     for c in col_days:
#         X[c] = X[c].fillna(0)
#
#     X = pd.get_dummies(X, prefix='in_country_', columns=['hotel_country_code'])
#    # X = pd.get_dummies(X, prefix='cancellation_policy_', columns=['cancellation_policy_code'])
#
#     X = pd.get_dummies(X, prefix='payment_type_', columns=['original_payment_type'])
#     X['charge_option'] = X['charge_option'].replace({'Pay Now': 1, 'Pay Later': 0, 'Pay at Check-in': 0})
#     return X, y

def preprocess(X: pd.DataFrame, y: pd.DataFrame):
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
        X = pd.concat([X,y],axis=1)
        X = preprocess_lines_training(X)
        new_y = X['original_selling_amount'].astype(float)
        X = X.drop(columns=['original_selling_amount'])
    else:  # In test
        X = preprocess_lines_testing(X)

    X = pd.get_dummies(X, prefix='in_country_', columns=['hotel_country_code'])
    X = pd.get_dummies(X, prefix='cancellation_policy_', columns=['cancellation_policy_code'])
    X = pd.get_dummies(X, prefix='payment_type_', columns=['original_payment_type'])

    drop_for_now = ['guest_nationality_country_name', 'hotel_live_date', 'original_payment_method']
    to_drop = ['hotel_id', 'hotel_brand_code', 'hotel_city_code', 'hotel_chain_code', 'hotel_area_code',
               'request_nonesmoke',
               'original_payment_currency', 'language', 'origin_country_code', 'no_of_extra_bed',
               'customer_nationality', 'h_customer_id','cancellation_datetime']
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
