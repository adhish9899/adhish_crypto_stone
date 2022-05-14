
import pandas as pd


def get_bid_ask_dict():
    df = pd.read_csv("sample_bid_ask.csv.gzip", compression="gzip").set_index("time")
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S.%f")
    df = df.sort_index()

    ## since we are using bid-ask data to calculate the mid price, we need only the best bid and best ask
    bid_ask_dict = {}
    for i, ind in enumerate(df.index.unique()):
        df_ = df.loc[ind]
        best_bid = df_[df_["side"] == "bids"]["price"].max()
        best_ask = df_[df_["side"] == "asks"]["price"].min()
        bid_ask_dict[i] = (best_bid, best_ask)

    return bid_ask_dict


class order(object):
    def __init__(self, symbol, price, qty, side, time, order_id):
        self.symbol = symbol
        self.price = price
        self.qty = qty
        self.side = side
        self.time = time
        self.order_id = order_id
        self.status = "pending" # pending, filled, partial, cancelled

class market_maker:
    
    prev_theo_price = None
    open_order_list = []

    def __init__(self, n_orders, offset, step_size):
        self.n_orders = n_orders
        self.offset = offset
        self.step_size = step_size

    def on_market_data(self, best_bid, best_ask):
        self.theo_price = (best_bid + best_ask) / 2

        self.place_orders()





