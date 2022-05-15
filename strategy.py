
import pandas as pd
import uuid
import datetime as dt

'''
1. To Do. (lte, gte)

'''


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

## USING IT FROM "bisect" MODULE and updating it to add reverse as well. 
def insort(a, x, reverse=False, insert=True, lo=0, hi=None):
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if (x > a[mid]) and not reverse:
            hi = mid
        
        elif (x < a[mid]) and reverse:
            hi = mid

        else: lo = mid+1
        
    if insert:        
        a.insert(lo, x)
    
    else:
        if lo != len(a) and a[lo-1] == x:
            return lo-1

        return -1

##
class create_order(object):
    def __init__(self, symbol, price, qty, side, time, filled_qty, order_id):
        self.exchange_order_id = str() # exchange order id. (not used in this example)
        self.symbol = symbol
        self.price = price
        self.qty = qty
        self.side = side
        self.time = time
        self.filled_qty = filled_qty
        self.order_id = order_id
        self.status = "pending" # pending, open, filled, partial, rejected

        self.place_limit_order() 
    
    def place_limit_order():
        pass

    def cancel_order():
        pass

    def __lte__(self, order_):
        if order_.price > self.price:
            return True
        
        ## need to check for corner cases
        elif (order_.price == self.price):
            if order_.status == "pending":
                return False
            
            elif order_.status == "partial":
                return False
            
            return True

        return False
    
    def __gte__(self, order_):
        if order_.price < self.price:
            return True
        
        ## need to check for corner cases
        elif (order_.price == self.price):
            if order_.status == "pending":
                return True
            
            elif order_.status == "partial":
                return True
            
            return False

        return False
        

class market_maker:
    
    prev_theo_price = None
    open_bid_order_list = []
    open_ask_order_list = []
    executed_bid_order_list = []
    executed_ask_order_list = []

    def __init__(self, symbol, n_orders, offset, step_size, qty=10):
        ## USER DEFINED INPUTS
        self.sym = symbol
        self.n_orders = n_orders
        self.offset = offset
        self.step_size = step_size
        self.qty = qty

    def on_market_data(self, best_bid, best_ask):
        self.theo_price = (best_bid + best_ask) / 2
        
        self.update_all_orders_status()
        self.check_filled_orders()
        self.send_orders()
    
    def get_exchange_ack(self, order):
        ## get "ack" from the exchange. 
        # order.status = self.get_exchange_ack(order.exchange_order_id) ->  in case of exchange api. 
        order.status = "open"

    def update_all_orders_status(self):
        max_len = max(len(self.open_bid_order_list), len(self.open_ask_order_list))
        for i in range(max_len):
            if i < len(self.open_ask_order_list) and self.open_ask_order_list[i].status != "filled":
                self.get_exchange_ack(self.open_ask_order_list[i])

            if i < len(self.open_bid_order_list) and self.open_bid_order_list[i].status != "filled":
                self.get_exchange_ack(self.open_bid_order_list[i])

    def check_filled_orders(self):
        ## check for filled orders.
        
        max_len = max(len(self.open_bid_order_list), len(self.open_ask_order_list))
        order_pop_list = []
        for i in range(max_len):
            if i < len(self.open_ask_order_list) and self.open_ask_order_list[i].status == "filled":
                self.executed_ask_order_list.append(self.open_ask_order_list[i])
                order_pop_list.append((i, "ask"))

            if i < len(self.open_bid_order_list) and self.open_bid_order_list[i].status == "filled":
                self.executed_bid_order_list.append(self.open_bid_order_list[i])
                order_pop_list.append((i, "bid"))

            if (i < len(self.open_bid_order_list) and self.open_bid_order_list[i].status == "open") and \
                (i < len(self.open_ask_order_list) and self.open_ask_order_list[i].status == "open"):
                # since the list is sorted, it means no further orders are executed.
                break
        
        ## reversing the list to pop the orders in the correct order.
        order_pop_list.sort(key=lambda x: x[0], reverse=True)
        for indx, side in order_pop_list:
            if side == "ask":
                self.open_ask_order_list.pop(indx)
            else:
                self.open_bid_order_list.pop(indx)

    def modify_orders(self, preference, theo_price_diff):
        
        ## preference = "bid" or "ask". In case theo price is closer to bid, we would want ot change the bid order first.
        ## If theo price change is same as the offset, we would only need to change top orders. 

        if (abs(theo_price_diff) % self.offset) == 0:
            num_depth_to_change = int(abs(theo_price_diff) / self.offset)

        

    def send_orders(self):
        if self.prev_theo_price is None:
            self.prev_theo_price = self.theo_price
            
            ## Creating inital orders
            for i in range(self.n_orders):
                order_id_ask = uuid.uuid4()
                order_id_bid = uuid.uuid4()

                ## placing limit orders
                order_ask = create_order(self.symbol, self.theo_price + self.offset + i*self.step_size, self.qty, "ask", dt.datetime.now(), 0, order_id_ask)
                order_bid = create_order(self.symbol, self.theo_price - self.offset - i*self.step_size, self.qty, "bid", dt.datetime.now(), 0, order_id_bid)
                insort(self.open_ask_order_list, order_ask)
                insort(self.open_bid_order_list, order_bid, reverse=True)
        
        elif self.theo_price > self.prev_theo_price:
            self.modify_orders("ask", self.theo_price - self.prev_theo_price)
            self.prev_theo_price = self.theo_price
        

        elif self.theo_price < self.prev_theo_price:
            self.modify_orders("bid", self.theo_price - self.prev_theo_price)
            self.prev_theo_price = self.theo_price
        
        else:
            print("NO CHANGE IN THEO PRICE")
            ## just check for any executed orders and place them again. 
            max_len = max(len(self.open_bid_order_list), len(self.open_ask_order_list))
            for i in range(max_len-1, 0, -1):
                if i < len(self.open_ask_order_list) and self.open_ask_order_list[i].status == "rejected":
                    id_ = uuid.uuid4()
                    order_ask = create_order(self.symbol, self.open_ask_order_list[i].price, self.qty, "ask", dt.datetime.now(), 0, id_)
                    self.open_ask_order_list.pop(i)
                    insort(self.open_ask_order_list, order_ask)


                if i < len(self.open_bid_order_list) and self.open_bid_order_list[i].status == "rejected":
                    id_ = uuid.uuid4()
                    order_bid = create_order(self.symbol, self.open_bid_order_list[i].price, self.qty, "bid", dt.datetime.now(), 0, id_)
                    self.open_bid_order_list.pop(i)
                    insort(self.open_bid_order_list, order_bid, reverse=True)

