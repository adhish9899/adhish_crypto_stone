
import uuid
import datetime as dt
import copy
import pandas as pd

## USING IT FROM "bisect" MODULE and updating it to add reverse as well and get index as well.  
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
    def __init__(self, symbol, price, qty, side, time, order_id):
        self.exchange_order_id = str() # exchange order id. (not used in this example)
        self.symbol = symbol
        self.price = price
        self.qty = qty
        self.side = side
        self.time = time
        self.filled_qty = 0
        self.order_id = order_id
        self.status = "pending" # pending, open, filled, partial, rejected

        self.place_limit_order() 
    
    def place_limit_order(self):
        pass

    def __str__(self):
        return "symbol: {}, price: {}, qty: {}, side: {}, time: {}, filled_qty: {}, order_id: {}, status: {}".format(self.symbol, self.price, self.qty, self.side, self.time, self.filled_qty, self.order_id, self.status)

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
    filled_order_list = []

    def __init__(self, symbol, n_orders, offset, step_size, qty=10):
        ## USER DEFINED INPUTS
        self.symbol = symbol
        self.n_orders = n_orders
        self.offset = offset
        self.step_size = step_size
        self.qty = qty        
    
    def get_exchange_ack(self, order):
        ## get "ack" from the exchange. 
        # order.status = self.get_exchange_ack(order.exchange_order_id) ->  in case of exchange api. 
        order.status = "open"

    def create_intial_ladder_orders(self):
        ## Creating inital orders
        for i in range(self.n_orders):
            order_id_ask = uuid.uuid4()
            order_id_bid = uuid.uuid4()

            ## placing limit orders
            order_ask = create_order(self.symbol, self.theo_price + self.offset + i*self.step_size, self.qty, "ask", dt.datetime.now(), order_id_ask)
            order_bid = create_order(self.symbol, self.theo_price - self.offset - i*self.step_size, self.qty, "bid", dt.datetime.now(), order_id_bid)
            self.open_ask_order_list.insert(i, order_ask)
            self.open_bid_order_list.insert(i, order_bid)


    def on_market_data(self, best_bid, best_ask):
        self.theo_price = (best_bid + best_ask) / 2

        ## Creating initial orders
        if self.prev_theo_price is None:
            self.prev_theo_price = self.theo_price
            self.create_intial_ladder_orders()
            ## Just to close the function. Not needed. Can be removed as well. 
            return None
              
        ## Checking for new orders
        theo_change = self.theo_price - self.prev_theo_price

        # check for partial filled orders
        ## Assuming only one partial order at a time. (No latency or other exchange related issues)
        self.partial_ask_order = [self.open_ask_order_list] if self.open_ask_order_list[0].status == "partial" else []
        self.partial_bid_order = [self.open_bid_order_list] if self.open_bid_order_list[0].status == "partial" else []
        
        open_ask_price_list = []
        open_bid_price_list = []
        if (abs(theo_change) % self.offset) == 0:
            open_ask_price_list = [x.price for x in self.open_ask_order_list if x.status in ["open", "pending"]]
            open_bid_price_list = [x.price for x in self.open_bid_order_list if x.status in ["open", "pending"]]

        for i in range(self.n_orders):
            new_ask_price = self.theo_price + self.offset + i*self.step_size
            new_bid_price = self.theo_price - self.offset - i*self.step_size
            
            # adjusting bid orders first. 
            loc_order_bid = insort(open_bid_price_list, new_bid_price, reverse=False, insert=False)
            loc_order_ask = insort(open_ask_price_list, new_ask_price, reverse=True, insert=False)

            self.get_exchange_ack(self.open_bid_order_list[i])
            self.get_exchange_ack(self.open_ask_order_list[i])

            if loc_order_bid >= 0: # there exist an order with similar price
                self.open_bid_order_list[i], self.open_bid_order_list[loc_order_bid] = self.open_bid_order_list[loc_order_bid], self.open_bid_order_list[i]

            if loc_order_ask >= 0:
                self.open_ask_order_list[i], self.open_ask_order_list[loc_order_ask] = self.open_ask_order_list[loc_order_ask], self.open_ask_order_list[i]

            if theo_change <= 0: # Handling bid order first
                self.open_bid_order_list[i] = self.handle_order(self.open_bid_order_list[i], new_bid_price)                
                self.open_ask_order_list[i] = self.handle_order(self.open_ask_order_list[i], new_ask_price)
            
            else:
                self.open_ask_order_list[i] = self.handle_order(self.open_ask_order_list[i], new_ask_price)
                self.open_bid_order_list[i] = self.handle_order(self.open_bid_order_list[i], new_bid_price)                

        ##    
        self.prev_theo_price = self.theo_price


    
    def on_cancel_order(self, order, qty=None, price=None):
        # self.xcchn.cancel_order(order) -> send "cancel" message to exchage
        qty = self.qty if qty is None else qty
        price = order.price if price is None else price
        
        return create_order(self.symbol, price, qty, order.side, dt.datetime.now(), order.order_id)

    def on_rejected_order(self, order, price):
        return create_order(self.symbol, price, self.qty, order.side, dt.datetime.now(), uuid.uuid4())

    ##
    def on_order_exec(self, order, price):
        self.filled_order_list(copy.deepcopy(order))
        return create_order(self.symbol, price, self.qty, order.side, dt.datetime.now(), uuid.uuid4())


    def on_partial_exec(self, order, price):
        
        if order.status == "filled":
            if order.side == "ask":
                order = copy.deepcopy(self.partial_ask_order[0])
            
            elif order.side == "bid":
                order = copy.deepcopy(self.partial_bid_order[0])
            
        elif order.status == "partial": # order status 
            if abs(order.price - price)/self.step_size < 2: # If partial order is just two step from theo, dont do anything.
                ## check for the ordering in the orderbook. (to do)
                return None
            
            else:
                remaining_qty = order.qty - order.filled_qty
                order = self.on_cancel_order(order, remaining_qty)

        else:
            remaining_qty = order.qty - order.filled_qty
            order = create_order(self.symbol, price, remaining_qty, order.side, dt.datetime.now(), uuid.uuid4())

        if order.side == "ask":
            self.partial_ask_order = []
        
        elif order.side == "bid":
            self.partial_bid_order = []
        
        return order
            
    ###
    def handle_order(self, order, order_price):

        if order.status == "pending":
            pass

        elif order.side == "ask" and self.partial_ask_order:
            order = self.on_partial_exec(order, order_price)
        
        elif order.side == "bid" and self.partial_bid_order:
            order = self.on_partial_exec(order, order_price)
                
        elif order.status == "open":
            if order.price != order_price:
                order = self.on_cancel_order(order, price=order_price)
                
        elif order.status in ["rejected"]:
            order = self.on_rejected_order(order, order_price)

        elif order.status == "filled":
            order = self.on_order_exec(order, order_price)
        
        else:
            raise Exception("Invalid order status %s" % order.status)
        
        return order

    def get_stats(self, best_bid, best_ask):
        
        '''
        1. Realised Profit (Based on Fifo)
        2. Unrealised Profit (Based on Fifo)
        3. Total traded quantity/value
        4. Pending orders quantity/value
        '''
        all_buy_trades = [[x.price, x.filled_qty, x.time] for x in self.filled_order_list if x.side == "bid"]
        all_sell_trades = [[x.price, x.filled_qty, x.time] for x in self.filled_order_list if x.side == "ask"]

        all_buy_trades.sort(key=lambda x: x[2])
        all_sell_trades.sort(key=lambda x: x[2])

        # check for partial fill
        if self.open_ask_order_list[0].status == "partial":
            all_buy_trades.append((self.open_ask_order_list[0].price, self.open_ask_order_list[0].filled_qty, self.open_ask_order_list[0].time))

        if self.open_bid_order_list[0].status == "partial":
            all_sell_trades.append((self.open_bid_order_list[0].price, self.open_bid_order_list[0].filled_qty, self.open_bid_order_list[0].time))
        
        total_buy_qty_traded = sum([x[1] for x in all_buy_trades])
        total_sell_qty_traded = sum([x[1] for x in all_sell_trades])

        total_buy_value_traded = sum([x[0]*x[1] for x in all_buy_trades])
        total_sell_value_traded = sum([x[0]*x[1] for x in all_sell_trades])

        total_pending_buy_qty = sum([x.qty for x in self.open_bid_order_list if x.status == "open"])
        total_pending_sell_qty = sum([x.qty for x in self.open_ask_order_list if x.status == "open"])

        total_pending_buy_value = sum([x.price*x.qty for x in self.open_bid_order_list if x.status == "open"])
        total_pending_sell_value = sum([x.price*x.qty for x in self.open_ask_order_list if x.status == "open"])

        min_qty_traded = min(total_buy_qty_traded, total_sell_qty_traded)
        if min_qty_traded == 0:
            if len(total_buy_qty_traded) == 0:
                vwap_sell_price = sum(total_sell_value_traded)/sum(total_sell_qty_traded)
                unrealised_pnl = (vwap_sell_price - best_ask) * total_sell_qty_traded
            
            elif len(total_sell_qty_traded) == 0:
                vwap_buy_price = sum(total_buy_value_traded)/sum(total_buy_qty_traded)
                unrealised_pnl = (best_bid - vwap_buy_price) * total_buy_qty_traded
            
            else:
                unrealised_pnl = 0

            realised_pnl = 0
        
        ##
        else:
            vway_buy_min_qty, vway_buy_remain = self.get_notional_vwap(all_buy_trades, min_qty_traded)
            vway_sell_min_qty, vway_sell_remain = self.get_notional_vwap(all_sell_trades, min_qty_traded)

            realised_pnl = (vway_sell_min_qty - vway_buy_min_qty) * min_qty_traded
            if vway_buy_remain is None:
                unrealised_pnl = (vway_sell_min_qty - best_bid) * min_qty_traded
            
            elif vway_sell_remain is None:
                unrealised_pnl = (best_ask - vway_buy_min_qty) * min_qty_traded
            
            else:
                unrealised_pnl = 0
        
        return {"symbol": self.symbol, "realised_pnl": realised_pnl, "unrealised_pnl": unrealised_pnl, 
                "total_buy_qty_traded": total_buy_qty_traded, "total_sell_qty_traded": total_sell_qty_traded, 
                "total_buy_value_traded": total_buy_value_traded, "total_sell_value_traded": total_sell_value_traded,
                "total_pending_buy_qty": total_pending_buy_qty, "total_pending_sell_qty": total_pending_sell_qty,
                "total_pending_buy_value": total_pending_buy_value, "total_pending_sell_value": total_pending_sell_value}

    
    @staticmethod
    def get_notional_vwap(trade_ls, min_qty_traded):
        qty_ = 0
        notional = 0
        remaining_vwap = None
        for i in range(len(trade_ls)):
            price = trade_ls[i][0]
            qty = trade_ls[i][1]
            if (qty_ + qty) >= min_qty_traded:
                notional += (min_qty_traded - qty_) * price
                
                ## remaining qty vwap price
                remaining_qty = qty - (min_qty_traded - qty_)
                trade_ls[i][1] = remaining_qty
                remaining_notional = (trade_ls[j][0]*trade_ls[j][1] for j in range(i, len(trade_ls)))
                if sum(remaining_notional):
                    remaining_vwap = sum(remaining_notional)/sum([x[1] for x in trade_ls[i:]])

                break
            else:
                notional += qty * price
                qty_ += qty
        
        return notional/min_qty_traded, remaining_vwap

            


if __name__ == "__main__":
    start_time = dt.datetime.now().replace(microsecond=0)
    print("Start time:", start_time)

    # sample way to run it. 


    '''
    mm_trading = market_maker("BTCUSDT", 5, 1, 2, 2)
    stats_df = pd.DataFrame()
    for data in data_generator:
        mm_trading.on_market_data(data.bid, data.ask)

        if dt.datetime.now() - start_time > dt.timedelta(seconds=60):
            start_time = dt.datetime.now()
            stats_df = stats_df.append(pd.DataFrame(mm_trading.get_stats(), index=["stats"].T))
    '''
    
