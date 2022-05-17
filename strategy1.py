
import uuid
import datetime as dt
import copy

'''
1. To Do. (lte, gte)

'''

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
        self.filled_qty = filled_qty
        self.order_id = order_id
        self.status = "pending" # pending, open, filled, partial, rejected

        self.place_limit_order() 
    
    def place_limit_order(self):
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
            order_ask = create_order(self.symbol, self.theo_price + self.offset + i*self.step_size, self.qty, "ask", dt.datetime.now(), 0, order_id_ask)
            order_bid = create_order(self.symbol, self.theo_price - self.offset - i*self.step_size, self.qty, "bid", dt.datetime.now(), 0, order_id_bid)
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
            if theo_change <= 0:
                loc_order_bid = insort(open_bid_price_list, new_bid_price, reverse=True, insert=False)
                self.get_exchange_ack(self.open_bid_order_list[i])

                if loc_order_bid >= 0: # there exist an order with similar price
                    self.open_bid_order_list[i], self.open_bid_order_list[loc_order_bid] = self.open_bid_order_list[loc_order_bid], self.open_bid_order_list[i]

                ###
                self.handle_order(self.open_bid_order_list[i], new_bid_price)
                
            else:
                loc_order_ask = insort(open_ask_price_list, new_ask_price, reverse=False, insert=False)
                self.get_exchange_ack(self.open_ask_order_list[i])

                if loc_order_ask >= 0: # there exist an order with similar price
                    self.open_ask_order_list[i], self.open_ask_order_list[loc_order_ask] = self.open_ask_order_list[loc_order_ask], self.open_ask_order_list[i]
                
                self.handle_order(self.open_ask_order_list[i], new_ask_price)

    

    def on_cancel_order(self, order, qty=None, price=None):
        # self.xcchn.cancel_order(order)
        qty = self.qty if qty is None else qty
        price = order.price if price is None else price
        
        return create_order(self.symbol, price, qty, order.side, dt.datetime.now(), order.order_id)

    def on_rejected_order(self, order, price):
        return create_order(self.symbol, price, self.qty, order.side, dt.datetime.now(), 0, uuid.uuid4())

    ##
    def on_order_exec(self, order, price):
        self.filled_order_list(copy.deepcopy(order))
        return create_order(self.symbol, price, self.qty, order.side, dt.datetime.now(), uuid.uuid4())


    def on_partial_exec(self, order, price):
        
        if order.status == "filled":
            if order.side == "ask":
                order = copy.deepcopy(self.open_ask_order_list[0])
            
            elif order.side == "bid":
                order = copy.deepcopy(self.open_bid_order_list[0])
            
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
            self.open_ask_order_list = []
        
        elif order.side == "bid":
            self.open_bid_order_list = []
            
    ###
    def handle_order(self, order, order_price):
        
        if order.side == "ask" and self.partial_ask_order:
            self.on_partial_order(order, order_price)
        
        elif order.side == "bid" and self.partial_bid_order:
            self.on_partial_order(order, order_price)
        
        if order.status == "pending":
            pass
        
        elif order.status == "open":
            if order.price != order_price:
                order = self.on_cancel_order(order, price=order_price)
                
        elif order.status in ["rejected"]:
            order = self.on_rejected_order(order, order_price)

        elif order.status == "filled":
            order = self.on_order_exec(order, order_price)
        
        else:
            raise Exception("Invalid order status %s" % order.status)

                



                        

if __name__ == "__main__":
    mm_trading = market_maker("BTCUSDT", 5, 1, 2, 2)
    mm_trading.on_market_data(99, 101)
    mm_trading.print_all_orders()



