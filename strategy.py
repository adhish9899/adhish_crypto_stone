
import uuid
import datetime as dt

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
    
    def place_limit_order(self):
        pass

    def cancel_order(self):
        # self.send_cancel_order_request(self.exchange_order_id)
        # self.status = self.get_exchange_ack(self..exchange_order_id)
        self.status = "cancelled"

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
    filled_bid_order_list = []
    filled_ask_order_list = []
    pending_close_order_list = []


    def __init__(self, symbol, n_orders, offset, step_size, qty=10):
        ## USER DEFINED INPUTS
        self.symbol = symbol
        self.n_orders = n_orders
        self.offset = offset
        self.step_size = step_size
        self.qty = qty

    def on_market_data(self, best_bid, best_ask):
        self.theo_price = (best_bid + best_ask) / 2
        
        self.close_pending_cancel_orders()
        self.update_all_orders_status()
        self.check_filled_orders()
        self.send_orders()

    def close_pending_cancel_orders(self):
        for i in range(len(self.pending_close_order_list)-1, 0, -1):
            self.get_exchange_ack(self.pending_close_order_list[i])
            if self.pending_close_order_list[i].status == "open":
                self.pending_close_order_list[i].cancel_order()
            
            elif self.pending_close_order_list[i].status == "rejected":
                pass

            elif self.pending_close_order_list[i].status == "partial":
                self.on_order_part_exec(self.pending_close_order_list[i])
                if self.pending_close_order_list[i].side == "bid":
                    self.open_bid_order_list.pop(1)
                
                else:
                    self.open_ask_order_list.pop(1)
            
            elif self.pending_close_order_list[i].status == "filled":
                if self.pending_close_order_list[i].side == "bid":
                    self.filled_bid_order_list.append(self.pending_close_order_list[i])
                
                else:
                    self.filled_ask_order_list.append(self.pending_close_order_list[i])
            
            else: # cancelled. Nothing to do. 
                pass
            
            self.pending_close_order_list.pop(i)
    
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
                self.filled_ask_order_list.append(self.open_ask_order_list[i])
                order_pop_list.append((i, "ask"))

            if i < len(self.open_bid_order_list) and self.open_bid_order_list[i].status == "filled":
                self.filled_bid_order_list.append(self.open_bid_order_list[i])
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


    def on_order_rejected(self, order, i):
        ## recreate the same order
        id_ = uuid.uuid4()
        order_ = create_order(self.symbol, self.open_ask_order_list[i].price, self.qty, order.side, dt.datetime.now(), 0, id_)
        if order.side == "bid":
            self.open_ask_order_list.pop(i)
            insort(self.open_ask_order_list, order_)

        else:
            self.open_bid_order_list.pop(i)
            insort(self.open_bid_order_list, order_, reverse=True)

    def on_order_part_exec(self, part_order):
        part_order.cancel_order()
        if part_order.side == "bid":
            remaining_qty = self.qty - part_order.filled_qty
            self.filled_bid_order_list.append(self.open_bid_order_list.pop(0))
            order_ = create_order(self.symbol, part_order.price, remaining_qty, "bid", dt.datetime.now(), 0, uuid.uuid4())
            self.open_bid_order_list.insert(0, order_)

        else:
            remaining_qty = self.qty - part_order.filled_qty
            self.filled_ask_order_list.append(self.open_ask_order_list.pop(0))
            order_ = create_order(self.symbol, part_order.price, remaining_qty, "ask", dt.datetime.now(), 0, uuid.uuid4())
            self.open_ask_order_list.insert(0, order_)

    # def print_all_orders(self):
    #     print("open bid order list: ", [x.price for x in self.open_bid_order_list])
    #     print("open ask order list: ", [x.price for x in self.open_ask_order_list])
    #     print("filled bid order list: ", [(x.price, x.filled_qty) for x in self.filled_bid_order_list])
    #     print("filled ask order list: ", [(x.price, x.filled_qty) for x in self.filled_bid_order_list])
        # print("pending close order list: ", self.pending_close_order_list)


    

    def send_orders(self):
        ## Creating initial orders
        if self.prev_theo_price is None:
            self.prev_theo_price = self.theo_price
            
            ## Creating inital orders
            for i in range(self.n_orders):
                order_id_ask = uuid.uuid4()
                order_id_bid = uuid.uuid4()

                ## placing limit orders
                order_ask = create_order(self.symbol, self.theo_price + self.offset + i*self.step_size, self.qty, "ask", dt.datetime.now(), 0, order_id_ask)
                order_bid = create_order(self.symbol, self.theo_price - self.offset - i*self.step_size, self.qty, "bid", dt.datetime.now(), 0, order_id_bid)
                self.open_ask_order_list.insert(i, order_ask)
                self.open_bid_order_list.insert(i, order_bid)
        
        else:
            theo_change = self.theo_price - self.prev_theo_price

            # check for partial filled orders
            ## Assuming only one partial order at a time. (No latency or other exchange related issues)
            partial_ask_order = [self.open_ask_order_list] if self.open_ask_order_list[0].status == "partial" else []
            partial_bid_order = [self.open_bid_order_list] if self.open_bid_order_list[0].status == "partial" else []

            if (abs(theo_change) % self.offset) == 0:
                open_ask_price_list = [x.price for x in self.open_ask_order_list if x.status in ["open", "pending"]]
                open_bid_price_list = [x.price for x in self.open_bid_order_list if x.status in ["open", "pending"]]

            for i in range(self.n_orders):
                new_ask_price = self.theo_price + self.offset + i*self.step_size
                new_bid_price = self.theo_price - self.offset - i*self.step_size

                if (abs(theo_change) % self.offset) == 0:
                    loc_order_bid = insort(open_bid_price_list, new_bid_price, reverse=True, insert=False)
                    
                    # Handling partial fill
                    ## If theo price has not changed, then we will not make any changes to the partial order
                    if (i == 0) and partial_bid_order and partial_bid_order[0].price != new_bid_price:
                        self.on_order_part_exec("bid", partial_bid_order[0], new_bid_price)

                    elif loc_order_bid >= 0: # We have an order with the same price in the orderbook
                        if self.open_bid_order_list[loc_order_bid].status == "rejected":
                            self.on_order_rejected(self.open_bid_order_list[loc_order_bid], loc_order_bid)
    
                        # Re-ordering the ladder. 
                        self.open_bid_order_list[i], self.open_bid_order_list[loc_order_bid] = self.open_bid_order_list[loc_order_bid], self.open_bid_order_list[i]
                    
                    elif i < len(self.open_bid_order_list):
                        if self.open_bid_order_list[i].status == "open":
                            self.open_bid_order_list[i].cancel_order()
                        
                        elif self.open_bid_order_list[i].status == "pending":
                            self.pending_close_order_list.append(self.open_bid_order_list[i])
                        
                        self.open_bid_order_list.pop(i)

                        order_id_bid = uuid.uuid4()
                        order_bid = create_order(self.symbol, self.theo_price - self.offset - i*self.step_size, self.qty, "bid", dt.datetime.now(), 0, order_id_bid)
                        insort(self.open_bid_order_list, order_bid, reverse=True)

                    else: # Some orders were filled and we need to create new ones to maintain ladder size. 
                        order_id_bid = uuid.uuid4()
                        order_bid = create_order(self.symbol, self.theo_price - self.offset - i*self.step_size, self.qty, "bid", dt.datetime.now(), 0, order_id_bid)
                        insort(self.open_bid_order_list, order_bid, reverse=True)
                    
                    ## Ask orders
                    loc_order_ask = insort(open_ask_price_list, new_ask_price, reverse=False, insert=False)
                    if (i == 0) and partial_ask_order and partial_ask_order[0].price != new_ask_price:
                        self.on_order_part_exec("ask", partial_ask_order[0], new_ask_price)

                    elif loc_order_ask >= 0:
                        if self.open_ask_order_list[loc_order_ask].status == "rejected":
                            self.on_order_rejected(self.open_ask_order_list[loc_order_ask], loc_order_ask)
    
                        else:
                            self.open_ask_order_list[i], self.open_ask_order_list[loc_order_ask] = self.open_ask_order_list[loc_order_ask], self.open_ask_order_list[i]
                    
                    elif i < len(self.open_ask_order_list):
                        if self.open_ask_order_list[i].status == "open":
                            self.open_ask_order_list[i].cancel_order()
                        
                        elif self.open_ask_order_list[i].status == "pending":
                            self.pending_close_order_list.append(self.open_ask_order_list[i])
                        
                        self.open_ask_order_list.pop(i)

                        order_id_ask = uuid.uuid4()
                        order_ask = create_order(self.symbol, self.theo_price - self.offset - i*self.step_size, self.qty, "ask", dt.datetime.now(), 0, order_id_ask)
                        insort(self.open_ask_order_list, order_ask)

                    else:
                        order_id_ask = uuid.uuid4()
                        order_ask = create_order(self.symbol, self.theo_price - self.offset - i*self.step_size, self.qty, "ask", dt.datetime.now(), 0, order_id_ask)
                        insort(self.open_ask_order_list, order_ask)
            

                else:
                    '''
                    Since theo_change is not a multiple of offset, we need to cancel all orders and create new ones.
                    Although this approach has several pitfalls, it is the simplest approach. 
                    But given the objective of the assignment, I will stick to this approach.

                    Following are the major pitfalls:
                        1. It will require putting new orders and cancelling old unfilled orders, which may violate stock exchange rules.
                        2. It is no efficient as it will put us in last position to take fill at a particular price.                     
                    '''
                    
                    if (i == 0) and theo_change > 0:
                        if (i == 0) and partial_ask_order and partial_ask_order[0].price != new_ask_price:
                            self.on_order_part_exec("ask", partial_ask_order[0], new_ask_price)
                        
                        else:
                            if self.open_ask_order_list[i].status == "pending":
                                self.pending_close_order_list.append(self.open_ask_order_list[i])

                            else:
                                self.open_ask_order_list[i].cancel_order()

                            self.open_ask_order_list.pop(i)
                            order_id_ask = uuid.uuid4()
                            order_ask = create_order(self.symbol, self.theo_price - self.offset - i*self.step_size, self.qty, "ask", dt.datetime.now(), 0, order_id_ask)
                            self.open_ask_order_list.insert(0, order_ask)

                    
                    elif (i == 0) and theo_change < 0:
                        if partial_bid_order and partial_bid_order[0].price != new_bid_price:
                            self.on_order_part_exec("bid", partial_bid_order[0], new_bid_price)
                        
                        else:
                            if self.open_bid_order_list[i].status == "pending":
                                self.pending_close_order_list.append(self.open_bid_order_list[i])

                            else:
                                self.open_bid_order_list[i].cancel_order()

                            self.open_bid_order_list.pop(i)
                            order_id_bid = uuid.uuid4()
                            order_bid = create_order(self.symbol, self.theo_price - self.offset - i*self.step_size, self.qty, "bid", dt.datetime.now(), 0, order_id_bid)
                            self.open_bid_order_list.insert(0,order_bid)
                    
                    else:
                        if self.open_bid_order_list[i].status == "pending":
                            self.pending_close_order_list.append(self.open_bid_order_list[i])

                        else:
                            self.open_bid_order_list[i].cancel_order()


                        if self.open_ask_order_list[i].status == "pending":
                            self.pending_close_order_list.append(self.open_ask_order_list[i])

                        else:
                            self.open_ask_order_list[i].cancel_order()

                        self.open_bid_order_list.pop(i)
                        self.open_ask_order_list.pop(i)

                        order_id_ask = uuid.uuid4()
                        order_ask = create_order(self.symbol, self.theo_price - self.offset - i*self.step_size, self.qty, "ask", dt.datetime.now(), 0, order_id_ask)
                        self.open_ask_order_list.insert(0, order_ask)

                        self.open_ask_order_list.insert(i, order_ask)

                        order_id_bid = uuid.uuid4()
                        order_bid = create_order(self.symbol, self.theo_price - self.offset - i*self.step_size, self.qty, "bid", dt.datetime.now(), 0, order_id_bid)
                        self.open_bid_order_list.insert(i,order_bid)




                        

if __name__ == "__main__":
    mm_trading = market_maker("BTCUSDT", 5, 1, 2, 2)
    mm_trading.on_market_data(99, 101)
    mm_trading.print_all_orders()




            