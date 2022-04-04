
import numpy as np

class trade_class(object):

    def __init__(self, stock1, stock2, stock1_price, stock2_price, entry_date, target=None,regression_coefs=None,
                target_operator=None, base_date=None, spread_mean=None, spread_std=None,  entry_spread=None):

        '''
        stock1 = x
        stock2 = y

        PARAMETERS:
        -----------------------------
        1. stock1: str, E.g: HDFCBANK
            Name of stock in leg1 of pair.
        
        2. stock2: str. E.g: KOTAKBANK
            Name of stock in leg2 of pair. 

        3. stock1_price: float between -np.inf to np.inf
            Stock price of the stock in leg1 of pair. If stock price < 0; it indicates the short position on particular stock/leg 
            and vice-versa.

        4. stock2_price: float between -np.inf to np.inf
            Stock price of the stock in leg2 of pair. If stock price < 0; it indicates the short position on particular stock/leg
            and vice-versa.
        
        5.  regression_coefs: output of st.linregress
            Output of the regression between stock1(x) and stock2(y)
        
        6. mean_reversion_days: int
            Avg number days it takes spread to revert to mean. 
        
        7. entry_date: dt.date() object. E.g: dt.date(2020, 1, 1)
            The date at which the trade was taken
        
        8. target: float
            Target spread at which the position will be squared off
        
        9. base_date: dt.date() object. E.g: dt.date(2020, 1, 1)
            Base date prices used to normalise stock prices for regression and calculate spread.

        10. spread_mean: float
            Mean of the insample spread

        11. spread_std: float
            standard deviation of the insample spread

        12. entry_spread: float
            Spread at which the trade was taken
        '''

        self.stock1 = stock1
        self.stock2 = stock2
        self.stock1_price = stock1_price
        self.stock2_price = stock2_price
        self.entry_date = entry_date
        self.exit_date = None
        self.base_date = base_date
        self.target = target
        self.target_operator = target_operator # "lte", "gte"

        self.stock1_exit_price = np.nan
        self.stock2_exit_price = np.nan

        if target_operator not in ["lte", "gte"]:
            raise Exception("Invalid target operator %s" %(target_operator))

        self.stock2_beta = 1
        self.stock1_beta = regression_coefs.slope
        self.constant = regression_coefs.intercept
        self.r_square = regression_coefs.rvalue ** 2

        self.m2m = -np.inf
        self.max_m2m = -np.inf
        self.spread_max_m2m = np.nan

        self.spread_mean = spread_mean
        self.spread_std = spread_std

        self.ticks_passed = 0

        self.target_hit = False
        self.stoploss_hit = False
        self.stats_dict = {}
        
        self.entry_spread = entry_spread
        self.exit_spread = np.nan


    def update_m2m(self, current_m2m):
        self.m2m = current_m2m
        self.max_m2m = max(self.m2m, self.max_m2m)


class trade_handler(object):

    def __init__(self):
        self.stock_trade_count = {}
        self.pair_trade_count = {}
        self.open_trade_list = [] # It will contain all the open positions 
        self.trade_history_dict = {}
    
    @staticmethod
    def get_long_stock(trade_object):
        
        '''
        trade_object: instance of "trade" class
        '''

        if trade_object.stock1_price > 0:
            return trade_object.stock1
        
        return trade_object.stock2


    @staticmethod
    def get_short_stock(trade_object):
        '''
        trade_object: instance of "trade" class
        '''

        if trade_object.stock1_price < 0:
            return trade_object.stock1
        
        return trade_object.stock2
    
    @staticmethod
    def get_pair_name(trade_object):
        '''
        trade_object: instance of "trade" class
        '''

        pair_name = sorted([trade_object.stock1, trade_object.stock2])
        return tuple(pair_name)

    def new_trade(self, trade_object):
        '''
        trade_object: instance of "trade" class
        '''

        # if not isinstance(trade_object, trade_class):
        #     raise TypeError("trade_object is not of 'trade' class")
        
        pair_name = self.get_pair_name(trade_object)
        long_stock_name = self.get_long_stock(trade_object)
        short_stock_name = self.get_short_stock(trade_object)

        self.pair_trade_count[pair_name] = self.pair_trade_count.get(pair_name, 0) + 1
        self.stock_trade_count[long_stock_name] = self.stock_trade_count.get(long_stock_name, 0) + 1
        self.stock_trade_count[short_stock_name] = self.stock_trade_count.get(short_stock_name, 0) - 1
        self.open_trade_list.append(trade_object)

