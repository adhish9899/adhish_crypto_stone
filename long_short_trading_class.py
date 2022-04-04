
import scipy.stats as st
import pandas as pd
import numpy as np
from config_read import config_object
# from pca_backtesting.pca_trading_utils import trade_class, trade_handler
from pca_trading_utils import trade_class, trade_handler
import copy
from statsmodels.tsa.stattools import coint
from class_alphas import alphas
from collections import namedtuple
import math
import datetime as dt

import bottleneck as bn
from methodtools import lru_cache

## TO BE USED IN PROFILING
# import line_profiler
# import atexit

# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats)


'''
1. Fit regression on "lookback" days or whole dataset


'''

ti_dict = {"r_square": [[21]]}
        #    "dist_corr_coeff": [[]],}

ti_dict = {}

class pca_trading_strategy:

    trade_stats = {}
    trade_manager = trade_handler()
    stats_df = pd.DataFrame()
    sq_off_trades_list = []
    
    def __init__(self):
        self.pair_beta_dict = {}

    def initalizer(self, data, timestamp, pair_list_class):

        '''
        PARAMETERS:
        1. data: pd.DataFrame
            All FO stocks historical data. 

        2. timestamp: pd.timestamp
            pd.Timestamp object of (current data)/(last date of data.index[-1]) 

        3. resultsDf: pd.DataFrame
            All quarterly results date dataFrame of all fo stocks
        
        4. pair_list_class: calculation_pairs object from "long_short_trading_utils"
            Used to get pair list of current date.
        '''

        self.data = data
        self.timestamp = timestamp
        data_lookback = max(config_object.pca_trading_lookback, config_object.bbands_lookback + config_object.return_bars + 2)
        self.df = self.data.loc[:timestamp].iloc[-data_lookback:]
        self.df = self.df.dropna(axis=1)

        if not  self.df.shape[1]:
            print("No data to trade", self.timestamp)
        
        self.pair_list_class = pair_list_class

        if config_object.data_structure == "numpy":
            self.data_series_dict = self.df.to_dict("series")

    @staticmethod
    def get_rolling_df(df, window, type_, array_type="pandas"):
        if array_type == "pandas":
            if type_ == "std":
                return df.rolling(window=window).std()
            
            elif type_ == "mean":
                return df.rolling(window=window).mean()
            
            else:
                raise NotImplementedError("type_ must be either 'std' or 'mean'")
        
        elif array_type == "numpy":
            if type_ == "std":
                return bn.move_std(df, window, ddof=1) # ddof = 1 (To match with pandas)
            
            elif type_ == "mean":
                return bn.move_mean(df, window)
            
            else:
                raise NotImplementedError("type_ must be either 'std' or 'mean'")
            
        else:
            raise NotImplementedError("array_type must be either 'pandas' or 'numpy'")


    def get_pair_list(self,date_):
        '''
        PARAMETERS:
        1. date_: dt.date() object

        '''
        assert isinstance(date_, dt.date), "date_ must be a dt.date object"

        self.pair_list_class.get_pair_list(date_)
        self.pair_list = self.pair_list_class.pair_list
    
    @staticmethod
    def normalise_stock_prices(df):
        '''
        1. df: pd.DataFrame
            dataFrame of 2 stocks(pair) containing lookback data.
        '''
        return df/df.iloc[0]
    
    def entry_trade_stats(self, trade_object):
        
        alphas_class = alphas(self.spread, trade_object.stock1, trade_object.stock2, {}, self.df)
        
        pair_dict = {}
        for key, value in ti_dict.items():
            
            if len(value) == 1:

                if len(value[0]) == 0:
                    key_ = key
                
                else:
                    key_ = key + "_" + "_".join([str(x) for x in value[0]])
                
                
                alpha_value = getattr(alphas_class, key)(*value[0])
                pair_dict.setdefault(key_,[]).append(alpha_value)
            
            else:
                for v in value:
                    key_ = key + "_" + "_".join([str(x) for x in v])

                    alpha_value = getattr(alphas_class, key)(*v)
                    pair_dict.setdefault(key_,[]).append(alpha_value)
        
        return pair_dict
    
        
    def entry_rule_check(self):
        '''
        PARAMETERS:
        1. stock1: str, E.g: HDFCBANK
            Name of stock in leg1 of pair.
        
        2. stock2: str. E.g: KOTAKBANK
            Name of stock in leg2 of pair. 
        
        3. regression_coefs: output of st.linregress
            contains slope, intercept, rvalue
        ''' 

        trade_passing_entry_rule = []

        ## USING BETA THRESHOLD TO FILTER TRADE
        if config_object.beta_entry_threshold and self.regression_coefs.slope < config_object.beta_entry_threshold:
            trade_passing_entry_rule.append(0)


        ## Price check
        if (self.df.iloc[-1][self.stock1] < config_object.price_entry_threshold) or (self.df.iloc[-1][self.stock2] < config_object.price_entry_threshold):
            trade_passing_entry_rule.append(0)

        ## To be worked on
        if config_object.spread_type == "normalised_price_regression":
            mean_ = np.mean(self.spread)
            std_ = np.std(self.spread)
            sigma_spread = (self.spread - mean_)/std_
        
        elif config_object.spread_type == "return_bar_regression":
            rolling_std = pca_trading_strategy.get_rolling_df(self.spread, config_object.bbands_lookback, "std", array_type=config_object.data_structure)
            rolling_mean = pca_trading_strategy.get_rolling_df(self.spread, config_object.bbands_lookback, "mean", array_type=config_object.data_structure)

            sigma_spread = [(self.spread[-1] - rolling_mean[-1])/rolling_std[-1]]
        
        else:
            NotImplementedError("Spread type: %s not available" % config_object.spread_type)
        
        try:
            assert (not math.isnan(sigma_spread[-1])), "sigma_spread must be float"
        except AssertionError as e:
            config_object.masterlog.info("sigma_spread must be float, SLOPE is 'nan'")
            return False

        if sigma_spread[-1] > config_object.sigma_filter_upper_thresh:
            trade_passing_entry_rule.append(1)

        elif sigma_spread[-1] < config_object.sigma_filter_lower_thresh:
            trade_passing_entry_rule.append(1)
        
        else:
            trade_passing_entry_rule.append(0)
            
        return all(trade_passing_entry_rule)


    # data_series_dict
    @staticmethod
    def rolling_sum_numpy(a, n):
        ret = np.cumsum(a)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:]


    @lru_cache(1)
    @staticmethod
    def get_named_tuple(tuple_name, *features):
        return namedtuple(tuple_name, features)

    
    def get_numpy_xy_arrays(self):
        x = self.data_series_dict[self.stock1].values
        y = self.data_series_dict[self.stock2].values

        # Calculating pct_change
        x = np.diff(x) / x[:-1]
        y = np.diff(y) / y[:-1]

        x = self.rolling_sum_numpy(x, config_object.return_bars)
        y = self.rolling_sum_numpy(y, config_object.return_bars)

        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]

        x = x[-config_object.bbands_lookback:]
        y = y[-config_object.bbands_lookback:]

        return x,y

    def spread_stats_numpy(self):
        if config_object.spread_type == "normalised_price_regression":
            raise NotImplementedError()
        
        elif config_object.spread_type == "return_bar_regression":

            x,y = self.get_numpy_xy_arrays()
            coefs = self.get_named_tuple("coefs", 'slope', 'intercept', "rvalue") #namedtuple('coefs', ['slope', 'intercept'])   
            slope =  ((x*y).mean() - x.mean()*y.mean()) / ((x**2).mean() - (x.mean())**2)
            intercept = y.mean() - slope*x.mean()

            self.regression_coefs = coefs(slope, intercept, 0)
            self.spread  = (x*self.regression_coefs.slope + self.regression_coefs.intercept) - y
        
        else:
            raise NotImplementedError("Spread type: %s not implemented" % config_object.spread_type)


    def spread_stats_pandas(self):

        '''
        Fitting linear regression between stock normalised prices of stock1 and stock2

        y = B*x + c

        1. x = stock1
        2  y = stock2
        '''
        if config_object.spread_type == "normalised_price_regression":
            regression_data = self.normalise_stock_prices(self.df[[self.stock1, self.stock2]])

            x = regression_data[self.stock1]
            y = regression_data[self.stock2]

            self.regression_coefs = st.linregress(x, y)
            self.spread = (x*self.regression_coefs.slope + self.regression_coefs.intercept) - y
        
        elif config_object.spread_type == "return_bar_regression":
            x = self.df[self.stock1].pct_change().rolling(window=config_object.return_bars).sum().dropna()
            y = self.df[self.stock2].pct_change().rolling(window=config_object.return_bars).sum().dropna()

            x = x.iloc[-config_object.bbands_lookback:]
            y = y.iloc[-config_object.bbands_lookback:]

            self.regression_coefs = st.linregress(x, y)
            self.spread  = (x*self.regression_coefs.slope + self.regression_coefs.intercept) - y

        else:
            raise NotImplementedError("Spread type: %s not implemented" % config_object.spread_type)
    
    def __target_spread(self, current_spread, upper_thresh=True):
                
        if upper_thresh:
            target_spread = current_spread - (config_object.sigma_entry_upper_threshold - config_object.sigma_exit_upper_threshold)
        
        else:
            target_spread = current_spread + (config_object.sigma_exit_lower_threshold - config_object.sigma_entry_lower_threshold)
        
        return target_spread
    
    
    def sigma_entry_params(self):
        '''
        PARAMETERS:
        1. stock1: str, E.g: HDFCBANK
            Name of stock in leg1 of pair.
        
        2. stock2: str. E.g: KOTAKBANK
            Name of stock in leg2 of pair. 
        
        3. regression_coefs: output of st.linregress
            contains slope, intercept, rvalue
        '''

        spread_mean, spread_std = np.mean(self.spread), np.std(self.spread)

        upper_entry_threshold = config_object.sigma_entry_upper_threshold
        lower_entry_threshold = config_object.sigma_entry_lower_threshold
        
        if config_object.spread_type == "normalised_price_regression":
            spread = (self.spread - self.spread.mean())/self.spread.std()
            current_spread = spread[-1]

        elif config_object.spread_type == "return_bar_regression": 
            rolling_std = self.get_rolling_df(self.spread, config_object.bbands_lookback, "std", array_type=config_object.data_structure)
            rolling_mean = self.get_rolling_df(self.spread, config_object.bbands_lookback, "mean", array_type=config_object.data_structure)

            spread = [(self.spread[-1] - rolling_mean[-1])/rolling_std[-1]]
            current_spread = spread[-1]

        if current_spread >= upper_entry_threshold:
            stock1_mult = -1
            stock2_mult = 1
            target_operator = "lte"
            
            target_spread = self.__target_spread(current_spread, upper_thresh=True)
        
        elif current_spread <= lower_entry_threshold:
            stock1_mult = 1
            stock2_mult = -1
            target_operator = "gte"
            
            ## Check
            target_spread = self.__target_spread(current_spread, upper_thresh=False)
        
        else:
            import ipdb; ipdb.set_trace()
            raise Exception("Inavlid entry")
        
        ####
        trade_object = trade_class(stock1=self.stock1, stock2=self.stock2, stock1_price=self.df[self.stock1][-1]*stock1_mult, \
                                stock2_price=self.df[self.stock2][-1]*stock2_mult,  entry_date=self.timestamp, target=target_spread, \
                                regression_coefs=copy.deepcopy(self.regression_coefs), target_operator=target_operator, \
                                base_date=self.df.index[0], spread_mean=spread_mean, spread_std=spread_std, entry_spread=current_spread)

        return trade_object


    def __ignore_pair_list(self):
        
        open_trade_list = pca_trading_strategy.trade_manager.open_trade_list

        ignore_pair = []
        for trade_object in open_trade_list:
            pair_name = list(pca_trading_strategy.trade_manager.get_pair_name(trade_object))
            ignore_pair.append(pair_name)
                
            # else:
            #     if (self.timestamp - trade_object.entry_date)/pd.Timedelta('1 hour') < config_object.gap_same_pair_trades:
            #         ignore_pair.append(pair_name)

        return ignore_pair

    def calculate_pair_spread(self, stock1, stock2):

        self.stock1, self.stock2 = stock1, stock2

        if config_object.data_structure == "pandas":
            self.spread_stats_pandas()

        elif config_object.data_structure == "numpy":
            self.spread_stats_numpy()

        if config_object.recalculate_entry_beta_thresh and self.regression_coefs.slope > config_object.recalculate_entry_beta_thresh:
            self.stock2, self.stock1 = stock1, stock2

            if config_object.data_structure == "pandas":
                self.spread_stats_pandas()
            
            elif config_object.data_structure == "numpy":
                self.spread_stats_numpy()

    # @profile
    def check_entry(self, stock_fo):
        '''
        stock_fo: list
            All the stocks that in FO at t+1 date. 
        '''

        ## Initalizing self.pair_list
        self.get_pair_list(self.timestamp.to_pydatetime())
        
        ## All dates list
        # all_dates_list = [x.date() for x in self.data.index]
        
        ## Ignore stock lists
        # ignore_pairs = [list(name) for name,count in pca_trading_strategy.trade_manager.pair_trade_count.items() if count >= config_object.max_position_pair]
        # ignore_stocks = [name for name,count in pca_trading_strategy.trade_manager.stock_trade_count.items() if abs(count) >= config_object.max_position]

        ignore_pairs =  []
        ignore_stocks = []
        current_pos = self.__ignore_pair_list()
        for stock1, stock2 in self.pair_list:
            
            # if len(self.trade_manager.open_trade_list) > 1:
            #     continue
            
            if stock1 in ignore_stocks or stock2 in ignore_stocks:
                continue

            if [stock1, stock2] in ignore_pairs or [stock2, stock1] in ignore_pairs:
                continue

            if [stock1, stock2] in current_pos or [stock2, stock1] in current_pos:
                # config_object.masterlog.info("ALREADY OPENED %s" % [stock1, stock2])
                continue
            
            if stock1 not in self.df.columns or stock2 not in self.df.columns:
                # config_object.masterlog.info("STOCK NOT IN DATAFRAME %s" % [stock1, stock2])
                continue

            # Checking if stock is in FO
            if (stock1 not in stock_fo) or (stock2 not in stock_fo):
                continue

            if config_object.static_entry_beta:
                key = list(self.pair_beta_dict.keys())[0] if self.pair_beta_dict else None

                if self.timestamp.hour in config_object.static_beta_update_freq and (self.timestamp.floor("h") not in self.pair_beta_dict):
                    self.pair_beta_dict = {}
                    self.pair_beta_dict[self.timestamp.floor("h")] = {}
                    key = self.timestamp.floor("h")

                if (stock1, stock2) in self.pair_beta_dict[key]:
                    self.stock1, self.stock2, self.regression_coefs = self.pair_beta_dict[key][(stock1, stock2)]
                    x,y = self.get_numpy_xy_arrays()
                    self.spread  = (x*self.regression_coefs.slope + self.regression_coefs.intercept) - y

                elif (stock2, stock1) in self.pair_beta_dict[key]:
                    self.stock1, self.stock2, self.regression_coefs = self.pair_beta_dict[key][(stock2, stock1)]
                    x,y = self.get_numpy_xy_arrays()
                    self.spread  = (x*self.regression_coefs.slope + self.regression_coefs.intercept) - y

                else:
                    self.calculate_pair_spread(stock1, stock2)
                    self.pair_beta_dict[key][(self.stock1, self.stock2)] = [self.stock1, self.stock2, self.regression_coefs]

            else:
                self.calculate_pair_spread(stock1, stock2)

            entry_check = self.entry_rule_check()
            
            if entry_check:
                ## We can try different entry functions but only 1 will work at a time
                trade_object = self.sigma_entry_params()
                                                
                ## Entry stats of trades
                trade_object.stats_dict = self.entry_trade_stats(trade_object)

                if config_object.run_type == "stats_only":
                    self.sq_off_trades_list.append(trade_object)

                else:
                    pca_trading_strategy.trade_manager.new_trade(trade_object)
                    # self.stats_df = pd.concat([self.pnl_stats(trade_object, all_dates_list), self.stats_df])
                    current_pos.append([self.stock1, self.stock2])
                
    ####
    @staticmethod
    def calculate_stock_return(entry_price, current_price):
        
        entry_price *= (1 + config_object.brokerage) if entry_price > 0 else (1 - config_object.brokerage)

        return ((current_price - abs(entry_price))/abs(entry_price)) * np.sign(entry_price)
    
    def m2m_positions(self):
        
        for trade_object in self.trade_manager.open_trade_list:

            current_stock1_price = self.data.loc[self.timestamp, trade_object.stock1]
            current_stock2_price = self.data.loc[self.timestamp, trade_object.stock2]
    
            stock1_beta, stock2_beta = trade_object.stock1_beta, trade_object.stock2_beta

            trade_object.stock1_exit_price = current_stock1_price
            trade_object.stock2_exit_price = current_stock2_price

            #####
            stock1_price_return = self.calculate_stock_return(trade_object.stock1_price, current_stock1_price)
            stock2_price_return = self.calculate_stock_return(trade_object.stock2_price, current_stock2_price)
    
            total_cap = stock1_beta + stock2_beta
            
            stock1_return = (stock1_price_return * stock1_beta)/total_cap
            stock2_return = (stock2_price_return* stock2_beta)/total_cap
            
            trade_object.update_m2m(stock1_return + stock2_return)
    
    
    def target_pft_check(self, trade_object, regression_spread):
        
        target = trade_object.target

        if trade_object.target_operator == "gte":
            if regression_spread[-1] >= target:
                trade_object.target_hit = True
        
        elif trade_object.target_operator == "lte":
            if regression_spread[-1] <= target:
                trade_object.target_hit = True
    
    def stop_loss_check(self, trade_object, regression_spread, in_sample_data):
        
        stoploss = (in_sample_data[-1] - trade_object.target) * config_object.rolling_stoploss_mult
        use_rolling_stop = (not config_object.sigma_exit_hard_threshold)
        
        stoploss_spread = stoploss + trade_object.spread_max_m2m if use_rolling_stop else stoploss + in_sample_data[-1]

        if trade_object.target_operator == "gte":
            if regression_spread[-1] <= stoploss_spread:
                trade_object.stoploss_hit = True
        
        elif trade_object.target_operator == "lte":
            if regression_spread[-1] >= stoploss_spread:
                trade_object.stoploss_hit = True
        
        
    # @profile
    def update_trade_object(self, trade_object):
        '''
        Update target profit/stoploss after average mean reversion days. Spread equally to max mean reversion days
        '''

        # Base date to normalise stock prices. (Legacy)
        base_date = trade_object.base_date 

        # Entry date of long short position
        entry_date = trade_object.entry_date

        # Base date to last date spread data
        spread_data = self.data.loc[base_date:self.timestamp][[trade_object.stock1, trade_object.stock2]]

        if config_object.spread_type == "normalised_price_regression":
            spread_norm_data = self.normalise_stock_prices(spread_data)
            regression_spread = spread_norm_data[trade_object.stock1]*trade_object.stock1_beta + trade_object.constant - spread_norm_data[trade_object.stock2] 

            regression_spread = (regression_spread - trade_object.spread_mean)/trade_object.spread_std
            entry_spread = regression_spread[base_date:entry_date]
        
        elif config_object.spread_type == "return_bar_regression":
            if config_object.data_structure == "pandas":
                x = spread_data[trade_object.stock1].pct_change().rolling(window=config_object.return_bars).sum().dropna()
                y = spread_data[trade_object.stock2].pct_change().rolling(window=config_object.return_bars).sum().dropna()
            
            elif config_object.data_structure == "numpy":
                x = self.data_series_dict[trade_object.stock1].values
                y = self.data_series_dict[trade_object.stock2].values

                # Calculating pct_change
                x = np.diff(x) / x[:-1]
                y = np.diff(y) / y[:-1]

                x = self.rolling_sum_numpy(x, config_object.return_bars)
                y = self.rolling_sum_numpy(y, config_object.return_bars)

            ##
            spread = (x*trade_object.stock1_beta + trade_object.constant) - y

            rolling_std = self.get_rolling_df(spread, config_object.bbands_lookback, "std", config_object.data_structure)
            rolling_mean = self.get_rolling_df(spread, config_object.bbands_lookback, "mean", config_object.data_structure)

            regression_spread = (spread - rolling_mean)/rolling_std

            ## ENTRY SPREAD
            entry_spread = [trade_object.entry_spread]

            # in_sample_value = regression_spread.loc[str(trade_object.entry_date)]
            # if isinstance(in_sample_value, np.float):
            #     in_sample_data = [in_sample_value]
            
            # else:
            #     in_sample_data = in_sample_value.tolist()
            
        else:
            raise NotImplementedError("Spread type: %s not available" % config_object.spread_type)
        
        ## Update Max m2m point
        if trade_object.m2m == trade_object.max_m2m:
            trade_object.spread_max_m2m = regression_spread[-1]
        
        ## Updating exit spread
        trade_object.exit_spread = regression_spread[-1]

        ###
        self.target_pft_check(trade_object, regression_spread)
        
        ## stop loss check
        self.stop_loss_check(trade_object, regression_spread, entry_spread)

    
    def sq_off_position(self, coin_active, force_sq_off=False):
                
        ## Update current m2m
        self.m2m_positions()

        self.trade_stats[self.timestamp] = []
        
        ignore_pair = []
        del_trade_obj_indx = []
        m2m_sum = 0
        obj_index = -1
        for obj_index, trade_object in enumerate(self.trade_manager.open_trade_list):

            pair_name = self.trade_manager.get_pair_name(trade_object)
            if list(pair_name) in ignore_pair:
                continue

            trade_object.exit_date = self.timestamp

            ## Update target
            if not force_sq_off:
                self.update_trade_object(trade_object)

            ## Check target pft
            if trade_object.target_hit:
                del_trade_obj_indx.append(obj_index)
                m2m_sum += trade_object.m2m
                continue
            
            ## Check stop loss
            elif trade_object.stoploss_hit:
                del_trade_obj_indx.append(obj_index)
                m2m_sum += trade_object.m2m
                continue
            
            ## not in fo
            elif (trade_object.stock1 not in coin_active):
                del_trade_obj_indx.append(obj_index)
                m2m_sum += trade_object.m2m
                config_object.masterlog.info("%s stock removed from FO %s" %(trade_object.stock1, self.timestamp.date()))
                continue
            
            elif (trade_object.stock2 not in coin_active):
                del_trade_obj_indx.append(obj_index)
                m2m_sum += trade_object.m2m
                config_object.masterlog.info("%s stock removed from FO %s" %(trade_object.stock2, self.timestamp.date()))
                continue
            
            elif force_sq_off:
                del_trade_obj_indx.append(obj_index)
                m2m_sum += trade_object.m2m
                config_object.masterlog.info("FORCE SQ-OFF: %s %s" %(trade_object.stock2, self.timestamp.date()))
            
            trade_object.ticks_passed += 1
        
        ####
        self.trade_stats[self.timestamp].append([m2m_sum, obj_index+1])
        
        del_trade_obj_indx.sort(reverse=True)
        
        for indx in del_trade_obj_indx:
            self.sq_off_trades_list.append(self.trade_manager.open_trade_list.pop(indx))


