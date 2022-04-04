
import logging
import sys
import hjson
import os
from pathlib import Path
from collections import OrderedDict

BASE_DIR = Path(__file__).parent
config_file = str(BASE_DIR / "config.hjson")

'''
Things left:
    1. Trading class params

'''

def print_key(dict_, print_ls, logger): 
    for key, value in dict_.items():
        if key in print_ls:
            logger.info("%s: %s" %(key, value))
        
        if isinstance(value, dict) or isinstance(value, OrderedDict): 
            print_key(value, print_ls, logger)

def setup_logger(filename, logger_name="pair_logs", level=logging.DEBUG):

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s: %(message)s", datefmt="%Y-%m-%d %I:%M %p")

    ## Settting File handler formatter to create log file
    file_handler = logging.FileHandler(filename, mode="w")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    ## Settting Stream handler formatter to print logs
    stream_handler = logging.StreamHandler(sys.stdout) # sys.stdout
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    ## Adding handler to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logging.getLogger(logger_name)


class config_object_class(object):

    def __init__(self):

        # with open("config.json", "r") as f:
        #     config = json.load(f)

        with open(config_file, "r") as f:
            config = hjson.load(f)


        self.config = config
        self.run_objective = config.get("run_objective", "No run objective")
        self.logfilename = config.get("log_filename", "logs.log")
        self.dump_path = config["dump_path"]
        self.masterlog = setup_logger(self.logfilename)

        self.masterlog.info(self.run_objective)
        self.masterlog.info("\n CONFIG \n")
        self.masterlog.info(config)

        self.masterlog.info("Setting up config object")

        ## Trade Dict
        self.pair_trade_dict = {}

        self.pair_selection_config = config.get("pair_selection_config", {})
        self.pair_selection_technique = self.pair_selection_config.get("pair_selection_technique", None)

        self.masterlog.info("USING PCA CONFIG")
        
        self.data_config = config.get("data_config", {})
        self.eod_data_file = self.data_config["eod_data"]
        self.intraday_data_file = self.data_config.get("intraday_data", None)
        self.intraday_data_frequency = self.data_config.get("intraday_data_freq", None)

        ## PCA config
        self.strategy_config = config.get("strategy_config", {})
        self.brokerage = self.strategy_config.get("brokerage", 0.002)
        self.spread_type = self.strategy_config["spread_type"]
        self.bbands_lookback = self.strategy_config.get("bbands_lookback", None)
        self.return_bars = self.strategy_config.get("return_bars", None)

        self.pca_file_path = self.strategy_config.get("pca_pair_file", None)
        self.pca_columns = self.strategy_config.get("pca_columns", None)
        self.combine_pca_columns = self.strategy_config.get("combine_pca_columns", "union") ####
        self.pca_trading_lookback = self.strategy_config.get("norm_trading_lookback", None)
        self.minimum_mean_reversion_days = self.strategy_config.get("minimum_mean_reversion_days", 2)
        self.pair_frequency = self.strategy_config.get("pair_frequency", "daily")
        self.data_resample_frequency = self.strategy_config.get("data_resample_freq", None)

        self.data_structure = self.strategy_config.get("data_structure", None)

        if self.data_structure not in ["pandas", "numpy"]:
            raise ValueError("data_structure must be pandas or numpy: %s" %self.data_structure)

        # static beta config
        self.beta_config = self.strategy_config.get("beta_config", {})
        self.beta_entry_threshold = self.beta_config["beta_entry_threshold"]
        self.recalculate_entry_beta_thresh = self.beta_config["recalculate_entry_beta_thresh"]
        self.static_entry_beta = self.beta_config["static_entry_beta"]
        self.static_beta_update_freq = self.beta_config["static_beta_update_freq"]

        if self.static_entry_beta:
            assert len(self.static_beta_update_freq) > 0, "Must have at least one update frequency"

        
        ## Price entry threshold
        self.price_entry_threshold = self.strategy_config["price_entry_threshold"]

        self.run_type_config = self.strategy_config.get("run_type_config", {})
        self.run_type = self.run_type_config.get("run_type", "backtesting")
        self.stats_base_file = self.run_type_config.get("base_file", None)

        ## PCA trade filter config
        self.trade_filter_config = self.strategy_config.get("trade_entry_config", {})

        ## PCA Sigma Filter
        self.sigma_filter_upper_thresh = self.trade_filter_config["entry_sigma_threshold"]
        self.sigma_filter_lower_thresh = self.trade_filter_config["entry_sigma_threshold"] * -1

        ## PCA ENTRY CONFIG
        self.trade_entry_config = self.strategy_config.get("trade_entry_config", {})
        
        ## Sigma Entry Parameters
        self.sigma_entry_upper_threshold = self.trade_entry_config["entry_sigma_threshold"]
        self.sigma_entry_lower_threshold = self.trade_entry_config["entry_sigma_threshold"] * -1        

        ## PCA EXIT CONFIG
        self.trade_exit_config = self.strategy_config.get("trade_exit_config", {})
        self.sigma_exit_upper_threshold = self.trade_exit_config["exit_sigma_threshold"]
        self.sigma_exit_lower_threshold = self.trade_exit_config["exit_sigma_threshold"] * -1
        self.sigma_exit_hard_threshold = self.trade_exit_config.get("hard_threshold", False)        

        # assert (self.sigma_exit_lower_threshold <= 0 and self.sigma_exit_upper_threshold >= 0), "Check sigma exit thresholds"

        self.stop_loss_config = self.strategy_config.get("stop_loss_config", None)
        self.rolling_stoploss = self.stop_loss_config.get("rolling_stoploss", False)
        self.rolling_stoploss_mult = self.stop_loss_config.get("stoploss_multiplier", None)

        assert self.stop_loss_config != None, "stop loss config not parsed" 
        assert self.rolling_stoploss_mult != None, "rolling stoploss multiplier not in stop loss config"

        print_key(self.config, self.config.get("print_keys", {}), self.masterlog)

        if os.popen('uname').read()[:-1] == "Darwin":
            self.masterlog.info("Running on Mac")
            self.eod_data_file = "../data/crypto_eod.csv"
            self.intraday_data_file = "../data/1min_crypto_sample.csv"
            self.intraday_data_frequency = 1

config_object = config_object_class()

