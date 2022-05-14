
from turtle import clear
import pandas as pd
import requests
import datetime as dt
import time

all_data = pd.DataFrame()
for i in range(1000):
    print(i)
    r = requests.get("https://api.binance.com/api/v3/depth",
                    params=dict(symbol="ETHUSDT"))
    results = r.json()


    frames = {side: pd.DataFrame(data=results[side], columns=["price", "quantity"],
                                dtype=float)
            for side in ["bids", "asks"]}

    ###
    frames_list = [frames[side].assign(side=side) for side in frames]
    data = pd.concat(frames_list, axis="index", 
                    ignore_index=True, sort=True)

    data["time"] = dt.datetime.now()
    all_data = pd.concat([all_data, data])
    time.sleep(1)





clear