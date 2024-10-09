import numpy as np
from numba import njit

@njit(cache=True)
def merge_pnl(arr1, arr2):

    out = np.zeros((len(arr1) + len(arr2)))
    idx = 1

    for i in range(len(arr1) + len(arr2)):
        
        if i % 2 == 0:
            out[i] = arr1[int(i/2)]
        else:
            out[i] = arr2[i-idx]

        idx += 1

    return out

@njit(cache=True)
def get_drawdowns(arr):

    drawdowns = np.zeros((len(arr)))
    max = arr[0]

    for i in range(1, len(drawdowns)-1):

        if arr[i-1] > arr[i] and arr[i] < arr[i+1]:
            min = arr[i]
            drawdowns[i] = max - min
        elif arr[i-1] < arr[i] and arr[i] > arr[i+1]:
            max = arr[i]

    return drawdowns

@njit(cache=True)
def get_pnl(trade_close_prices, trade_open_prices, commission, slippage, init_inv, trade_size, is_buy):

    pnl_list = np.zeros(len(trade_close_prices))

    for i in range(len(trade_close_prices)):

        temp_n_assets = int(init_inv * trade_size / trade_open_prices[i])

        if is_buy == 1:
            temp_pnl = temp_n_assets * (trade_close_prices[i] - trade_open_prices[i] * (1 + slippage))
        else:
            temp_pnl = -temp_n_assets * (trade_close_prices[i] - trade_open_prices[i] * (1 - slippage))

        temp_pnl = temp_pnl * (1 - commission)
        init_inv += temp_pnl
        pnl_list[i] = temp_pnl

    return pnl_list

@njit(cache=True)
def trading_signals(buy_signal, sell_signal):
    buy = np.where(buy_signal, 1, 0)
    sell = np.where(sell_signal, -1, 0)
    signal = buy + sell
    buy_idxs = []
    sell_idxs = []
    is_buy = 0
    is_sell = 0
    for i in range(len(signal)):
        if signal[i] == 1 and is_buy == 0:
            buy_idxs.append(i + 1)
            is_buy = 1
            is_sell = 0
        elif signal[i] == -1 and is_sell == 0:
            sell_idxs.append(i + 1)
            is_sell = 1
            is_buy = 0
    if len(buy_idxs) > len(sell_idxs):
        buy_idxs = buy_idxs[:-(len(buy_idxs) - len(sell_idxs))]
    elif len(buy_idxs) < len(sell_idxs):
        sell_idxs = sell_idxs[:-(len(sell_idxs) - len(buy_idxs))]
    return buy_idxs, sell_idxs