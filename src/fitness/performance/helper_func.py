import numpy as np
from numba import njit, prange

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

@njit(cache=True)
def get_lag(prices, lag=1):
    n = len(prices)
    result = np.full(n, -999, dtype=np.float64)  # Initialize with -999

    for i in range(lag, n):
        result[i] = prices[i - lag]

    return result

@njit(cache=True)
def get_max_drawdown(pnl_list):
    max_dd = 0.0
    peak = pnl_list[0]

    for i in range(1, len(pnl_list)):
        if pnl_list[i] > peak:
            peak = pnl_list[i]
        drawdown = 100 * (peak - pnl_list[i]) / peak
        if drawdown > max_dd:
            max_dd = drawdown

    return max_dd

@njit(cache=True)
def get_random_idxs(arr, num_elements, exclude_arr=np.array([])):
    # Get the length of the input array
    n = len(arr)

    if num_elements > n:
        raise ValueError("num_elements must be less than or equal to the length of the array.")

    # Set up the set of excluded indices for quick look-up
    exclude_set = set(exclude_arr)

    # Create an array to hold unique indices
    indices = np.zeros(num_elements, dtype=np.int32)
    chosen_set = set()

    for i in range(num_elements):
        idx = np.random.randint(0, n)

        # Ensure the index is unique and not in the exclude_arr
        while idx in chosen_set or idx in exclude_set:
            idx = np.random.randint(0, n)

        indices[i] = idx
        chosen_set.add(idx)

    # Return the randomly selected elements, sorted
    return np.sort(indices)

@njit(cache=True)
def get_monkey_test_results(open_prices, buy_idxs, sell_idxs, COMMISSION, SLIPPAGE, AVAILABLE_CAPITAL, TRADE_SIZE, n_runs=8000):
    
    pnl_arr = np.zeros(n_runs)
    max_dd_arr = np.zeros(n_runs)
    
    for i in prange(n_runs):
        # Generate new buy indexes
        new_buy_idxs = get_random_idxs(np.arange(len(open_prices)), num_elements=len(buy_idxs))

        # Generate new sell indexes
        new_sell_idxs = get_random_idxs(np.arange(len(open_prices)), num_elements=len(sell_idxs), exclude_arr=new_buy_idxs)

        # Filter sell indexes to avoid overlaps
        buy_idxs = new_buy_idxs
        sell_idxs = new_sell_idxs

        # Fetch buy and sell prices
        buy_prices = open_prices[buy_idxs]
        sell_prices = open_prices[sell_idxs]

        # Calculate P&L and merge results based on buy/sell sequence
        if buy_idxs[0] < sell_idxs[0]:
            buy_arr = get_pnl(sell_prices, buy_prices, COMMISSION, SLIPPAGE, AVAILABLE_CAPITAL, TRADE_SIZE, 1)
            buy_pnl = np.sum(buy_arr)
            sell_arr = get_pnl(buy_prices[1:], sell_prices[:-1], COMMISSION, SLIPPAGE, AVAILABLE_CAPITAL, TRADE_SIZE, 0)
            sell_pnl = np.sum(sell_arr)
            all_arr = merge_pnl(buy_arr, sell_arr)
        else:
            sell_arr = get_pnl(buy_prices, sell_prices, COMMISSION, SLIPPAGE, AVAILABLE_CAPITAL, TRADE_SIZE, 0)
            sell_pnl = np.sum(sell_arr)
            buy_arr = get_pnl(sell_prices[1:], buy_prices[:-1], COMMISSION, SLIPPAGE, AVAILABLE_CAPITAL, TRADE_SIZE, 1)
            buy_pnl = np.sum(buy_arr)
            all_arr = merge_pnl(sell_arr, buy_arr)

        pnl_arr[i] = buy_pnl + sell_pnl
        max_dd_arr[i] = get_max_drawdown(all_arr)
    
    return pnl_arr, max_dd_arr