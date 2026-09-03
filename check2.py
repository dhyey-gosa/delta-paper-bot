import ccxt

# Try OKX
try:
    ex = ccxt.okx({'enableRateLimit': True})
    data = ex.fetch_ohlcv('XAUT/USDT', '1m', limit=3)
    print(f"OKX XAUT: {data[-1][4]}")
except Exception as e:
    print(f"OKX FAIL: {e}")

# Try Bybit
try:
    ex = ccxt.bybit({'enableRateLimit': True})
    data = ex.fetch_ohlcv('XAUT/USDT', '1m', limit=3)
    print(f"Bybit XAUT: {data[-1][4]}")
except Exception as e:
    print(f"Bybit FAIL: {e}")
