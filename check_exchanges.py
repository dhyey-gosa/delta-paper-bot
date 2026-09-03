import ccxt

exchanges_to_try = [
    ('okx', {}),
    ('bybit', {}),
    ('kucoin', {}),
    ('gate', {}),
    ('mexc', {}),
]

for name, opts in exchanges_to_try:
    try:
        cls = getattr(ccxt, name)
        ex = cls(opts)
        data = ex.fetch_ohlcv('XAUT/USDT', '1m', limit=3)
        if data:
            print(f"OK  {name}: XAUT/USDT price={data[-1][4]}")
        else:
            print(f"NO DATA {name}: XAUT/USDT")
    except Exception as e:
        print(f"FAIL {name}: {e}")

# Try BTC on each
for name, opts in exchanges_to_try:
    try:
        cls = getattr(ccxt, name)
        ex = cls(opts)
        data = ex.fetch_ohlcv('BTC/USDT', '1m', limit=3)
        if data:
            print(f"OK  {name}: BTC/USDT price={data[-1][4]}")
    except Exception as e:
        print(f"FAIL {name}: BTC/USDT - {e}")
