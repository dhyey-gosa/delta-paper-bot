import ccxt

# Check OKX for silver
try:
    ex = ccxt.okx()
    markets = ex.load_markets()
    for sym in sorted(markets.keys()):
        s = sym.lower()
        if 'xag' in s or 'slv' in s or 'silver' in s:
            t = markets[sym].get('type', '')
            print(f'OKX: {sym} type={t}')
except Exception as e:
    print(f'OKX error: {e}')

# Check Binance futures for XAG
try:
    ex2 = ccxt.binance({'options': {'defaultType': 'future'}})
    markets2 = ex2.load_markets()
    for sym in sorted(markets2.keys()):
        if 'XAG' in sym:
            t = markets2[sym].get('type', '')
            print(f'Binance-futures: {sym} type={t}')
except Exception as e:
    print(f'Binance-futures error: {e}')

# Check bybit
try:
    ex3 = ccxt.bybit()
    markets3 = ex3.load_markets()
    for sym in sorted(markets3.keys()):
        if 'XAG' in sym:
            t = markets3[sym].get('type', '')
            print(f'Bybit: {sym} type={t}')
except Exception as e:
    print(f'Bybit error: {e}')
