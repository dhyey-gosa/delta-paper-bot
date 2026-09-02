import sys, time, requests
sys.path.insert(0, r'E:\Profitable-Strats\delta exchange strats\forward_test')
from bot import ForwardBot, ASSETS, AssetState

bot = ForwardBot()
bot.api = None
bot.mode = 'testnet'
bot._discover_products()

# Init states
for sym, cfg in ASSETS.items():
    pid = bot.product_map.get(sym)
    if pid:
        bot.states[sym] = AssetState(sym, cfg['capital_inr'], cfg['leverage'])

# Test candle fetching
now = int(time.time())
for sym, state in bot.states.items():
    pid = bot.product_map[sym]
    try:
        r = requests.get('https://api.india.delta.exchange/v2/history/candles',
                        params={'resolution': '1m', 'symbol': sym,
                                'start': now - 120*60, 'end': now},
                        timeout=10)
        data = r.json()
        candles = data.get('result', [])
        print(f"{sym}: {len(candles)} candles fetched")
        if candles:
            c = candles[-1]
            print(f"  Latest: time={c['time']} O={c['open']} H={c['high']} L={c['low']} C={c['close']} V={c['volume']}")
    except Exception as e:
        print(f"{sym}: Error - {e}")

# Test Flask
from bot import create_app
app = create_app()
with app.test_client() as client:
    resp = client.get('/')
    print(f"\nFlask / endpoint: {resp.status_code}")
    print(resp.get_json())
