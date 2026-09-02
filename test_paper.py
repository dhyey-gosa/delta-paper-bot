import sys, time
sys.path.insert(0, r'E:\Profitable-Strats\delta exchange strats\forward_test')

# Quick test: fetch candles + run strategy for a few cycles
from paper_bot import PaperBot, PaperExchange, check_entry

bot = PaperBot()
bot.setup()

# Fetch candles for each asset
for sym, state in bot.states.items():
    bot.fetch_candles(sym, state)
    print(f"{sym}: {len(state.candles_1m)} 1min bars, {len(state.candles_15m)} 15min bars, price={state.last_price}")

# Run a few cycles
print("\nRunning 5 strategy cycles...")
for i in range(5):
    for sym, state in bot.states.items():
        bot.process(sym, state)
    time.sleep(0.1)

# Check results
for sym, state in bot.states.items():
    print(f"\n{sym}:")
    print(f"  Position: {state.position}")
    print(f"  Trades: {len(state.trades)}")
    print(f"  Capital: {state.capital:.2f}")

# Test Flask
from paper_bot import create_app
app = create_app()
with app.test_client() as client:
    r = client.get('/')
    import json
    data = r.get_json()
    print(f"\nFlask / - Status: {r.status_code}")
    for sym, info in data['assets'].items():
        print(f"  {sym}: price={info['last_price']}, capital={info['capital']}, trades={info['trades_count']}")
