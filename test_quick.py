import time
from paper_bot import app, bot

time.sleep(3)
r = app.test_client().get('/')
print("Status:", r.status_code)
d = r.get_json()
print("Bot status:", d['status'])
for k, v in d['assets'].items():
    print(f"  {k}: price={v['last_price']}, cap={v['capital']}, trades={v['trades_count']}")

r2 = app.test_client().get('/health')
print("Health:", r2.status_code, r2.get_json())

bot.stop()
