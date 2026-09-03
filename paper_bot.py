"""
Delta Exchange Paper Forward Test Bot
Live prices via ccxt, paper trading with 100 INR per asset, 50x leverage.
Deployed on Render, kept alive by UptimeRobot.

Assets: XAUT/USDT (gold), XAG/USDT (silver), DOGE/USDT
Strategy: VWAP Pullback Scalper on 1min
"""
import os
import sys
import time
import json
import logging
import threading
from datetime import datetime, timezone
from collections import deque

import ccxt
import numpy as np

# === CONFIG ===
ASSETS = {
    'XAUT/USDT':  {'capital_inr': 100, 'leverage': 50, 'exchange': 'binance', 'type': 'spot'},
    'XAG/USDT':   {'capital_inr': 100, 'leverage': 50, 'exchange': 'binance', 'type': 'future'},
    'DOGE/USDT':  {'capital_inr': 100, 'leverage': 50, 'exchange': 'binance', 'type': 'spot'},
}

# Strategy params
TARGET_PCT = 0.0020   # 0.20%
STOP_PCT = 0.0015     # 0.15%
FEE_PER_SIDE = 0.0001  # 0.01%
COOLDOWN_BARS = 0

# How often to fetch candles (seconds)
FETCH_INTERVAL = 15

# === LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger('paper_bot')


# === INDICATORS (numpy, no deps) ===
def ema_np(arr, n):
    alpha = 2.0 / (n + 1)
    out = np.empty(len(arr), dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
    return out

def sma_np(arr, n):
    out = np.full(len(arr), np.nan, dtype=float)
    cs = np.cumsum(arr)
    out[n-1:] = (cs[n-1:] - np.concatenate([[0], cs[:-n]])) / n
    return out

def rsi_np(arr, n=14):
    out = np.full(len(arr), 50.0, dtype=float)
    if len(arr) < n + 2:
        return out
    d = np.diff(arr)
    g = np.where(d > 0, d, 0.0)
    l = np.where(d < 0, -d, 0.0)
    ag = np.mean(g[:n])
    al = np.mean(l[:n])
    for i in range(n, len(d)):
        ag = (ag * (n - 1) + g[i]) / n
        al = (al * (n - 1) + l[i]) / n
        out[i + 1] = 100 - (100 / (1 + ag / al)) if al > 0 else 100
    return out


# === ASSET STATE ===
class AssetState:
    def __init__(self, symbol, capital, leverage):
        self.symbol = symbol
        self.capital = capital
        self.initial_capital = capital
        self.leverage = leverage
        self.position = None
        self.trades = []
        self.win_count = 0
        self.total_pnl = 0
        self.candles_1m = []
        self.candles_15m = []
        self.last_price = 0
        self.active = True
        self.last_update = None
        self.errors = []
        self.cycle_count = 0

    def to_dict(self):
        return {
            'symbol': self.symbol,
            'capital': round(self.capital, 2),
            'initial_capital': self.initial_capital,
            'leverage': self.leverage,
            'position': {k: round(v, 6) if isinstance(v, float) else v
                         for k, v in self.position.items()} if self.position else None,
            'total_pnl': round(self.total_pnl, 4),
            'trades_count': len(self.trades),
            'wins': self.win_count,
            'win_rate': round(self.win_count / max(1, len(self.trades)) * 100, 1),
            'last_price': self.last_price,
            'active': self.active,
            'last_update': self.last_update,
            'recent_trades': self.trades[-5:],
            'errors': self.errors[-3:],
        }


# === PAPER EXCHANGE ===
class PaperExchange:
    def __init__(self):
        self.exchanges = {}

    def _get_exchange(self, name, mtype='spot'):
        key = f"{name}_{mtype}"
        if key not in self.exchanges:
            opts = {'enableRateLimit': True}
            if mtype == 'future':
                opts['options'] = {'defaultType': 'future'}
            self.exchanges[key] = ccxt.binance(opts)
        return self.exchanges[key]

    def fetch_ohlcv(self, symbol, exchange_name, timeframe='1m', limit=200, mtype='spot'):
        ex = self._get_exchange(exchange_name, mtype)
        try:
            data = ex.fetch_ohlcv(symbol, timeframe, limit=limit)
            return data
        except Exception as e:
            log.error(f"Fetch error {symbol}: {e}")
            return []

    def fetch_ticker_price(self, symbol, exchange_name, mtype='spot'):
        ex = self._get_exchange(exchange_name, mtype)
        try:
            t = ex.fetch_ticker(symbol)
            return t['last']
        except:
            return None


# === STRATEGY ===
def check_entry(state):
    if len(state.candles_1m) < 30 or len(state.candles_15m) < 25:
        return None

    c1m = np.array(state.candles_1m, dtype=float)  # [o, h, l, c, v]
    c15m = np.array(state.candles_15m, dtype=float)

    closes_1m = c1m[:, 3]   # close
    closes_15m = c15m[:, 3]

    # 15min trend
    e9 = ema_np(closes_15m, 9)
    e21 = ema_np(closes_15m, 21)
    if np.isnan(e9[-1]) or np.isnan(e21[-1]):
        return None
    trend_up = e9[-1] > e21[-1]

    # 1min VWAP
    typical = (c1m[:, 1] + c1m[:, 2] + c1m[:, 3]) / 3
    tp_vol = typical * c1m[:, 4]
    cum_tp_vol = np.cumsum(tp_vol)
    cum_vol = np.cumsum(c1m[:, 4])
    with np.errstate(divide='ignore', invalid='ignore'):
        vwap = np.where(cum_vol > 0, cum_tp_vol / cum_vol, np.nan)
    vwap_val = vwap[-1]
    if np.isnan(vwap_val):
        return None

    # 1min RSI
    rsi_vals = rsi_np(closes_1m, 14)
    rsi_val = rsi_vals[-1]

    # 1min ATR
    highs = c1m[:, 1]
    lows = c1m[:, 2]
    trigs = np.maximum(highs[1:] - lows[1:],
                       np.maximum(np.abs(highs[1:] - closes_1m[:-1]),
                                  np.abs(lows[1:] - closes_1m[:-1])))
    atr_val = trigs[-14:].mean() if len(trigs) >= 14 else trigs.mean()

    price = closes_1m[-1]
    o = c1m[-1, 0]
    vol = c1m[-1, 4]

    # Volume filter
    vol_sma = sma_np(c1m[:, 4], 20)
    vol_ok = vol > vol_sma[-1] * 0.8 if not np.isnan(vol_sma[-1]) else True
    if not vol_ok:
        return None

    vwap_dist = abs(price - vwap_val) / price

    # LONG
    if (trend_up and vwap_dist < 0.0005 and
        40 < rsi_val < 55 and price > o and price > vwap_val * 0.999):
        sd = max(atr_val * 2, price * STOP_PCT * 0.5)
        sd = min(sd, price * STOP_PCT)
        return ('long', price, price - sd, price + sd * 1.5)

    # SHORT
    if (not trend_up and vwap_dist < 0.0005 and
        45 < rsi_val < 60 and price < o and price < vwap_val * 1.001):
        sd = max(atr_val * 2, price * STOP_PCT * 0.5)
        sd = min(sd, price * STOP_PCT)
        return ('short', price, price + sd, price - sd * 1.5)

    return None


def check_exit(state, current_candle):
    pos = state.position
    if not pos:
        return None
    h, l, c = current_candle[1], current_candle[2], current_candle[3]
    pos['bars_held'] = pos.get('bars_held', 0) + 1
    bars = pos['bars_held']

    if pos['direction'] == 'long':
        if l <= pos['stop']:
            return (pos['stop'], 'stop')
        if h >= pos['target']:
            return (pos['target'], 'target')
        if bars >= 15:
            return (c, 'timeout')
        if bars >= 5:
            be = pos['entry'] * (1 + FEE_PER_SIDE * 2.5)
            if be > pos['stop']:
                pos['stop'] = be
    else:
        if h >= pos['stop']:
            return (pos['stop'], 'stop')
        if l <= pos['target']:
            return (pos['target'], 'target')
        if bars >= 15:
            return (c, 'timeout')
        if bars >= 5:
            be = pos['entry'] * (1 - FEE_PER_SIDE * 2.5)
            if be < pos['stop']:
                pos['stop'] = be
    return None


# === BOT ===
class PaperBot:
    def __init__(self):
        self.exchange = PaperExchange()
        self.states = {}
        self.running = False
        self.start_time = datetime.now(timezone.utc).isoformat()
        self.log_lines = deque(maxlen=200)

    def setup(self):
        for sym, cfg in ASSETS.items():
            state = AssetState(sym, cfg['capital_inr'], cfg['leverage'])
            self.states[sym] = state
            log.info(f"  {sym}: capital={cfg['capital_inr']} INR, leverage={cfg['leverage']}x")
        return True

    def fetch_candles(self, symbol, state):
        cfg = ASSETS[symbol]
        mtype = cfg.get('type', 'spot')
        data_1m = self.exchange.fetch_ohlcv(symbol, cfg['exchange'], '1m', 200, mtype)
        if not data_1m:
            return
        state.candles_1m = [(d[1], d[2], d[3], d[4], d[5]) for d in data_1m]
        state.last_price = data_1m[-1][4]

        data_15m = self.exchange.fetch_ohlcv(symbol, cfg['exchange'], '15m', 100, mtype)
        if data_15m:
            state.candles_15m = [(d[1], d[2], d[3], d[4], d[5]) for d in data_15m]

        state.last_update = datetime.now(timezone.utc).isoformat()

    def process(self, symbol, state):
        if not state.active or len(state.candles_1m) < 2:
            return
        state.cycle_count += 1
        candle = state.candles_1m[-1]

        # Exit check
        if state.position:
            ex = check_exit(state, candle)
            if ex:
                ep, reason = ex
                self._close(symbol, state, ep, reason)

        # Entry check
        if not state.position:
            entry = check_entry(state)
            if entry:
                d, ep, stop, target = entry
                self._open(symbol, state, d, ep, stop, target)

    def _open(self, symbol, state, direction, entry, stop, target):
        notional = state.capital * state.leverage * 0.5
        state.position = {
            'direction': direction,
            'entry': entry,
            'stop': stop,
            'target': target,
            'notional': notional,
            'bars_held': 0,
        }
        log.info(f"[{symbol}] OPEN {direction.upper()} @ {entry:.6f} | Stop={stop:.6f} Target={target:.6f}")

    def _close(self, symbol, state, exit_price, reason):
        pos = state.position
        if pos['direction'] == 'long':
            pnl_pct = (exit_price - pos['entry']) / pos['entry']
        else:
            pnl_pct = (pos['entry'] - exit_price) / pos['entry']
        pnl_pct *= state.leverage
        pnl_inr = pnl_pct * state.capital
        fees = pos['notional'] * FEE_PER_SIDE * 2
        pnl_net = pnl_inr - fees

        state.capital += pnl_net
        state.total_pnl += pnl_net
        if pnl_net > 0:
            state.win_count += 1
        state.trades.append({
            'd': pos['direction'], 'e': round(pos['entry'], 6),
            'x': round(exit_price, 6), 'pnl': round(pnl_net, 4),
            'r': reason, 'bars': pos['bars_held'],
        })
        log.info(f"[{symbol}] CLOSE {pos['direction'].upper()} @ {exit_price:.6f} | PnL={pnl_net:+.4f} INR | Cap={state.capital:.2f} | {reason}")
        state.position = None

    def run(self):
        log.info("=" * 60)
        log.info("PAPER FORWARD BOT STARTING")
        log.info("=" * 60)
        self.setup()
        self.running = True
        cycle = 0

        while self.running:
            cycle += 1
            for symbol, state in self.states.items():
                if not state.active:
                    continue
                try:
                    self.fetch_candles(symbol, state)
                    self.process(symbol, state)
                except Exception as e:
                    err = f"[{symbol}] Error: {e}"
                    log.error(err)
                    state.errors.append(err)

            if cycle % 20 == 0:
                for sym, s in self.states.items():
                    wr = s.win_count / max(1, len(s.trades)) * 100
                    log.info(f"[{sym}] Price={s.last_price:.4f} Cap={s.capital:.2f} PnL={s.total_pnl:+.4f} Trades={len(s.trades)} WR={wr:.0f}%")

            time.sleep(FETCH_INTERVAL)

    def stop(self):
        self.running = False


# === FLASK ===
bot = PaperBot()

def create_app():
    from flask import Flask, jsonify
    app = Flask(__name__)

    @app.route('/')
    def index():
        return jsonify({
            'status': 'running',
            'bot': 'Delta Paper Forward Test',
            'uptime_since': bot.start_time,
            'assets': {s: st.to_dict() for s, st in bot.states.items()},
        })

    @app.route('/status')
    def status():
        return jsonify({
            'status': 'ok',
            'uptime': bot.start_time,
            'assets': {s: st.to_dict() for s, st in bot.states.items()},
        })

    @app.route('/trades')
    def trades():
        return jsonify({s: st.trades[-20:] for s, st in bot.states.items()})

    @app.route('/health')
    def health():
        return jsonify({'status': 'healthy'})

    return app


# === MAIN ===
app = create_app()

# Start bot thread at module level (for gunicorn)
_bot_thread = threading.Thread(target=bot.run, daemon=True)
_bot_thread.start()
log.info("Bot thread started at module level")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    log.info(f"Flask on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
