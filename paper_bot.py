"""
Delta Exchange Paper Forward Test Bot
Live prices via Delta Exchange public API, paper trading with 100 INR per asset, 50x leverage.
Deployed on Render, kept alive by UptimeRobot.

Assets: XAUTUSD (gold), SLVONUSD (silver), DOGEUSD
Strategy: VWAP Pullback Scalper on 1min
Product IDs: XAUTUSD=131253, SLVONUSD=124058, DOGEUSD=14745
"""
import os
import sys
import time
import json
import logging
import threading
from datetime import datetime, timezone
from collections import deque

import numpy as np
import requests

# === CONFIG ===
API_BASE = 'https://api.india.delta.exchange'

ASSETS = {
    131253: {'symbol': 'XAUTUSD', 'name': 'Gold (XAUT)',  'capital_inr': 100, 'leverage': 50, 'tick': 0.01,
             'target_pct': 0.0020, 'stop_pct': 0.0015},   # 0.20% target, 0.15% stop
    124058: {'symbol': 'SLVONUSD', 'name': 'Silver (SLV)','capital_inr': 100, 'leverage': 50, 'tick': 0.001,
             'target_pct': 0.0035, 'stop_pct': 0.0025},   # 0.35% target, 0.25% stop (wider for volatility)
    14745:  {'symbol': 'DOGEUSD', 'name': 'DOGE',         'capital_inr': 100, 'leverage': 50, 'tick': 0.00001,
             'target_pct': 0.0020, 'stop_pct': 0.0015},   # 0.20% target, 0.15% stop
}

# Strategy params (default - overridden per-asset above)
TARGET_PCT = 0.0020   # 0.20%
STOP_PCT = 0.0015     # 0.15%
FEE_PER_SIDE = 0.0001  # 0.01%
COOLDOWN_BARS = 0

FETCH_INTERVAL = 15  # seconds

# === LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger('paper_bot')


# === DELTA PUBLIC API (no auth needed) ===
class DeltaPublicAPI:
    def __init__(self, base_url=API_BASE):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'DeltaPaperBot/1.0'})

    def get_candles(self, symbol, resolution='1m', limit=200):
        """Fetch candles from Delta public API using symbol + start/end timestamps"""
        import time as _time
        end = int(_time.time())
        # Calculate start based on resolution and limit
        resolution_seconds = {
            '1m': 60, '3m': 180, '5m': 300, '15m': 900,
            '30m': 1800, '1h': 3600, '4h': 14400, '1d': 86400,
        }
        secs = resolution_seconds.get(resolution, 60)
        start = end - (secs * limit)

        url = f"{self.base_url}/v2/history/candles"
        params = {
            'resolution': resolution,
            'symbol': symbol,
            'start': start,
            'end': end,
        }
        try:
            r = self.session.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and 'result' in data:
                return data['result'][-limit:]
            elif isinstance(data, list):
                return data[-limit:]
            return []
        except Exception as e:
            log.error(f"Candle fetch error {symbol} {resolution}: {e}")
            return []

    def get_ticker(self, symbol):
        """Fetch current ticker"""
        url = f"{self.base_url}/v2/tickers/{symbol}"
        try:
            r = self.session.get(url, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.error(f"Ticker fetch error {symbol}: {e}")
            return None


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
    if len(arr) < n:
        return out
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
    def __init__(self, product_id, symbol, name, capital, leverage, target_pct=0.0020, stop_pct=0.0015):
        self.product_id = product_id
        self.symbol = symbol
        self.name = name
        self.capital = capital
        self.initial_capital = capital
        self.leverage = leverage
        self.target_pct = target_pct
        self.stop_pct = stop_pct
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
            'product_id': self.product_id,
            'symbol': self.symbol,
            'name': self.name,
            'capital': round(self.capital, 2),
            'initial_capital': self.initial_capital,
            'leverage': self.leverage,
            'target_pct': f'{self.target_pct*100:.2f}%',
            'stop_pct': f'{self.stop_pct*100:.2f}%',
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

    # Per-asset stop/target
    target_pct = getattr(state, 'target_pct', TARGET_PCT)
    stop_pct = getattr(state, 'stop_pct', STOP_PCT)

    # LONG
    if (trend_up and vwap_dist < 0.0005 and
        40 < rsi_val < 55 and price > o and price > vwap_val * 0.999):
        sd = max(atr_val * 2, price * stop_pct * 0.5)
        sd = min(sd, price * stop_pct)
        return ('long', price, price - sd, price + sd * 1.5)

    # SHORT
    if (not trend_up and vwap_dist < 0.0005 and
        45 < rsi_val < 60 and price < o and price < vwap_val * 1.001):
        sd = max(atr_val * 2, price * stop_pct * 0.5)
        sd = min(sd, price * stop_pct)
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
        self.api = DeltaPublicAPI()
        self.states = {}
        self.running = False
        self.start_time = datetime.now(timezone.utc).isoformat()
        self.log_lines = deque(maxlen=200)

    def setup(self):
        for pid, cfg in ASSETS.items():
            state = AssetState(pid, cfg['symbol'], cfg['name'], cfg['capital_inr'], cfg['leverage'],
                               target_pct=cfg.get('target_pct', TARGET_PCT),
                               stop_pct=cfg.get('stop_pct', STOP_PCT))
            self.states[pid] = state
            tp = cfg.get('target_pct', TARGET_PCT) * 100
            sp = cfg.get('stop_pct', STOP_PCT) * 100
            log.info(f"  {cfg['symbol']} (id={pid}): capital={cfg['capital_inr']} INR, leverage={cfg['leverage']}x, target={tp:.2f}%, stop={sp:.2f}%")
        return True

    def fetch_candles(self, state):
        # 1min candles
        raw_1m = self.api.get_candles(state.symbol, '1m', 200)
        if not raw_1m:
            return
        # Parse: {time, open, high, low, close, volume}
        state.candles_1m = [
            (float(c['open']), float(c['high']), float(c['low']),
             float(c['close']), float(c['volume']))
            for c in raw_1m
        ]
        state.last_price = float(raw_1m[-1]['close'])

        # 15min candles
        raw_15m = self.api.get_candles(state.symbol, '15m', 100)
        if raw_15m:
            state.candles_15m = [
                (float(c['open']), float(c['high']), float(c['low']),
                 float(c['close']), float(c['volume']))
                for c in raw_15m
            ]

        state.last_update = datetime.now(timezone.utc).isoformat()

    def process(self, state):
        if not state.active or len(state.candles_1m) < 2:
            return
        state.cycle_count += 1
        candle = state.candles_1m[-1]

        # Exit check
        if state.position:
            ex = check_exit(state, candle)
            if ex:
                ep, reason = ex
                self._close(state, ep, reason)

        # Entry check
        if not state.position:
            entry = check_entry(state)
            if entry:
                d, ep, stop, target = entry
                self._open(state, d, ep, stop, target)

    def _open(self, state, direction, entry, stop, target):
        notional = state.capital * state.leverage * 0.5
        state.position = {
            'direction': direction,
            'entry': entry,
            'stop': stop,
            'target': target,
            'notional': notional,
            'bars_held': 0,
        }
        log.info(f"[{state.symbol}] OPEN {direction.upper()} @ {entry:.6f} | Stop={stop:.6f} Target={target:.6f}")

    def _close(self, state, exit_price, reason):
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
        log.info(f"[{state.symbol}] CLOSE {pos['direction'].upper()} @ {exit_price:.6f} | PnL={pnl_net:+.4f} INR | Cap={state.capital:.2f} | {reason}")
        state.position = None

    def run(self):
        log.info("=" * 60)
        log.info("PAPER FORWARD BOT STARTING (Delta Exchange API)")
        log.info("=" * 60)
        self.running = True
        cycle = 0

        while self.running:
            cycle += 1
            for pid, state in self.states.items():
                if not state.active:
                    continue
                try:
                    self.fetch_candles(state)
                    self.process(state)
                except Exception as e:
                    err = f"[{state.symbol}] Error: {e}"
                    log.error(err)
                    state.errors.append(err)

            if cycle % 20 == 0:
                for pid, s in self.states.items():
                    wr = s.win_count / max(1, len(s.trades)) * 100
                    log.info(f"[{s.symbol}] Price={s.last_price:.4f} Cap={s.capital:.2f} PnL={s.total_pnl:+.4f} Trades={len(s.trades)} WR={wr:.0f}%")

            time.sleep(FETCH_INTERVAL)

    def stop(self):
        self.running = False


# === FLASK ===
bot = PaperBot()
bot.setup()  # Initialize states at module level so Flask routes work immediately

# Shared mutable state - thread writes, Flask reads
_shared = {'bot': bot}

def create_app():
    from flask import Flask, jsonify
    app = Flask(__name__)
    app.bot = bot  # Bind bot to Flask app instance

    @app.route('/')
    def index():
        b = app.bot
        return jsonify({
            'status': 'running',
            'bot': 'Delta Paper Forward Test',
            'api': 'Delta Exchange India (public)',
            'strategy': 'VWAP Pullback Scalper',
            'config': {
                'target': 'per-asset (see assets)',
                'stop': 'per-asset (see assets)',
                'leverage': '50x',
                'fees': f'{FEE_PER_SIDE*100:.2f}%/side',
                'capital_per_asset': '100 INR',
            },
            'uptime_since': b.start_time,
            'assets': {str(s): st.to_dict() for s, st in b.states.items()},
        })

    @app.route('/status')
    def status():
        b = app.bot
        total_pnl = sum(s.total_pnl for s in b.states.values())
        total_trades = sum(len(s.trades) for s in b.states.values())
        total_wins = sum(s.win_count for s in b.states.values())
        return jsonify({
            'status': 'ok',
            'uptime': b.start_time,
            'summary': {
                'total_pnl': round(total_pnl, 4),
                'total_trades': total_trades,
                'total_wins': total_wins,
                'overall_win_rate': round(total_wins / max(1, total_trades) * 100, 1),
            },
            'assets': {str(s): st.to_dict() for s, st in b.states.items()},
        })

    @app.route('/trades')
    def trades():
        b = app.bot
        return jsonify({str(s): st.trades[-20:] for s, st in b.states.items()})

    @app.route('/health')
    def health():
        return jsonify({'status': 'healthy'})

    return app


# === MAIN ===
app = create_app()

# Start bot thread
_bot_thread = threading.Thread(target=bot.run, daemon=True)
_bot_thread.start()
log.info("Bot thread started")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    log.info(f"Starting waitress on port {port}")
    from waitress import serve
    serve(app, host='0.0.0.0', port=port, threads=4)
