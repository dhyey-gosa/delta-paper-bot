"""
Delta Exchange Forward Test Bot
Runs on Render, kept alive by UptimeRobot.
Trades XAUTUSD, SLVONUSD, DOGE with 100 INR each, 50x leverage.

Strategy: VWAP Pullback Scalper
- 15min trend filter (EMA9 vs EMA21)
- 1min VWAP pullback entry
- RSI confirmation
- 0.20% target, 0.15% stop, 1.5R
- Breakeven after 5 bars, timeout at 15 bars
"""
import os
import sys
import time
import hmac
import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from collections import deque

import requests
import numpy as np

# === CONFIG ===
PROD_URL = 'https://api.india.delta.exchange'
TESTNET_URL = 'https://cdn-ind.testnet.deltaex.org'

# Assets to trade - 100 INR each, 50x leverage
ASSETS = {
    'XAUTUSD': {'capital_inr': 100, 'leverage': 50, 'product_id': None},
    'SLVONUSD': {'capital_inr': 100, 'leverage': 50, 'product_id': None},
    'DOGEUSD': {'capital_inr': 100, 'leverage': 50, 'product_id': None},
}

# Strategy params (best from backtest, adjusted for 50x)
TARGET_PCT = 0.0020   # 0.20%
STOP_PCT = 0.0015     # 0.15%
FEE_PER_SIDE = 0.0001  # 0.01% as specified
COOLDOWN_BARS = 0
MAX_TRADES_PER_DAY = 9999

# Candle fetch interval
CANDLE_INTERVAL_SEC = 60  # fetch new 1min candle every 60s

# === LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger('forward_bot')

# === STATE (per asset) ===
class AssetState:
    def __init__(self, symbol, capital, leverage):
        self.symbol = symbol
        self.capital = capital
        self.initial_capital = capital
        self.leverage = leverage
        self.position = None  # {direction, entry, stop, target, notional, entry_time, bars_held}
        self.trades = []
        self.equity_history = deque(maxlen=1000)
        self.last_signal_bar = -999
        self.candles_1m = deque(maxlen=200)   # (ts, o, h, l, c, v)
        self.candles_15m = deque(maxlen=100)  # (ts, o, h, l, c, v)
        self.total_pnl = 0
        self.win_count = 0
        self.loss_count = 0
        self.active = True
        self.last_update = None
        self.errors = []
    
    def to_dict(self):
        return {
            'symbol': self.symbol,
            'capital': round(self.capital, 2),
            'initial_capital': self.initial_capital,
            'leverage': self.leverage,
            'position': self.position,
            'total_pnl': round(self.total_pnl, 2),
            'trades': len(self.trades),
            'win_rate': round(self.win_count / max(1, len(self.trades)) * 100, 1),
            'active': self.active,
            'last_update': self.last_update,
            'recent_trades': self.trades[-5:] if self.trades else [],
            'errors': self.errors[-3:] if self.errors else [],
        }


# === DELTA API ===
class DeltaAPI:
    def __init__(self, base_url, api_key, api_secret):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'DeltaForwardBot/1.0'})
    
    def _sign(self, method, path, body=''):
        timestamp = str(int(time.time()))
        msg = method + timestamp + path + body
        sig = hmac.new(self.api_secret.encode(), msg.encode(), hashlib.sha256).hexdigest()
        return {
            'api-key': self.api_key,
            'signature': sig,
            'timestamp': timestamp,
            'Content-Type': 'application/json',
            'User-Agent': 'DeltaForwardBot/1.0',
        }
    
    def _get(self, path, params=None):
        url = self.base_url + path
        if params:
            qs = '&'.join(f"{k}={v}" for k, v in params.items())
            full_path = path + '?' + qs
        else:
            full_path = path
        headers = self._sign('GET', full_path)
        r = self.session.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()
    
    def _post(self, path, data):
        url = self.base_url + path
        body = json.dumps(data)
        headers = self._sign('POST', path, body)
        r = self.session.post(url, data=body, headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()
    
    def get_products(self):
        return self._get('/v2/products')
    
    def get_ticker(self, product_id):
        return self._get(f'/v2/tickers/{product_id}')
    
    def get_candles(self, symbol, resolution='1m', start=None, end=None):
        params = {'resolution': resolution, 'symbol': symbol}
        if start: params['start'] = int(start)
        if end: params['end'] = int(end)
        return self._get('/v2/history/candles', params)
    
    def place_order(self, product_id, size, side, order_type='market_order',
                    limit_price=None, post_only=False):
        data = {
            'product_id': int(product_id),
            'size': int(size),
            'side': side,
            'order_type': order_type,
            'post_only': 'true' if post_only else 'false',
        }
        if limit_price:
            data['limit_price'] = str(limit_price)
        return self._post('/v2/orders', data)
    
    def get_positions(self):
        return self._get('/v2/positions')
    
    def get_wallet(self):
        return self._get('/v2/wallet/balances')
    
    def cancel_all(self, product_id):
        return self._delete('/v2/orders/all', {'product_id': product_id})
    
    def _delete(self, path, params=None):
        url = self.base_url + path
        if params:
            qs = '&'.join(f"{k}={v}" for k, v in params.items())
            full_path = path + '?' + qs
        else:
            full_path = path
        headers = self._sign('DELETE', full_path)
        r = self.session.delete(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()


# === INDICATORS ===
def ema(arr, n):
    alpha = 2 / (n + 1)
    result = np.empty_like(arr, dtype=float)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        if np.isnan(arr[i]):
            result[i] = result[i-1]
        else:
            result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
    return result

def sma(arr, n):
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(n-1, len(arr)):
        result[i] = np.mean(arr[i-n+1:i+1])
    return result

def rsi_calc(arr, n=14):
    result = np.full_like(arr, 50.0, dtype=float)
    if len(arr) < n + 1:
        return result
    deltas = np.diff(arr)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[:n])
    avg_loss = np.mean(losses[:n])
    
    for i in range(n, len(deltas)):
        avg_gain = (avg_gain * (n-1) + gains[i]) / n
        avg_loss = (avg_loss * (n-1) + losses[i]) / n
        if avg_loss == 0:
            result[i+1] = 100
        else:
            rs = avg_gain / avg_loss
            result[i+1] = 100 - (100 / (1 + rs))
    return result


def compute_vwap(candles):
    """Compute VWAP from candle list. candles = [(ts, o, h, l, c, v), ...]"""
    if not candles:
        return np.nan
    c = np.array([x[4] for x in candles], dtype=float)
    h = np.array([x[2] for x in candles], dtype=float)
    l = np.array([x[3] for x in candles], dtype=float)
    v = np.array([x[5] for x in candles], dtype=float)
    tp = (h + l + c) / 3
    cum_tp_vol = np.cumsum(tp * v)
    cum_vol = np.cumsum(v)
    with np.errstate(divide='ignore', invalid='ignore'):
        vwap = np.where(cum_vol > 0, cum_tp_vol / cum_vol, np.nan)
    return vwap[-1] if len(vwap) > 0 else np.nan


# === STRATEGY ENGINE ===
def check_entry(state, candle_1m, candle_15m_prev):
    """
    Check if we should enter a trade.
    candle_1m = current 1min candle (ts, o, h, l, c, v)
    candle_15m_prev = list of recent 15min candles
    Returns: (direction, entry_price, stop, target) or None
    """
    if len(state.candles_1m) < 30 or len(state.candles_15m) < 25:
        return None
    
    # 15min trend: EMA9 vs EMA21
    closes_15m = np.array([x[4] for x in state.candles_15m], dtype=float)
    ema9_15 = ema(closes_15m, 9)
    ema21_15 = ema(closes_15m, 21)
    
    if np.isnan(ema9_15[-1]) or np.isnan(ema21_15[-1]):
        return None
    
    trend_up = ema9_15[-1] > ema21_15[-1]
    
    # 1min VWAP
    vwap = compute_vwap(list(state.candles_1m))
    if np.isnan(vwap):
        return None
    
    # 1min RSI
    closes_1m = np.array([x[4] for x in state.candles_1m], dtype=float)
    rsi_vals = rsi_calc(closes_1m, 14)
    rsi_val = rsi_vals[-1]
    
    # 1min ATR
    highs = np.array([x[2] for x in state.candles_1m], dtype=float)
    lows = np.array([x[3] for x in state.candles_1m], dtype=float)
    trigs = np.maximum(highs[1:] - lows[1:], 
                       np.maximum(np.abs(highs[1:] - closes_1m[:-1]),
                                  np.abs(lows[1:] - closes_1m[:-1])))
    atr_val = trigs[-14:].mean() if len(trigs) >= 14 else trigs.mean()
    
    price = candle_1m[4]  # close
    o = candle_1m[1]      # open
    vol = candle_1m[5]
    
    # Volume filter
    vols = np.array([x[5] for x in state.candles_1m], dtype=float)
    vol_avg = sma(vols, 20)
    vol_ok = vol > vol_avg[-1] * 0.8 if not np.isnan(vol_avg[-1]) else True
    if not vol_ok:
        return None
    
    # VWAP distance
    vwap_dist = abs(price - vwap) / price
    
    # LONG
    if (trend_up and
        vwap_dist < 0.0005 and
        40 < rsi_val < 55 and
        price > o and  # green candle
        price > vwap * 0.999):
        
        stop_dist = max(atr_val * 2, price * STOP_PCT * 0.5)
        stop_dist = min(stop_dist, price * STOP_PCT)
        stop = price - stop_dist
        target = price + stop_dist * 1.5
        return ('long', price, stop, target)
    
    # SHORT
    if (not trend_up and
        vwap_dist < 0.0005 and
        45 < rsi_val < 60 and
        price < o and  # red candle
        price < vwap * 1.001):
        
        stop_dist = max(atr_val * 2, price * STOP_PCT * 0.5)
        stop_dist = min(stop_dist, price * STOP_PCT)
        stop = price + stop_dist
        target = price - stop_dist * 1.5
        return ('short', price, stop, target)
    
    return None


def check_exit(state, candle_1m):
    """Check if position should be closed. Returns (exit_price, reason) or None."""
    if state.position is None:
        return None
    
    h = candle_1m[2]
    l = candle_1m[3]
    c = candle_1m[4]
    
    pos = state.position
    bars = pos.get('bars_held', 0)
    
    if pos['direction'] == 'long':
        if l <= pos['stop']:
            return (pos['stop'], 'stop')
        if h >= pos['target']:
            return (pos['target'], 'target')
        if bars >= 15:
            return (c, 'timeout')
        # Breakeven after 5 bars
        if bars >= 5:
            be_stop = pos['entry'] * (1 + FEE_PER_SIDE * 2.5)
            if be_stop > pos['stop']:
                pos['stop'] = be_stop
    else:  # short
        if h >= pos['stop']:
            return (pos['stop'], 'stop')
        if l <= pos['target']:
            return (pos['target'], 'target')
        if bars >= 15:
            return (c, 'timeout')
        if bars >= 5:
            be_stop = pos['entry'] * (1 - FEE_PER_SIDE * 2.5)
            if be_stop < pos['stop']:
                pos['stop'] = be_stop
    
    return None


# === MAIN BOT LOOP ===
class ForwardBot:
    def __init__(self):
        self.api = None
        self.states = {}
        self.running = False
        self.product_map = {}  # symbol -> product_id
        self.mode = 'testnet'  # or 'production'
        self.log_buffer = deque(maxlen=200)
        self.start_time = datetime.now(timezone.utc).isoformat()
    
    def setup(self):
        """Initialize API and discover products."""
        # Read from env
        api_key = os.environ.get('DELTA_API_KEY', '')
        api_secret = os.environ.get('DELTA_API_SECRET', '')
        self.mode = os.environ.get('DELTA_MODE', 'testnet')
        
        if self.mode == 'production':
            base_url = PROD_URL
        else:
            base_url = TESTNET_URL
        
        if not api_key or not api_secret:
            log.warning("No API keys found. Running in MONITOR ONLY mode (no trades).")
            self.api = None
        else:
            self.api = DeltaAPI(base_url, api_key, api_secret)
            log.info(f"API connected to {self.mode}: {base_url}")
        
        # Discover products
        self._discover_products()
        
        # Init states
        for sym, cfg in ASSETS.items():
            pid = self.product_map.get(sym)
            if pid:
                self.states[sym] = AssetState(sym, cfg['capital_inr'], cfg['leverage'])
                log.info(f"  {sym}: product_id={pid}, capital={cfg['capital_inr']} INR, leverage={cfg['leverage']}x")
            else:
                log.warning(f"  {sym}: NOT FOUND on Delta Exchange. Skipping.")
        
        return len(self.states) > 0
    
    def _discover_products(self):
        """Find product IDs for our assets."""
        try:
            if self.api:
                products = self.api.get_products()
            else:
                # Fallback: try public endpoint
                r = requests.get(f"{PROD_URL}/v2/products", timeout=10)
                products = r.json()
            
            for p in products.get('result', []):
                sym = p.get('symbol', '')
                if sym in ASSETS:
                    self.product_map[sym] = p['id']
                    log.info(f"  Found {sym}: id={p['id']}, description={p.get('description', '')}")
        except Exception as e:
            log.error(f"Product discovery failed: {e}")
            # Hardcoded fallbacks (may change - verify on Delta)
            self.product_map = {
                'XAUTUSD': 346,
                'SLVONUSD': 347,
                'DOGEUSD': 172,
            }
    
    def fetch_candles(self, symbol):
        """Fetch latest 1min and 15min candles."""
        if not self.api:
            return
        
        state = self.states[symbol]
        now = int(time.time())
        
        try:
            # 1min candles - last 200 bars
            resp = self.api.get_candles(symbol, resolution='1m', start=now - 200*60, end=now)
            candles = resp.get('result', [])
            if candles:
                # Delta returns: {time, open, high, low, close, volume}
                state.candles_1m.clear()
                for c in candles:
                    state.candles_1m.append((
                        int(c['time']), float(c['open']), float(c['high']),
                        float(c['low']), float(c['close']), float(c['volume'])
                    ))
            
            # 15min candles - last 100 bars
            resp = self.api.get_candles(symbol, resolution='15m', start=now - 100*15*60, end=now)
            candles = resp.get('result', [])
            if candles:
                state.candles_15m.clear()
                for c in candles:
                    state.candles_15m.append((
                        int(c['time']), float(c['open']), float(c['high']),
                        float(c['low']), float(c['close']), float(c['volume'])
                    ))
            
            state.last_update = datetime.now(timezone.utc).isoformat()
            
        except Exception as e:
            err = f"[{symbol}] Candle fetch error: {e}"
            log.error(err)
            state.errors.append(err)
    
    def process_candle(self, symbol):
        """Process a new 1min candle - check entries/exits."""
        state = self.states[symbol]
        if not state.active or len(state.candles_1m) < 2:
            return
        
        candle = state.candles_1m[-1]
        
        # Check exit first
        if state.position is not None:
            state.position['bars_held'] = state.position.get('bars_held', 0) + 1
            exit_result = check_exit(state, candle)
            
            if exit_result:
                exit_price, reason = exit_result
                self._close_position(symbol, exit_price, reason)
        
        # Check entry (only if no position)
        if state.position is None:
            entry = check_entry(state, candle, list(state.candles_15m))
            
            if entry:
                direction, entry_price, stop, target = entry
                self._open_position(symbol, direction, entry_price, stop, target)
    
    def _open_position(self, symbol, direction, entry_price, stop, target):
        """Open a position on Delta Exchange."""
        state = self.states[symbol]
        pid = self.product_map.get(symbol)
        
        # Calculate size: 50% of max leverage
        notional_inr = state.capital * state.leverage * 0.5
        
        # For crypto: size is in contracts. For now use notional / price.
        # Delta uses "size" in contract units - need to check per product
        # For a first pass, we'll use market orders with notional
        
        if self.api:
            try:
                side = 'buy' if direction == 'long' else 'sell'
                # Delta size is in units. For DOGE this might be in DOGE.
                # For XAUT it might be in grams. Need to check product specs.
                # For now: size = notional / entry_price (rough)
                size = max(1, int(notional_inr / entry_price))
                
                resp = self.api.place_order(
                    product_id=pid,
                    size=size,
                    side=side,
                    order_type='market_order'
                )
                log.info(f"[{symbol}] ORDER PLACED: {direction} {size} units @ ~{entry_price:.4f}")
            except Exception as e:
                err = f"[{symbol}] Order error: {e}"
                log.error(err)
                state.errors.append(err)
                return
        
        state.position = {
            'direction': direction,
            'entry': entry_price,
            'stop': stop,
            'target': target,
            'notional': notional_inr,
            'entry_time': datetime.now(timezone.utc).isoformat(),
            'bars_held': 0,
        }
        
        log.info(f"[{symbol}] OPEN {direction.upper()} @ {entry_price:.4f} | Stop={stop:.4f} Target={target:.4f}")
    
    def _close_position(self, symbol, exit_price, reason):
        """Close position and record trade."""
        state = self.states[symbol]
        pid = self.product_map.get(symbol)
        pos = state.position
        
        if not pos:
            return
        
        # Calculate PnL
        if pos['direction'] == 'long':
            pnl_pct = (exit_price - pos['entry']) / pos['entry']
        else:
            pnl_pct = (pos['entry'] - exit_price) / pos['entry']
        
        pnl_pct *= state.leverage
        pnl_inr = pnl_pct * state.capital
        fees = pos['notional'] * FEE_PER_SIDE * 2
        pnl_net = pnl_inr - fees
        
        # Close on exchange
        if self.api:
            try:
                close_side = 'sell' if pos['direction'] == 'long' else 'buy'
                size = max(1, int(pos['notional'] / pos['entry']))
                self.api.place_order(
                    product_id=pid,
                    size=size,
                    side=close_side,
                    order_type='market_order',
                    reduce_only=True
                )
            except Exception as e:
                log.error(f"[{symbol}] Close order error: {e}")
        
        state.capital += pnl_net
        state.total_pnl += pnl_net
        
        trade = {
            'direction': pos['direction'],
            'entry': pos['entry'],
            'exit': exit_price,
            'pnl': round(pnl_net, 4),
            'reason': reason,
            'bars': pos['bars_held'],
            'time': datetime.now(timezone.utc).isoformat(),
        }
        state.trades.append(trade)
        
        if pnl_net > 0:
            state.win_count += 1
        else:
            state.loss_count += 1
        
        log.info(f"[{symbol}] CLOSE {pos['direction'].upper()} @ {exit_price:.4f} | PnL={pnl_net:+.2f} INR | Reason={reason} | Capital={state.capital:.2f}")
        
        state.position = None
    
    def run(self):
        """Main loop."""
        log.info("=" * 60)
        log.info("FORWARD TEST BOT STARTING")
        log.info("=" * 60)
        
        if not self.setup():
            log.error("No products found. Exiting.")
            return
        
        self.running = True
        cycle = 0
        
        while self.running:
            cycle += 1
            
            for symbol, state in self.states.items():
                if not state.active:
                    continue
                
                try:
                    self.fetch_candles(symbol)
                    self.process_candle(symbol)
                except Exception as e:
                    err = f"[{symbol}] Cycle error: {e}"
                    log.error(err)
                    state.errors.append(err)
            
            # Log status every 10 cycles
            if cycle % 10 == 0:
                for sym, state in self.states.items():
                    wr = state.win_count / max(1, len(state.trades)) * 100
                    log.info(f"[{sym}] Capital={state.capital:.2f} PnL={state.total_pnl:+.2f} Trades={len(state.trades)} WR={wr:.0f}% Pos={'YES' if state.position else 'NO'}")
            
            time.sleep(CANDLE_INTERVAL_SEC)
    
    def stop(self):
        self.running = False


# === FLASK WEB SERVER ===
bot = ForwardBot()

def create_app():
    from flask import Flask, jsonify
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        return jsonify({
            'status': 'running',
            'bot': 'Delta Forward Test',
            'mode': bot.mode,
            'uptime_since': bot.start_time,
            'assets': {sym: s.to_dict() for sym, s in bot.states.items()},
        })
    
    @app.route('/status')
    def status():
        return jsonify({
            'status': 'ok',
            'mode': bot.mode,
            'uptime': bot.start_time,
            'assets': {sym: s.to_dict() for sym, s in bot.states.items()},
        })
    
    @app.route('/trades')
    def trades():
        all_trades = {}
        for sym, state in bot.states.items():
            all_trades[sym] = state.trades[-20:]
        return jsonify(all_trades)
    
    @app.route('/health')
    def health():
        return jsonify({'status': 'healthy'})
    
    return app


# === ENTRY POINT ===
if __name__ == '__main__':
    import sys
    
    # Start bot in background thread
    bot_thread = threading.Thread(target=bot.run, daemon=True)
    bot_thread.start()
    
    # Start Flask server
    app = create_app()
    port = int(os.environ.get('PORT', 8080))
    log.info(f"Flask server starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
