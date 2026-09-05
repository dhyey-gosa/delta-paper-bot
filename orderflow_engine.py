"""
Delta Exchange Orderflow Engine
Connects to Delta's public WebSocket for real-time:
- Trade stream (CVD, aggression, large trades)
- L2 orderbook (imbalance, sweep detection, absorption)

Data sources:
- wss://public-socket.india.delta.exchange (public, no auth)
- Channels: trades, ob_updates

No API key needed. Free public data.
"""
import sys
import json
import time
import threading
import logging
from collections import deque
from datetime import datetime, timezone

import numpy as np
import websocket

log = logging.getLogger('orderflow')

WS_URL = "wss://public-socket.india.delta.exchange"

# Rolling windows (seconds)
CVD_WINDOWS = [5, 15, 30, 60]
TRADE_HISTORY_SEC = 300  # keep 5 min of trades

# Signal thresholds
CVD_SPIKE_PCT = 0.70      # CVD ratio > 0.70 in any window = aggression
ABSORPTION_VOL_MULT = 3.0  # trade volume > 3x avg = large trade
ABSORPTION_PRICE_MOVE = 0.0005  # price moved < 0.05% after large trades = absorption
ORDERBOOK_IMBALANCE = 0.65  # 65%+ one-sided depth = imbalance
SWEEP_DEPTH_PCT = 0.002     # orderbook sweep: price pierces 0.2% into opposite side


class DeltaOrderflow:
    """Real-time orderflow from Delta Exchange public WebSocket."""

    def __init__(self, symbols):
        """
        Args:
            symbols: list of symbol strings e.g. ['XAUTUSD', 'SLVONUSD']
        """
        self.symbols = [s.upper() for s in symbols]
        self.ws = None
        self.ws_thread = None
        self.running = False
        self.connected = False

        # Trade storage: {symbol: deque of (timestamp, price, qty, side)}
        self.trades = {s: deque(maxlen=5000) for s in self.symbols}

        # Orderbook: {symbol: {'bids': [(price, qty), ...], 'asks': [(price, qty), ...], 'ts': timestamp}}
        self.orderbook = {s: {'bids': [], 'asks': [], 'ts': 0} for s in self.symbols}

        # Cached signals (computed every tick)
        self.signals = {s: {} for s in self.symbols}

        # Last price from trade stream
        self.last_price = {s: 0.0 for s in self.symbols}

        # Debug counter
        self._debug_count = 0
        self._msg_counts = {}

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return

        msg_type = data.get('type', '')

        # Debug: log every message type we receive (first 5 of each)
        self._msg_counts[msg_type] = self._msg_counts.get(msg_type, 0) + 1
        if self._msg_counts[msg_type] <= 5:
            log.info(f"[Orderflow] MSG type={msg_type} keys={list(data.keys())} sample={str(data)[:300]}")

        if msg_type in ('trades', 'all_trades', 'all_trades_snapshot'):
            self._process_trades(data)
        elif msg_type in ('ob_updates', 'l2_updates'):
            self._process_l2(data)
        elif msg_type in ('ob_l2', 'l2_orderbook'):
            self._process_l2_snapshot(data)
        elif msg_type in ('subscribed', 'unsubscribed'):
            log.info(f"WS event: {msg_type}")
        elif msg_type == 'error':
            log.error(f"WS error: {data}")

    def _process_trades(self, data):
        # Delta trades channel sends individual trades: {"p":"price","r":"m/t","s":qty,"sy":"SYMBOL","type":"trades"}
        # Also handle batch format: {"result": [...trades...]}
        results = data.get('result', data.get('trades', []))
        if not isinstance(results, list):
            results = []

        # Individual trade (common format on this channel)
        if data.get('type') == 'trades' and 'sy' in data:
            results = [data]

        now = time.time()

        for t in results:
            symbol = t.get('sy', t.get('symbol', '')).upper()
            if symbol not in self.trades:
                continue

            price = float(t.get('p', t.get('price', t.get('fill_price', 0))))
            qty = float(t.get('s', t.get('size', t.get('qty', 0))))

            # Delta 'r' field: "m" = maker was seller (buyer taker = aggressive buy)
            #                   "t" = taker was seller (seller taker = aggressive sell)
            role = t.get('r', t.get('side', ''))
            if role == 'm':
                aggression = 'buy'   # buyer was taker
            elif role == 't':
                aggression = 'sell'  # seller was taker
            elif 'side' in t:
                aggression = t['side'].lower()
            elif 'seller' in t:
                aggression = 'buy' if t.get('seller') in (True, 'True', 'true') else 'sell'
            else:
                aggression = 'buy'

            self.trades[symbol].append((now, price, qty, aggression))
            self.last_price[symbol] = price

            if self._debug_count < 20:
                log.info(f"[Orderflow] TRADE {symbol} {aggression} {qty}@{price} role={role}")
                self._debug_count += 1

        # Update signals after batch
        symbols_hit = set()
        for t in results:
            s = t.get('sy', t.get('symbol', '')).upper()
            if s in self.symbols:
                symbols_hit.add(s)
        for symbol in symbols_hit:
            self._compute_signals(symbol)

    def _process_l2(self, data):
        # Delta ob_updates format: {"a":[[price,qty],...], "b":[[price,qty],...], "sy":"SYMBOL", "type":"ob_updates"}
        symbol = data.get('sy', data.get('symbol', '')).upper()
        if symbol not in self.orderbook:
            return

        # 'b' = bids, 'a' = asks
        bids = data.get('b', data.get('buy', []))
        asks = data.get('a', data.get('sell', []))

        if bids:
            self.orderbook[symbol]['bids'] = [(float(p), float(q)) for p, q in bids]
        if asks:
            self.orderbook[symbol]['asks'] = [(float(p), float(q)) for p, q in asks]
        self.orderbook[symbol]['ts'] = time.time()

        self._compute_signals(symbol)

    def _process_l2_snapshot(self, data):
        results = data.get('result', [])
        if not isinstance(results, list):
            return
        for entry in results:
            symbol = entry.get('symbol', '').upper()
            if symbol not in self.orderbook:
                continue
            buy = entry.get('buy', [])
            sell = entry.get('sell', [])
            if buy:
                self.orderbook[symbol]['bids'] = [(float(p), float(q)) for p, q in buy]
            if sell:
                self.orderbook[symbol]['asks'] = [(float(p), float(q)) for p, q in sell]
            self.orderbook[symbol]['ts'] = time.time()

    def _compute_signals(self, symbol):
        """Compute all orderflow signals for a symbol."""
        now = time.time()
        sig = {'ts': now}

        # --- CVD (Cumulative Volume Delta) ---
        trade_list = list(self.trades[symbol])
        for window_sec in CVD_WINDOWS:
            cutoff = now - window_sec
            window_trades = [(t, p, q, s) for t, p, q, s in trade_list if t >= cutoff]
            buy_vol = sum(q for _, _, q, s in window_trades if s == 'buy')
            sell_vol = sum(q for _, _, q, s in window_trades if s == 'sell')
            total = buy_vol + sell_vol
            if total > 0:
                cvd_ratio = buy_vol / total  # >0.5 = buy aggression
            else:
                cvd_ratio = 0.5
            sig[f'cvd_{window_sec}s'] = round(cvd_ratio, 4)
            sig[f'buy_vol_{window_sec}s'] = round(buy_vol, 4)
            sig[f'sell_vol_{window_sec}s'] = round(sell_vol, 4)

        # --- Aggression: any window shows >70% one-sided ---
        for window_sec in CVD_WINDOWS:
            ratio = sig.get(f'cvd_{window_sec}s', 0.5)
            if ratio > CVD_SPIKE_PCT:
                sig['aggression'] = 'buy'
                sig['aggression_strength'] = round(ratio, 4)
                sig['aggression_window'] = window_sec
                break
            elif ratio < (1 - CVD_SPIKE_PCT):
                sig['aggression'] = 'sell'
                sig['aggression_strength'] = round(1 - ratio, 4)
                sig['aggression_window'] = window_sec
                break
        else:
            sig['aggression'] = None
            sig['aggression_strength'] = 0

        # --- Absorption: large trades that didn't move price ---
        avg_vol = np.mean([q for _, _, q, _ in trade_list[-50:]]) if len(trade_list) >= 50 else 0
        large_trades = [(t, p, q, s) for t, p, q, s in trade_list[-20:]
                        if q > avg_vol * ABSORPTION_VOL_MULT and avg_vol > 0]
        if len(large_trades) >= 2:
            prices = [p for _, p, _, _ in large_trades]
            price_range = (max(prices) - min(prices)) / min(prices) if min(prices) > 0 else 999
            if price_range < ABSORPTION_PRICE_MOVE:
                # Large trades hitting but price barely moved = absorption
                buy_side = sum(q for _, _, q, s in large_trades if s == 'buy')
                sell_side = sum(q for _, _, q, s in large_trades if s == 'sell')
                if buy_side > sell_side * 2:
                    sig['absorption'] = 'buy'  # sellers absorbed, buyers defending
                elif sell_side > buy_side * 2:
                    sig['absorption'] = 'sell'  # buyers absorbed, sellers defending
                else:
                    sig['absorption'] = None
            else:
                sig['absorption'] = None
        else:
            sig['absorption'] = None

        # --- Orderbook Imbalance ---
        ob = self.orderbook[symbol]
        if ob['bids'] and ob['asks']:
            bid_depth = sum(q for _, q in ob['bids'][:10])
            ask_depth = sum(q for _, q in ob['asks'][:10])
            total_depth = bid_depth + ask_depth
            if total_depth > 0:
                imbalance = bid_depth / total_depth  # >0.5 = more bids
            else:
                imbalance = 0.5
            sig['ob_imbalance'] = round(imbalance, 4)
            sig['bid_depth'] = round(bid_depth, 4)
            sig['ask_depth'] = round(ask_depth, 4)

            # Best bid/ask
            sig['best_bid'] = ob['bids'][0][0] if ob['bids'] else 0
            sig['best_ask'] = ob['asks'][0][0] if ob['asks'] else 0
            sig['spread'] = round(sig['best_ask'] - sig['best_bid'], 6)
            sig['spread_pct'] = round(sig['spread'] / sig['best_bid'] * 100, 4) if sig['best_bid'] > 0 else 0
        else:
            sig['ob_imbalance'] = 0.5

        # --- Sweep Detection ---
        # Did price wick into the opposite side of the book and come back?
        if ob['bids'] and ob['asks'] and len(trade_list) >= 5:
            recent_prices = [p for _, p, _, _ in trade_list[-10:]]
            if recent_prices:
                recent_high = max(recent_prices)
                recent_low = min(recent_prices)
                # Check if recent trades pierced into heavy order zones
                heavy_bid_levels = [(p, q) for p, q in ob['bids'][:5] if q > avg_vol * 2]
                heavy_ask_levels = [(p, q) for p, q in ob['asks'][:5] if q > avg_vol * 2]

                sig['sweep_low'] = False
                sig['sweep_high'] = False
                for bp, bq in heavy_bid_levels:
                    if recent_low <= bp and self.last_price[symbol] > bp:
                        sig['sweep_low'] = True  # swept bids, now above = bullish sweep
                        break
                for ap, aq in heavy_ask_levels:
                    if recent_high >= ap and self.last_price[symbol] < ap:
                        sig['sweep_high'] = True  # swept asks, now below = bearish sweep
                        break

        self.signals[symbol] = sig

    def get_signal(self, symbol):
        """Get current orderflow signal dict for a symbol."""
        return self.signals.get(symbol.upper(), {})

    def get_trade_count(self, symbol, window_sec=60):
        """Count trades in the last N seconds."""
        now = time.time()
        cutoff = now - window_sec
        return sum(1 for t, _, _, _ in self.trades.get(symbol.upper(), []) if t >= cutoff)

    def _ws_loop(self):
        """WebSocket reconnect loop."""
        while self.running:
            try:
                log.info(f"[Orderflow] Connecting to {WS_URL}...")
                self.ws = websocket.WebSocketApp(
                    WS_URL,
                    on_message=self._on_message,
                    on_error=lambda ws, e: log.error(f"[Orderflow] WS error: {e}"),
                    on_close=lambda ws, c, m: log.warning(f"[Orderflow] WS closed: {c} {m}"),
                    on_open=self._on_open,
                )
                self.ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                log.error(f"[Orderflow] Connection error: {e}")
            finally:
                self.connected = False

            if self.running:
                log.info("[Orderflow] Reconnecting in 5s...")
                time.sleep(5)

    def _on_open(self, ws):
        """Subscribe to channels on connect."""
        self.connected = True
        log.info(f"[Orderflow] Connected. Subscribing to {self.symbols}...")

        # Delta public WebSocket: correct channel names from docs
        # https://docs.delta.exchange -> "trades", "ob_updates", "ob_l2"
        # Max 4 symbols per subscription for orderbook channels
        sub = {
            "type": "subscribe",
            "payload": {
                "channels": [
                    {"name": "trades", "symbols": self.symbols},
                    {"name": "ob_updates", "symbols": self.symbols[:4]},
                ]
            }
        }
        ws.send(json.dumps(sub))
        log.info(f"[Orderflow] Sent subscribe: {json.dumps(sub)}")

    def start(self):
        """Start WebSocket in background thread."""
        self.running = True
        self.ws_thread = threading.Thread(target=self._ws_loop, daemon=True, name='orderflow-ws')
        self.ws_thread.start()
        log.info("[Orderflow] Background thread started")

    def stop(self):
        """Stop WebSocket."""
        self.running = False
        if self.ws:
            self.ws.close()

    def is_alive(self):
        """Check if WebSocket is connected."""
        return self.connected


# === CONVENIENCE: Signal summary for strategy ===
def format_signal(sig):
    """Format signal dict for logging."""
    if not sig:
        return "NO SIGNAL"
    parts = []
    agg = sig.get('aggression')
    if agg:
        parts.append(f"AGGR={agg}({sig.get('aggression_strength', 0):.2f})")
    abs_ = sig.get('absorption')
    if abs_:
        parts.append(f"ABS={abs_}")
    imb = sig.get('ob_imbalance', 0.5)
    if imb > ORDERBOOK_IMBALANCE:
        parts.append(f"OB_BULL({imb:.2f})")
    elif imb < (1 - ORDERBOOK_IMBALANCE):
        parts.append(f"OB_BEAR({imb:.2f})")
    if sig.get('sweep_low'):
        parts.append("SWEEP_LOW")
    if sig.get('sweep_high'):
        parts.append("SWEEP_HIGH")
    spread = sig.get('spread_pct', 0)
    if spread > 0.1:
        parts.append(f"WIDE_SPREAD({spread:.3f}%)")
    return ' | '.join(parts) if parts else 'NEUTRAL'
