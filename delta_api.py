"""
Delta Exchange India REST API Client
Production: https://api.india.delta.exchange
Testnet: https://cdn-ind.testnet.deltaex.org
"""
import hmac
import hashlib
import time
import json
import requests


class DeltaClient:
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
        }
    
    def _get(self, path, params=None):
        url = self.base_url + path
        if params:
            qs = '&'.join(f"{k}={v}" for k, v in params.items())
            path_with_qs = path + '?' + qs
        else:
            path_with_qs = path
        
        headers = self._sign('GET', path_with_qs)
        r = self.session.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()
    
    def _post(self, path, data):
        url = self.base_url + path
        body = json.dumps(data)
        headers = self._sign('POST', path, body)
        headers['Content-Type'] = 'application/json'
        r = self.session.post(url, data=body, headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()
    
    def _delete(self, path, params=None):
        url = self.base_url + path
        if params:
            qs = '&'.join(f"{k}={v}" for k, v in params.items())
            path_with_qs = path + '?' + qs
        else:
            path_with_qs = path
        headers = self._sign('DELETE', path_with_qs)
        r = self.session.delete(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()
    
    # === Market Data (public) ===
    
    def get_products(self):
        return self._get('/v2/products')
    
    def get_product(self, product_id):
        return self._get(f'/v2/products/{product_id}')
    
    def get_candles(self, product_id, resolution='1m', start=None, end=None):
        params = {'resolution': resolution, 'product_id': product_id}
        if start: params['start'] = int(start)
        if end: params['end'] = int(end)
        return self._get('/v2/history/candles', params)
    
    def get_ticker(self, product_id):
        return self._get(f'/v2/tickers/{product_id}')
    
    def get_l2_orderbook(self, product_id):
        return self._get('/v2/l2orderbook', {'product_id': product_id})
    
    # === Account (private) ===
    
    def get_wallet(self):
        return self._get('/v2/wallet/balances')
    
    def get_positions(self):
        return self._get('/v2/positions')
    
    def get_position(self, product_id):
        return self._get('/v2/positions', {'product_id': product_id})
    
    # === Orders (private) ===
    
    def place_order(self, product_id, size, side, order_type='market_order',
                    limit_price=None, stop_price=None, time_in_force='gtc',
                    post_only=False, reduce_only=False):
        data = {
            'product_id': int(product_id),
            'size': int(size),
            'side': side,
            'order_type': order_type,
            'time_in_force': time_in_force,
            'post_only': 'true' if post_only else 'false',
            'reduce_only': 'true' if reduce_only else 'false',
        }
        if limit_price:
            data['limit_price'] = str(limit_price)
        if stop_price:
            data['stop_price'] = str(stop_price)
        return self._post('/v2/orders', data)
    
    def cancel_order(self, order_id, product_id):
        return self._delete('/v2/orders', {'id': order_id, 'product_id': product_id})
    
    def cancel_all_orders(self, product_id):
        return self._delete('/v2/orders/all', {'product_id': product_id})
    
    def get_open_orders(self, product_id=None):
        params = {}
        if product_id:
            params['product_id'] = product_id
        return self._get('/v2/orders', params)
    
    # === Leverage ===
    
    def set_leverage(self, product_id, leverage):
        return self._post('/v2/orders/leverage', {
            'product_id': int(product_id),
            'leverage': str(leverage),
        })
    
    # === Position Management ===
    
    def close_position(self, product_id):
        """Close all positions for a product"""
        positions = self.get_positions()
        for pos in positions.get('result', []):
            if pos.get('product_id') == int(product_id) and pos.get('size', 0) != 0:
                side = 'sell' if pos['size'] > 0 else 'buy'
                return self.place_order(
                    product_id=product_id,
                    size=abs(pos['size']),
                    side=side,
                    order_type='market_order',
                    reduce_only=True
                )
        return None
