from __future__ import annotations

import hashlib, hmac, os, threading, time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Any
from urllib.parse import urlencode

import numpy as np
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template
from flask_cors import CORS

load_dotenv(); app = Flask(__name__); CORS(app)
SYMBOLS = [x.strip().upper() for x in os.getenv('SYMBOLS','BTCUSDT,ETHUSDT,BNBUSDT,ADAUSDT,DOGEUSDT').split(',') if x.strip()]
# Binance Spot REST klines do not support 1-second candles. 1s is a live polling view.
SUPPORTED = ['1s','1m','3m','5m','15m','30m','1h','2h','4h']
INTERVALS = [x.strip() for x in os.getenv('TIMEFRAMES',','.join(SUPPORTED)).split(',') if x.strip() in SUPPORTED]
TRADE_TF = os.getenv('TRADE_TIMEFRAME','15m'); TRADE_TF = TRADE_TF if TRADE_TF in SUPPORTED else '15m'
CFG = {'initial':float(os.getenv('INITIAL_CAPITAL','800')), 'risk':float(os.getenv('RISK_PER_TRADE','0.10')), 'rr':float(os.getenv('RISK_REWARD_RATIO','3')), 'max_loss':float(os.getenv('MAX_DAILY_LOSS','0.05')), 'max_pos':int(os.getenv('MAX_OPEN_POSITIONS','1')), 'score':int(os.getenv('MIN_SIGNAL_SCORE','75')), 'poll':max(1,int(os.getenv('POLL_SECONDS','15'))), 'limit':min(1000,max(210,int(os.getenv('CANDLE_LIMIT','250'))))}
KEY, SECRET = os.getenv('BINANCE_API_KEY',''), os.getenv('BINANCE_API_SECRET','')
TESTNET = os.getenv('BINANCE_TESTNET','true').lower() == 'true'; ORDERS = os.getenv('ENABLE_ORDERS','false').lower() == 'true'
BASE = os.getenv('BINANCE_BASE_URL','https://testnet.binance.vision/api' if TESTNET else 'https://api.binance.com/api')
state: dict[str,Any] = {'balance':CFG['initial'],'initial_balance':CFG['initial'],'is_trading':False,'total_trades':0,'winning_trades':0,'losing_trades':0,'total_profit':0.0,'daily_loss':0.0,'open_positions':[],'closed_trades':[],'last_error':None,'last_update':datetime.now(timezone.utc).isoformat(),'mode':'TESTNET' if TESTNET else 'LIVE','orders_enabled':ORDERS}
lock=threading.RLock(); http=requests.Session(); rules_cache={}; seen={}

def now(): return datetime.now(timezone.utc).isoformat()
def api(method,path,params=None,signed=False):
    p=dict(params or {})
    if signed:
        if not KEY or not SECRET: raise RuntimeError('BINANCE_API_KEY and BINANCE_API_SECRET are required')
        p.update(timestamp=int(time.time()*1000),recvWindow=5000); q=urlencode(p); p['signature']=hmac.new(SECRET.encode(),q.encode(),hashlib.sha256).hexdigest()
    r=http.request(method,BASE+path,params=p,headers={'X-MBX-APIKEY':KEY} if KEY else {},timeout=15)
    if not r.ok: raise RuntimeError(f'Binance {r.status_code}: {r.text[:250]}')
    return r.json()

def candles(symbol, tf):
    # No 1-second REST candle exists. Use the latest 1-minute closed candle set for 1s view.
    source='1m' if tf=='1s' else tf
    raw=api('GET','/v3/klines',{'symbol':symbol,'interval':source,'limit':CFG['limit']})
    if len(raw)<205: raise RuntimeError(f'Not enough {source} candles for {symbol}')
    return np.array([[float(k[1]),float(k[2]),float(k[3]),float(k[4]),float(k[5]),int(k[0])] for k in raw[:-1]],dtype=float)

def ema(v,n):
    a=2/(n+1); out=float(v[0])
    for x in v[1:]: out=a*float(x)+(1-a)*out
    return out

def rsi(v,n=14):
    d=np.diff(v)[-n:]; g=np.mean(np.maximum(d,0)); l=np.mean(np.maximum(-d,0)); return 100.0 if l==0 else float(100-100/(1+g/l))

def atr(c,n=14):
    return float(np.mean(np.maximum(c[-n:,1]-c[-n:,2],np.maximum(abs(c[-n:,1]-c[-n-1:-1,3]),abs(c[-n:,2]-c[-n-1:-1,3])))))

def reading(c):
    o,h,l,x=c[-1,:4]; po,_,_,px=c[-2,:4]; body=abs(x-o); full=max(h-l,1e-12); up=h-max(o,x); lo=min(o,x)-l
    be=px<po and x>o and x>=po and o<=px; se=px>po and x<o and o>=px and x<=po
    bp=lo>=max(body,full*.01)*2 and up<=body; sp=up>=max(body,full*.01)*2 and lo<=body
    name='bullish_engulfing' if be else 'bearish_engulfing' if se else 'bullish_pinbar' if bp and x>o else 'bearish_pinbar' if sp and x<o else 'none'
    return {'pattern':name,'bullish':bool(be or (bp and x>o)),'bearish':bool(se or (sp and x<o)),'body_ratio':round(body/full,3)}

def smc(c):
    hi,lo,x=c[:,1],c[:,2],c[:,3]; sh=float(np.max(hi[-30:-3])); sl=float(np.min(lo[-30:-3])); p=float(x[-1])
    return {'swing_high':sh,'swing_low':sl,'bos':'BULLISH' if p>sh else 'BEARISH' if p<sl else 'NONE','liquidity_sweep':'LOW' if lo[-1]<sl and p>sl else 'HIGH' if hi[-1]>sh and p<sh else 'NONE'}

def analyze(symbol,tf=TRADE_TF):
    c=candles(symbol,tf); close=c[:,3]; p=float(close[-1]); st=smc(c); pat=reading(c); e20,e50,e200=ema(close,20),ema(close,50),ema(close,200); rv=rsi(close); score=0; why=[]
    checks=[(e20>e50>e200,25,'EMA trend bullish'),(p>e20,10,'price above EMA20'),(st['bos']=='BULLISH',25,'bullish BOS'),(st['liquidity_sweep']=='LOW',20,'sell-side liquidity sweep'),(pat['bullish'],20,pat['pattern']),(50<=rv<=70,10,'RSI confirmation')]
    for ok,points,text in checks:
        if ok: score+=points; why.append(text)
    signal='BUY' if score>=CFG['score'] else 'NEUTRAL'; distance=max(atr(c)*1.5,p*.001); stop=min(p-distance,st['swing_low']*.999) if signal=='BUY' else p-distance; risk=max(p-stop,p*.001)
    return {'symbol':symbol,'timeframe':tf,'source_timeframe':'1m' if tf=='1s' else tf,'price':round(p,8),'rsi':round(rv,2),'ema20':round(e20,8),'ema50':round(e50,8),'ema200':round(e200,8),'signal':signal,'strength':min(score,100),'reasons':why,'candle':pat,'market_structure':st,'stop_loss':round(stop,8),'take_profit':round(p+risk*CFG['rr'],8),'risk_reward':CFG['rr'],'candle_time':int(c[-1,5])}

def qty_rules(symbol):
    if symbol not in rules_cache:
        info=api('GET','/v3/exchangeInfo',{'symbol':symbol})['symbols'][0]; fs={f['filterType']:f for f in info['filters']}; lot=fs.get('LOT_SIZE',{}); no=fs.get('NOTIONAL',fs.get('MIN_NOTIONAL',{})); rules_cache[symbol]={'step':float(lot.get('stepSize','0.000001')),'min':float(lot.get('minQty','0.000001')),'notional':float(no.get('minNotional','5'))}
    return rules_cache[symbol]
def floor_step(v,step): return float((Decimal(str(v))/Decimal(str(step))).to_integral_value(rounding=ROUND_DOWN)*Decimal(str(step)))
def order(symbol,side,q): return api('POST','/v3/order',{'symbol':symbol,'side':side,'type':'MARKET','quantity':f'{q:.8f}','newOrderRespType':'RESULT'},True)

def open_pos(a):
    with lock:
        if a['signal']!='BUY' or len(state['open_positions'])>=CFG['max_pos'] or any(x['symbol']==a['symbol'] for x in state['open_positions']): return
        q=CFG['balance']*CFG['risk']/max(a['price']-a['stop_loss'],a['price']*.001)
        if ORDERS:
            rr=qty_rules(a['symbol']); q=max(floor_step(q,rr['step']),rr['min'])
            if q*a['price']<rr['notional']: raise RuntimeError(f'Order below {rr["notional"]} USDT minimum')
            ex=order(a['symbol'],'BUY',q)
        else: ex={'status':'SIGNAL_ONLY'}
        state['open_positions'].append({'symbol':a['symbol'],'side':'BUY','entry_price':a['price'],'quantity':q,'stop_loss':a['stop_loss'],'take_profit':a['take_profit'],'order':ex,'opened_at':now()})

def resolve():
    with lock:
        left=[]
        for p in state['open_positions']:
            try: price=float(api('GET','/v3/ticker/price',{'symbol':p['symbol']})['price'])
            except Exception: left.append(p); continue
            reason='TAKE_PROFIT' if price>=p['take_profit'] else 'STOP_LOSS' if price<=p['stop_loss'] else None
            if not reason: left.append(p); continue
            if ORDERS: order(p['symbol'],'SELL',floor_step(p['quantity'],qty_rules(p['symbol'])['step']))
            pnl=(p['take_profit'] if reason=='TAKE_PROFIT' else p['stop_loss']-p['entry_price'])*p['quantity'] if False else ((p['take_profit'] if reason=='TAKE_PROFIT' else p['stop_loss'])-p['entry_price'])*p['quantity']
            state['balance']+=pnl; state['total_profit']+=pnl; state['total_trades']+=1
            if pnl>=0: state['winning_trades']+=1
            else: state['losing_trades']+=1; state['daily_loss']+=abs(pnl)
            state['closed_trades'].append({'symbol':p['symbol'],'pnl':round(pnl,4),'status':'WIN' if pnl>=0 else 'LOSS','reason':reason,'time':now()})
        state['open_positions']=left

def loop():
    while True:
        try:
            if state['is_trading'] and state['daily_loss']<CFG['initial']*CFG['max_loss']:
                for s in SYMBOLS:
                    a=analyze(s); k=(s,TRADE_TF)
                    if seen.get(k)!=a['candle_time']: seen[k]=a['candle_time']; open_pos(a)
                resolve(); state['last_error']=None; state['last_update']=now()
            time.sleep(CFG['poll'])
        except Exception as e: state['last_error']=str(e); print('trading loop error:',e); time.sleep(CFG['poll'])
threading.Thread(target=loop,daemon=True).start()

@app.get('/')
def index(): return render_template('index.html')
@app.get('/health')
def health(): return jsonify({'ok':True,'mode':state['mode'],'timeframes':INTERVALS,'trade_timeframe':TRADE_TF,'one_second_note':'1s uses live polling with 1m closed candles; Binance REST has no 1s klines'})
@app.post('/api/trading/start')
def start(): state['is_trading']=True; return jsonify({'is_trading':True,'mode':state['mode'],'orders_enabled':ORDERS})
@app.post('/api/trading/stop')
def stop(): state['is_trading']=False; return jsonify({'is_trading':False})
@app.get('/api/stats')
def stats():
    with lock:
        total=state['total_trades']; return jsonify({**state,'open_positions':len(state['open_positions']),'win_rate':round(state['winning_trades']/total*100,2) if total else 0,'profit_percent':round((state['balance']/state['initial_balance']-1)*100,2),'risk_per_trade':CFG['risk'],'risk_reward':CFG['rr']})
@app.get('/api/positions')
def positions(): return jsonify({'open_positions':state['open_positions'],'closed_trades':state['closed_trades'][-50:]})
@app.get('/api/analyze/<symbol>')
def analyze_route(symbol):
    try: return jsonify(analyze(symbol.upper()))
    except Exception as e: return jsonify({'error':str(e),'symbol':symbol.upper()}),502
@app.get('/api/analyze-all')
def all_analysis():
    out={}
    for s in SYMBOLS:
        try: out[s]=analyze(s)
        except Exception as e: out[s]={'symbol':s,'error':str(e)}
    return jsonify(out)
@app.get('/api/timeframes')
def timeframes():
    out={'trade_timeframe':TRADE_TF,'supported':INTERVALS,'data':{}}
    for tf in INTERVALS:
        try: out['data'][tf]=analyze(SYMBOLS[0],tf)
        except Exception as e: out['data'][tf]={'timeframe':tf,'error':str(e)}
    return jsonify(out)
if __name__=='__main__': app.run(host='0.0.0.0',port=int(os.getenv('PORT','5000')))
