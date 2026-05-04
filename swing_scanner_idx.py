import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBClassifier
import os
import requests
from datetime import datetime

# =================================================================
# CONFIGURATION
# =================================================================
PROB_THRESHOLD = 0.75      
HOLD_DAYS = 10             
TARGET_PROFIT = 0.07       
MIN_TURNOVER_7B = 7e9      
MIN_PRICE = 100            
MAX_PRICE = 10000          
MASTER_PATH = 'master_dataset_ihsg.csv'
LOG_FILE = 'log_rekomendasi_swing_v3.csv'

# TOKEN & ID dari GitHub Secrets
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

TICKERS_IDX = [
    'BBCA.JK', 'BBRI.JK', 'BMRI.JK', 'BBNI.JK', 'BRIS.JK', 'BBTN.JK', 'ARTO.JK', 'BDMN.JK', 'BNGA.JK',
    'ADRO.JK', 'PTBA.JK', 'ITMG.JK', 'HRUM.JK', 'MEDC.JK', 'ENRG.JK', 'ELSA.JK', 'AKRA.JK', 
    'ANTM.JK', 'INCO.JK', 'MBMA.JK', 'MDKA.JK', 'BRMS.JK', 'NCKL.JK', 'TINS.JK', 'PGEO.JK', 
    'BREN.JK', 'AMMN.JK', 'TLKM.JK', 'ISAT.JK', 'EXCL.JK', 'JSMR.JK', 'PGAS.JK', 'TOWR.JK', 
    'TBIG.JK', 'WIKA.JK', 'PTPP.JK', 'ADHI.JK', 'ICBP.JK', 'INDF.JK', 'UNVR.JK', 'KLBF.JK', 
    'AMRT.JK', 'SIDO.JK', 'MYOR.JK', 'HEAL.JK', 'MIKA.JK', 'SILO.JK', 'BSDE.JK', 'CTRA.JK', 
    'SMRA.JK', 'PWON.JK', 'PANI.JK', 'ASRI.JK', 'SMGR.JK', 'INTP.JK', 'SSIA.JK', 'ACES.JK', 
    'MAPI.JK', 'MAPA.JK', 'ERAA.JK', 'GOTO.JK', 'BUKA.JK', 'BELI.JK', 'ASII.JK', 'AUTO.JK', 
    'ASSA.JK', 'TPIA.JK', 'CUAN.JK', 'FILM.JK', 'AVIA.JK', 'SMDR.JK', 'RAJA.JK', 'ESSA.JK', 'WIIM.JK',
    'LAJU.JK', 'BJBR.JK', 'BAIK.JK', 'TOTL.JK', 'ALII.JK', 'TRIN.JK', 'WMPP.JK', 'ENZO.JK', 
    'DEWI.JK', 'MEJA.JK', 'DFAM.JK', 'BJTM.JK', 'JAYA.JK', 'ULTJ.JK', 'MAIN.JK', 'DIVA.JK', 
    'UVCR.JK', 'MERI.JK', 'SRSN.JK', 'FPNI.JK', 'MSJA.JK', 'ASGR.JK', 'MEDS.JK', 'INCF.JK', 
    'APIC.JK', 'NTBK.JK', 'GULA.JK', 'ICON.JK', 'GSMF.JK', 'MARK.JK', 'GMFI.JK', 'MORA.JK', 
    'POLA.JK', 'APEX.JK', 'BSML.JK', 'KOBX.JK', 'ASLI.JK', 'JGLE.JK', 'OMED.JK', 'DMAS.JK', 
    'ELTY.JK', 'OPMS.JK', 'MSKY.JK', 'GJTL.JK', 'LEAD.JK', 'BIPP.JK', 'CTTH.JK', 'CMNT.JK', 
    'GGRM.JK', 'MBSS.JK', 'KOCI.JK', 'DKFT.JK', 'RMKO.JK', 'OILS.JK', 'PKPK.JK', 'INDS.JK', 
    'RODA.JK', 'AYAM.JK', 'HMSP.JK', 'ASHA.JK', 'RLCO.JK', 'COIN.JK', 'TRUE.JK', 'SRTG.JK', 
    'LAND.JK', 'SOCI.JK', 'TKIM.JK', 'AALI.JK', 'TOOL.JK', 'PADA.JK', 'ASPR.JK', 'PSKT.JK', 
    'BELL.JK', 'CBRE.JK', 'GTSI.JK', 'SINI.JK', 'SDMU.JK', 'DSNG.JK', 'NSSS.JK', 'CYBR.JK', 
    'BMTR.JK', 'KAQI.JK', 'PSDN.JK', 'APLN.JK', 'IRSX.JK', 'YELO.JK', 'HUMI.JK', 'SUPA.JK', 
    'KUAS.JK', 'LSIP.JK', 'JPFA.JK', 'OASA.JK', 'MSIN.JK', 'FORE.JK', 'INKP.JK', 'RMKE.JK', 
    'COCO.JK', 'BFIN.JK', 'HRTA.JK', 'TAPG.JK', 'SIMP.JK', 'PPRE.JK', 'BAPA.JK', 'TCPI.JK', 
    'ARKO.JK', 'WMUU.JK', 'ZATA.JK', 'ESIP.JK', 'ARCI.JK', 'PACK.JK', 'IMPC.JK', 'INDY.JK', 
    'MDIA.JK', 'UNTR.JK', 'PADI.JK', 'MINA.JK', 'ADMR.JK', 'INET.JK', 'VKTR.JK', 'EMAS.JK', 
    'AADI.JK', 'KOTA.JK', 'CDIA.JK', 'BULL.JK', 'BIPI.JK', 'BUVA.JK', 'DEWA.JK', 'BNBR.JK', 'BUMI.JK'
]

def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[SKIP] Telegram credentials not found.")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"[ERROR] Telegram failed: {e}")

def get_live_data():
    print(f"--- Downloading {len(TICKERS_IDX)} Tickers ---")
    data = yf.download(TICKERS_IDX, period="3y", interval="1d", group_by='column', auto_adjust=True, threads=True)
    dfs = []
    for ticker in TICKERS_IDX:
        try:
            temp = data.xs(ticker, axis=1, level=1).dropna(subset=['Close']).copy()
            if len(temp) < 100: continue
            if temp['Close'].iloc[-1] < MIN_PRICE: continue
            if temp['Volume'].tail(5).mean() < 500000: continue
            temp['Ticker'] = ticker
            dfs.append(temp)
        except: continue
    return pd.concat(dfs).reset_index()

def process_features(df):
    df = df.sort_values(['Ticker', 'Date'])
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    
    if os.path.exists(MASTER_PATH):
        macro = pd.read_csv(MASTER_PATH, index_col=0)
        macro.index = pd.to_datetime(macro.index, format='mixed').tz_localize(None)
        df = df.set_index('Date').join(macro.pct_change().shift(1), how='inner', rsuffix='_m').reset_index()

    for w in [5, 20, 60]:
        df[f'ret_{w}'] = df.groupby('Ticker')['Close'].pct_change(w)
    
    df['std_5'] = df.groupby('Ticker')['Close'].transform(lambda x: x.pct_change().rolling(5).std())
    df['std_20'] = df.groupby('Ticker')['Close'].transform(lambda x: x.pct_change().rolling(20).std())
    df['vcp_squeeze'] = df['std_5'] / (df['std_20'] + 1e-9)
    df['avg_turnover'] = (df['Close'] * df['Volume']).groupby(df['Ticker']).transform(lambda x: x.rolling(20).mean())
    
    df['target'] = (df.groupby('Ticker')['Close'].shift(-HOLD_DAYS) / df['Close'] - 1 >= TARGET_PROFIT).astype(int)
    return df.dropna(subset=['vcp_squeeze', 'ret_20'])

def run_prediction(df):
    features = ["ret_5", "ret_20", "ret_60", "vcp_squeeze", "avg_turnover"]
    macro_cols = ['IHSG', 'USDIDR', 'SP500', 'GOLD']
    features += [c for c in macro_cols if c in df.columns]

    train_df = df.dropna(subset=['target'])
    latest_df = df.groupby('Ticker').last().reset_index()

    model = XGBClassifier(n_estimators=400, max_depth=7, learning_rate=0.015, scale_pos_weight=4, eval_metric='logloss')
    model.fit(train_df[features], train_df['target'])

    latest_df['prob'] = model.predict_proba(latest_df[features])[:, 1]
    return latest_df[(latest_df['prob'] >= PROB_THRESHOLD) & (latest_df['avg_turnover'] >= MIN_TURNOVER_7B)].copy()

if __name__ == "__main__":
    try:
        raw = get_live_data()
        processed = process_features(raw)
        picks = run_prediction(processed)
        
        now_str = datetime.now().strftime('%d/%m/%Y %H:%M')
        if picks.empty:
            msg = f"🤖 *IDX Scanner ({now_str})*\n\nBelum ada sinyal yang memenuhi kriteria."
        else:
            msg = f"🚀 *IDX SWING SIGNALS* ({now_str})\n\n"
            for _, row in picks.sort_values('prob', ascending=False).iterrows():
                prob_pct = f"{row['prob']*100:.1f}%"
                val_b = f"{row['avg_turnover']/1e9:.2f}B"
                msg += f"• *{row['Ticker']}*: Rp{row['Close']} | Prob: {prob_pct} | Vol: {val_b}\n"
            
            # Save Log local
            picks['Log_Date'] = datetime.now().strftime('%Y-%m-%d')
            picks.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)
            
        print(msg)
        send_telegram(msg)
        
    except Exception as e:
        err_msg = f"❌ *Scanner Error:* {str(e)}"
        print(err_msg)
        send_telegram(err_msg)