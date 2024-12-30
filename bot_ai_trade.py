import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import logging
from datetime import datetime, timedelta
from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL
import time
import signal
import sys
import threading
import json
import os

# Fungsi untuk mencatat log ke file
def log_to_file(message):
    with open('bot_log.txt', 'a') as log_file:
        log_file.write(f"{datetime.now().isoformat()} - {message}\n")

# Fungsi untuk mengirim pesan ke Telegram
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    try:
        requests.post(url, json=payload)
    except Exception as e:
        log_to_file(f"Error sending message to Telegram: {e}")

# Penanganan sinyal untuk interupsi pengguna
def signal_handler(sig, frame):
    reason = "Interupsi pengguna" if sig == signal.SIGINT else "Sinyal penghentian"
    log_to_file(f"Bot terhenti dengan alasan: {reason}")
    send_telegram_message(f"Bot terhenti dengan alasan: {reason}")
    sys.exit(0)

# Tangani sinyal SIGINT dan SIGTERM
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Konfigurasi API Binance Testnet
API_KEY = '9lW8rdz9w8KlZMOuQDEkg4QDoxUDspCJXllgULTa7cyQ95cxwu8VjkFttv01BjjI'
API_SECRET = 'LA8iHHFoSM7esevMQDK3opZNZnXCjuR8li73aZy9MofXPcbYeCRBg5Sfx6OYuVrt'
BASE_URL = 'https://testnet.binance.vision'

# Konfigurasi Telegram
TELEGRAM_BOT_TOKEN = '7885418020:AAFwD4z6b7JvDO7jdP_Q4mtvFsG0veNBfgQ'
TELEGRAM_CHAT_ID = '1446673557'

# Inisialisasi Binance Client
client = Client(API_KEY, API_SECRET, testnet=True)

# File JSON untuk menyimpan blacklist
BLACKLIST_FILE = 'blacklist.json'
PURCHASE_HISTORY_FILE = 'purchase_history.json'

# Fungsi untuk memuat blacklist dari file JSON
def load_blacklist():
    if os.path.exists(BLACKLIST_FILE):
        with open(BLACKLIST_FILE, 'r') as file:
            return json.load(file)
    return []

# Fungsi untuk menyimpan blacklist ke file JSON
def save_blacklist(blacklist):
    with open(BLACKLIST_FILE, 'w') as file:
        json.dump(blacklist, file)

# Fungsi untuk menambahkan token ke blacklist
def add_to_blacklist(symbol):
    blacklist = load_blacklist()
    if symbol not in blacklist:
        blacklist.append(symbol)
        save_blacklist(blacklist)
        print(f"Token {symbol} telah ditambahkan ke blacklist.")

# Fungsi untuk memeriksa apakah token ada di blacklist
def is_blacklisted(symbol):
    blacklist = load_blacklist()
    return symbol in blacklist

# Tambahkan token yang diberikan ke blacklist
blacklist_tokens = [
    "PIVX", "REN", "KEY", "OGN", "YFI", "WBTC", "UNFI", "QNT", "GNO", "KP3R",
    "BIFI", "LDO", "LQTY", "WBETH", "AI", "W", "TAO", "BNSOL"
]

for token in blacklist_tokens:
    add_to_blacklist(token)

# Fungsi untuk memuat riwayat pembelian dari file JSON
def load_purchase_history():
    if os.path.exists(PURCHASE_HISTORY_FILE):
        with open(PURCHASE_HISTORY_FILE, 'r') as file:
            return json.load(file)
    return {}

# Fungsi untuk menyimpan riwayat pembelian ke file JSON
def save_purchase_history(purchase_history):
    with open(PURCHASE_HISTORY_FILE, 'w') as file:
        json.dump(purchase_history, file)

# Fungsi untuk menambahkan pembelian ke riwayat
def add_purchase(symbol, price, quantity):
    purchase_history = load_purchase_history()
    purchase_history[symbol] = {
        'price': price,
        'quantity': quantity,
        'timestamp': datetime.now().isoformat()
    }
    save_purchase_history(purchase_history)

# Fungsi untuk mendapatkan saldo token di atas threshold
def get_balances_above_threshold(threshold):
    balances = client.get_account()['balances']
    result = []
    for balance in balances:
        asset = balance['asset']
        free_balance = float(balance['free'])
        if free_balance > threshold and not is_blacklisted(asset):
            try:
                value_in_usdt = free_balance * float(client.get_symbol_ticker(symbol=f"{asset}USDT")['price'])
                result.append((asset, free_balance, value_in_usdt))
            except Exception as e:
                log_to_file(f"Error processing {asset}USDT: {e}")
                send_telegram_message(f"Error processing {asset}USDT: {e}")
                add_to_blacklist(f"{asset}USDT")
    return result

# Fungsi untuk mendapatkan data historis
def get_historical_data(symbol, interval, limit):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    data = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 
        'taker_buy_quote_asset_volume', 'ignore'
    ])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.set_index('timestamp', inplace=True)
    data = data.astype(float)
    return data

# Fungsi untuk mendapatkan 10 token dengan PNL antara -1% hingga -5% di Binance
def get_top_10_pnl_range_tokens():
    tickers = client.get_all_tickers()
    pnl_list = []

    # Baca token yang mengalami error dari file
    token_errors = set()
    try:
        with open('token_errors.txt', 'r') as f:
            token_errors = set(f.read().splitlines())
    except FileNotFoundError:
        pass

    for ticker in tickers:
        symbol = ticker['symbol']
        if symbol.endswith('USDT') and symbol not in token_errors and not is_blacklisted(symbol):
            try:
                # Dapatkan data historis
                interval = '1h'
                limit = 200
                data = get_historical_data(symbol, interval, limit)

                if data.empty:
                    continue

                # Preprocessing data
                data = add_technical_indicators(data)
                data['returns'] = data['close'].pct_change()
                data.dropna(inplace=True)

                # Hitung PNL
                pnl = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]
                if -0.05 <= pnl <= -0.01:
                    pnl_list.append((symbol, pnl))
            except Exception as e:
                log_to_file(f"Error processing {symbol}: {e}")
                send_telegram_message(f"Error processing {symbol}: {e}")
                add_to_blacklist(symbol)

    # Urutkan berdasarkan PNL dan ambil 10 teratas
    pnl_list.sort(key=lambda x: x[1])
    return pnl_list[:10]

# Fungsi untuk menambahkan indikator teknis
def add_technical_indicators(data):
    data['ma_5'] = data['close'].rolling(window=5).mean()
    data['ma_10'] = data['close'].rolling(window=10).mean()
    data['rsi'] = compute_rsi(data['close'], window=14)
    data['macd'], data['macd_signal'] = compute_macd(data['close'])
    data['bollinger_upper'], data['bollinger_lower'] = compute_bollinger_bands(data['close'])
    return data

# Fungsi untuk menghitung RSI
def compute_rsi(series, window):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Fungsi untuk menghitung MACD
def compute_macd(series, fast=12, slow=26, signal=9):
    fast_ema = series.ewm(span=fast, min_periods=1).mean()
    slow_ema = series.ewm(span=slow, min_periods=1).mean()
    macd = fast_ema - slow_ema
    macd_signal = macd.ewm(span=signal, min_periods=1).mean()
    return macd, macd_signal

# Fungsi untuk menghitung Bollinger Bands
def compute_bollinger_bands(series, window=20, num_std_dev=2):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    bollinger_upper = rolling_mean + (rolling_std * num_std_dev)
    bollinger_lower = rolling_mean - (rolling_std * num_std_dev)
    return bollinger_upper, bollinger_lower

# Fungsi untuk membuat order
def create_order(symbol, side, usdt_amount):
    try:
        symbol_info = client.get_symbol_info(symbol)
        if not symbol_info:
            raise Exception(f"Symbol info not found for {symbol}")

        # Dapatkan harga terkini
        price = float(client.get_symbol_ticker(symbol=symbol)['price'])

        # Hitung jumlah token yang akan dibeli berdasarkan jumlah USDT
        quantity = usdt_amount / price

        # Dapatkan minimum notional dan lot size dari filter simbol
        min_notional = None
        step_size = None
        for f in symbol_info['filters']:
            if f['filterType'] == 'MIN_NOTIONAL':
                min_notional = float(f['minNotional'])
            if f['filterType'] == 'LOT_SIZE':
                step_size = float(f['stepSize'])

        if min_notional:
            # Pastikan jumlah order memenuhi persyaratan minimum notional
            if usdt_amount < min_notional:
                #raise Exception(f"Jumlah USDT tidak memenuhi persyaratan minimum notional: {min_notional} USDT")
                log_to_file(f"Jumlah USDT tidak memenuhi minimal notional ({min_notional} USDT). Meminta input dari pengguna.")
                send_telegram_message(f"Jumlah USDT tidak memenuhi minimal notional ({min_notional} USDT). Silakan masukkan jumlah baru.")
                usdt_amount = get_user_input_amount(f"Masukkan jumlah USDT baru untuk membeli {symbol} (minimal {min_notional} USDT): ", 30)
                if usdt_amount < min_notional:
                    error_message = f"Jumlah baru yang dimasukkan masih di bawah minimal notional ({min_notional} USDT). Pembelian dibatalkan."
                    log_to_file(error_message)
                    send_telegram_message(error_message)
                    return None

        if step_size:
            # Pastikan jumlah order sesuai dengan step size
            quantity = round(quantity // step_size * step_size, 8)

        order = client.create_order(
            symbol=symbol,
            side=side,
            type='MARKET',
            quantity=quantity
        )
        log_to_file(f"Order created: {order}")  # Log transaksi
        send_telegram_message(f"Order created: {order}")

        if side == SIDE_BUY:
            add_purchase(symbol, price, quantity)

        return order
    except Exception as e:
        log_to_file(f"Error creating order: {e}")
        send_telegram_message(f"Error creating order: {e}")
        return None

# Fungsi untuk meminta input dari pengguna dengan batas waktu
def get_user_input_amount(prompt, timeout):
    user_input = [None]
    def ask_input():
        user_input[0] = input(prompt)
    thread = threading.Thread(target=ask_input)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        print("\nWaktu habis! Menggunakan default $25 USDT untuk pembelian.")
        user_input[0] = '25'  # Default to '25' if no input is provided
    return float(user_input[0])

# Fungsi untuk memeriksa dan melikuidasi token jika harga naik 3% dari harga pembelian
def check_take_profit():
    purchase_history = load_purchase_history()
    for symbol, details in purchase_history.items():
        current_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
        purchase_price = details['price']
        if current_price >= purchase_price * 1.03:
            quantity = details['quantity']
            order = create_order(symbol, SIDE_SELL, quantity)
            if order:
                log_to_file(f"Token {symbol} telah dilikuidasi dengan harga {current_price}, yang ekivalen dengan {current_price * quantity} USDT")
                send_telegram_message(f"Token {symbol} telah dilikuidasi dengan harga {current_price}, yang ekivalen dengan {current_price * quantity} USDT")

# Fungsi untuk mencatat saldo token di atas threshold ke file log
def log_balances_above_threshold():
    threshold = 20  # Set threshold value
    balances = get_balances_above_threshold(threshold)
    for asset, free_balance, value_in_usdt in balances:
        log_to_file(f"Asset: {asset}, Free Balance: {free_balance}, Value in USDT: {value_in_usdt}")
        send_telegram_message(f"Asset: {asset}, Free Balance: {free_balance}, Value in USDT: {value_in_usdt}")  # Kirim pesan ke Telegram

# Fungsi untuk meminta input dari pengguna dengan batas waktu (10 detik)
def get_user_input_with_timeout(prompt, timeout=20):
    user_input = [None]

    def ask_input():
        user_input[0] = input(prompt).strip().lower()

    input_thread = threading.Thread(target=ask_input)
    input_thread.start()
    input_thread.join(timeout)

    if input_thread.is_alive():
        print("\nWaktu habis! Pilihan default 'n' akan diterapkan.")
        user_input[0] = 'n'  # Default to 'n' if no input is provided

    return user_input[0]

def get_usdt_balance():
    try:
        balances = client.get_account()['balances']
        usdt_balance = float(next((b['free'] for b in balances if b['asset'] == 'USDT'), 0))
        return usdt_balance
    except Exception as e:
        log_to_file(f"Error fetching USDT balance: {e}")
        send_telegram_message(f"Error fetching USDT balance: {e}")
        return 0

# Fungsi utama
def main():
    try:
        log_to_file("Bot Running")
        send_telegram_message("Bot Running")
        # Fetch and log USDT balance
        usdt_balance = get_usdt_balance()
        log_to_file(f"Available USDT Balance: {usdt_balance:.2f}")
        send_telegram_message(f"Available USDT Balance: {usdt_balance:.2f} USDT")
        log_balances_above_threshold()  # Log saldo token saat bot mulai berjalan

        load_purchase_history()  # Load purchase history from file

        # Meminta input dari pengguna untuk menggunakan indikator teknis atau tidak
        user_choice = get_user_input_with_timeout("Apakah Anda ingin menggunakan indikator teknis? (y/n): ").strip().lower()
        use_technical_indicators = user_choice == 'y'
        log_to_file(f"Pengguna memilih untuk menggunakan indikator teknis: {use_technical_indicators}")
        send_telegram_message(f"Pengguna memilih untuk menggunakan indikator teknis: {use_technical_indicators}")       

        while True:
            log_to_file("Memulai loop utama")
            send_telegram_message("Memulai loop utama")
            # Dapatkan 10 token dengan PNL antara -1% hingga -5%
            pnl_range_tokens = get_top_10_pnl_range_tokens()
            log_to_file(f"Token dengan PNL antara -1% hingga -5%: {pnl_range_tokens}")
            send_telegram_message(f"Token dengan PNL antara -1% hingga -5%: {pnl_range_tokens}")    

            for symbol, pnl in pnl_range_tokens:
                log_to_file(f"Memproses token: {symbol}")
                send_telegram_message(f"Memproses token: {symbol}") 
                # Dapatkan data historis
                interval = '1h'
                limit = 200
                data = get_historical_data(symbol, interval, limit)

                if data.empty:
                    log_to_file(f"Tidak ada data yang diperoleh untuk {symbol}.")
                    continue

                # Preprocessing data
                if use_technical_indicators:
                    data = add_technical_indicators(data)
                data['returns'] = data['close'].pct_change()
                data.dropna(inplace=True)

                # Fitur dan target
                if use_technical_indicators:
                    X = data[['open', 'high', 'low', 'close', 'volume', 'ma_5', 'ma_10', 'rsi', 'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower']]
                else:
                    X = data[['open', 'high', 'low', 'close', 'volume']]
                y = (data['returns'] > 0).astype(int)

                # Split data menjadi train dan test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Scaling data
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Inisialisasi RandomForestClassifier
                clf = RandomForestClassifier(random_state=42)

                # Hyperparameter tuning menggunakan GridSearchCV
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
                grid_search.fit(X_train_scaled, y_train)

                # Model terbaik
                best_clf = grid_search.best_estimator_

                # Memprediksi set pengujian
                y_pred = best_clf.predict(X_test_scaled)

                # Menghitung akurasi
                accuracy = np.mean(y_pred == y_test)
                print(f"Akurasi untuk {symbol}: {accuracy * 100:.2f}%")
                log_to_file(f"Akurasi untuk {symbol}: {accuracy * 100:.2f}%")
                send_telegram_message(f"Akurasi untuk {symbol}: {accuracy * 100:.2f}%") 

                # Cek apakah token sudah ada dalam riwayat pembelian dan saldo lebih dari $20 USDT
                if symbol in load_purchase_history():
                    balances = get_balances_above_threshold(threshold=20)
                    for asset, free_balance, value_in_usdt in balances:
                        if asset + 'USDT' == symbol and value_in_usdt > 20:
                            log_to_file(f"Token {symbol} sudah ada dalam riwayat pembelian dan saldo lebih dari $20 USDT, pembelian dilewatkan.")
                            continue

                # Sinyal beli berdasarkan indikator teknis atau prediksi model
                if use_technical_indicators:
                    latest_data = data.iloc[-1]
                    if latest_data['rsi'] < 30 or (latest_data['macd'] > latest_data['macd_signal'] and latest_data['close'] < latest_data['bollinger_lower']):
                        log_to_file(f"Bot mendapatkan sinyal beli, Token {symbol} akan dibeli")
                        send_telegram_message(f"Bot mendapatkan sinyal beli, Token {symbol} akan dibeli")
                        usdt_amount = get_user_input_amount("Masukkan jumlah USDT yang akan digunakan untuk membeli token (minimal $25 USDT): ", 30)
                        if usdt_amount < 25:
                            usdt_amount = 25
                        buy_order = create_order(symbol, SIDE_BUY, usdt_amount)
                        if buy_order:
                            initial_buy_price = float(buy_order['fills'][0]['price'])  # Harga dari API Binance
                            bought_quantity = float(buy_order['executedQty'])
                            log_to_file(f"Bot telah selesai membeli Token {symbol} sejumlah = {bought_quantity}, yang ekivalen dengan = {initial_buy_price * bought_quantity:.2f} USDT")
                            send_telegram_message(f"Bot telah selesai membeli Token {symbol} sejumlah = {bought_quantity}, yang ekivalen dengan = {initial_buy_price * bought_quantity:.2f} USDT")
                else:                    
                    # Prediksi sinyal pada data terbaru
                    latest_data = scaler.transform([X.iloc[-1].values])
                    prediction = best_clf.predict(latest_data)[0]
                    log_to_file(f"Sinyal prediksi untuk {symbol}: {'Beli' if prediction == 1 else 'Tidak Beli'}")
                    send_telegram_message(f"Sinyal prediksi untuk {symbol}: {'Beli' if prediction == 1 else 'Tidak Beli'}")

                    # Tampilkan akurasi model ke log dan Telegram
                    log_to_file(f"Akurasi model prediksi untuk {symbol}: {accuracy * 100:.2f}%")
                    send_telegram_message(f"Akurasi model prediksi untuk {symbol}: {accuracy * 100:.2f}%")

                    # Jika sinyal beli, lakukan order
                    if prediction == 1:
                        usdt_amount = get_user_input_amount("Masukkan jumlah USDT yang akan digunakan untuk membeli token (minimal $25 USDT): ", 30)
                        if usdt_amount < 25:
                            usdt_amount = 25
                        buy_order = create_order(symbol, SIDE_BUY, usdt_amount)
                        if buy_order:
                            initial_buy_price = float(buy_order['fills'][0]['price'])  # Harga dari API Binance
                            bought_quantity = float(buy_order['executedQty'])
                            log_to_file(f"Bot telah selesai membeli Token {symbol} sejumlah = {bought_quantity}, yang ekivalen dengan = {initial_buy_price * bought_quantity:.2f} USDT")
                            send_telegram_message(f"Bot telah selesai membeli Token {symbol} sejumlah = {bought_quantity}, yang ekivalen dengan = {initial_buy_price * bought_quantity:.2f} USDT")


            check_take_profit()
            log_balances_above_threshold()
            time.sleep(300)

    except Exception as e:
        log_to_file(f"Bot mengalami kesalahan: {e}")
        send_telegram_message(f"Bot mengalami kesalahan: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
