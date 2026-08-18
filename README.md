# AI Trading Intelligence V1 - Real Account Ready

Aplikasi dasbor analisis dan otomatisasi perdagangan berbasis Python (Streamlit) yang menggabungkan kecerdasan buatan (OpenAI) dengan sistem konfirmasi teknikal berlapis.

## 📈 Logika Strategi & Pengambilan Keputusan (Sempurna)
Sistem ini memadukan arsitektur riset bawaan dengan filter perdagangan akun riil yang ketat:
1. **Analisis Tren Dominan**: Membaca arah pergerakan harga makro (Naik/Turun).
2. **Market Structure & ICT**: Mendeteksi titik krusial perubahan struktur harga (*Break of Structure* / BOS dan *Change of Character* / ChoCH).
3. **Fibonacci Retracement**: Menemukan titik jenuh entri koreksi pada area *Golden Ratio* (61.8% - 78.6%).

### 🛡️ Aturan Filter Eksekusi Ketat (Anti-Sideways)
* **Eksekusi Entry (BUY/SELL)**: Hanya diperbolehkan jika minimal **2 dari 3 indikator** teknikal di atas memberikan konfirmasi searah dengan bias pasar.
* **No Trade (Proteksi Modal)**: Jika pasar terdeteksi mendatar (*Sideways*), atau jika **hanya 1 indikator** yang berhasil terkonfirmasi, AI secara mutlak akan membatalkan pesanan entry demi keamanan dana akun riil.

### 💼 Manajemen Risiko Finansial (Rasio Fixed 1:2)
* **Stop Loss (SL)**: Jarak kerugian maksimal dibatasi setara dengan **Rp 50.000,-** (Rasio 1)
* **Take Profit (TP)**: Target keuntungan otomatis ditetapkan dua kali lipat, yaitu **Rp 100.000,-** (Rasio 2)

---

## 🛠️ Cara Instalasi & Penggunaan

### 1. Instalasi Pustaka
Disarankan menggunakan Python 3.11+. Buka terminal atau Command Prompt, lalu jalankan:
```bash
pip install -r requirements.txt
```

### 2. Jalankan Dasbor Web
Aplikasi ini berjalan sebagai visual dashboard di browser Anda. Eksekusi dengan perintah:
```bash
streamlit run app.py
```

### 3. Struktur Data CSV (`sample_data.csv`)
Pastikan file data historis pasar Anda memiliki kolom wajib berikut dalam format huruf kecil:
`open`, `high`, `low`, `close` (Opsional: `timestamp`, `volume`).

### 4. API Key Keamanan
* Masukkan OpenAI API key Anda melalui kolom input di panel samping (*sidebar*) saat aplikasi berjalan di web browser.
* **PERINGATAN**: Jangan pernah mengetikkan atau mengunggah API key langsung di dalam file kode sumber GitHub demi keamanan saldo OpenAI Anda.

## 🏗️ Alur Arsitektur Sistem V1 (Sempurna)
`Data CSV Pasar` ➡️ `Engine Indikator (Trend, Fibo, ICT)` ➡️ `Streamlit Dashboard` ➡️ `OpenAI Reasoning (Prompt Kustom)` ➡️ `Rencana Eksekusi Risiko & Laporan Riset`.
