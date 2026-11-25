"""alat.py -- fungsi bersama ketiga varian. Jangan diubah sendiri-sendiri."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.preprocessing import StandardScaler

SEED = 42
RENTANG_K = range(2, 11)
BERKAS_BERSIH = 'data/data_bersih.csv'
FOLDER_HASIL = 'data/hasil_varian'

# Label dan satuan untuk antarmuka aplikasi (ketentuan Tahap 3: setiap
# masukan harus punya label dan satuan yang dipahami orang luar).
LABEL = {
    'BALANCE': ('Saldo rata-rata', 'USD'),
    'BALANCE_FREQUENCY': ('Keseringan saldo diperbarui', 'skala 0-1'),
    'PURCHASES': ('Total pembelian', 'USD'),
    'ONEOFF_PURCHASES': ('Pembelian sekali bayar', 'USD'),
    'INSTALLMENTS_PURCHASES': ('Pembelian cicilan', 'USD'),
    'CASH_ADVANCE': ('Penarikan tunai', 'USD'),
    'PURCHASES_FREQUENCY': ('Keseringan berbelanja', 'skala 0-1'),
    'ONEOFF_PURCHASES_FREQUENCY': ('Keseringan belanja sekali bayar', 'skala 0-1'),
    'PURCHASES_INSTALLMENTS_FREQUENCY': ('Keseringan belanja cicilan', 'skala 0-1'),
    'CASH_ADVANCE_FREQUENCY': ('Keseringan menarik tunai', 'skala 0-1'),
    'CASH_ADVANCE_TRX': ('Jumlah transaksi tarik tunai', 'kali'),
    'PURCHASES_TRX': ('Jumlah transaksi belanja', 'kali'),
    'CREDIT_LIMIT': ('Limit kartu', 'USD'),
    'PAYMENTS': ('Total pembayaran', 'USD'),
    'MINIMUM_PAYMENTS': ('Pembayaran minimum', 'USD'),
    'PRC_FULL_PAYMENT': ('Proporsi pelunasan penuh', 'skala 0-1'),
    'TENURE': ('Lama berlangganan', 'bulan'),
}


def label_satuan(kolom: str) -> tuple[str, str]:
    return LABEL.get(kolom, (kolom.replace('_', ' ').title(), ''))


def muat_bersih(fitur: list[str] | None = None) -> pd.DataFrame:
    """Membaca data bersih kelompok. Kolom identitas selalu dibuang."""
    df = pd.read_csv(BERKAS_BERSIH)
    # Hanya kolom numerik yang menjadi fitur pembentuk klaster.
    # Kolom identitas seperti CUST_ID tidak membawa informasi kemiripan.
    df = df.select_dtypes(include='number')
    return df[fitur] if fitur else df


def skalakan(df: pd.DataFrame):
    sc = StandardScaler()
    return sc.fit_transform(df), sc


def jalankan_k(X_scaled, rentang=RENTANG_K) -> pd.DataFrame:
    """Menjalankan K-Means untuk tiap k, mengembalikan tabel metrik."""
    baris = []
    for k in rentang:
        km = KMeans(n_clusters=k, init='k-means++', n_init=20,
                    random_state=SEED)
        lab = km.fit_predict(X_scaled)
        baris.append({
            'k': k,
            'inertia': float(km.inertia_),
            'silhouette': float(silhouette_score(X_scaled, lab)),
            'davies_bouldin': float(davies_bouldin_score(X_scaled, lab)),
            'calinski_harabasz': float(calinski_harabasz_score(X_scaled, lab)),
        })
    return pd.DataFrame(baris)


def titik_siku(ks, inertia) -> int:
    """Titik siku = jarak tegak lurus terjauh dari garis ujung ke ujung."""
    x = np.asarray(ks, dtype=float)
    y = np.asarray(inertia, dtype=float)
    x_n = (x - x.min()) / (x.max() - x.min())
    y_n = (y - y.min()) / (y.max() - y.min())
    x1, y1, x2, y2 = x_n[0], y_n[0], x_n[-1], y_n[-1]
    jarak = np.abs((y2 - y1) * x_n - (x2 - x1) * y_n + x2 * y1 - y2 * x1)
    jarak /= np.hypot(y2 - y1, x2 - x1)
    return int(x[int(np.argmax(jarak))])


def profil_relatif(df: pd.DataFrame, label) -> pd.DataFrame:
    """Simpangan rata-rata tiap klaster terhadap rata-rata seluruh data,
    dalam satuan simpangan baku. Baris = fitur, kolom = klaster."""
    d = df.copy()
    d['cluster'] = label
    rata = d.drop(columns='cluster').mean()
    sd = d.drop(columns='cluster').std().replace(0, np.nan)
    return ((d.groupby('cluster').mean() - rata) / sd).round(2).T


def nama_otomatis(profil: pd.DataFrame) -> dict[str, str]:
    """Dua fitur dengan simpangan terbesar secara mutlak, beserta arahnya.
    WAJIB diganti mahasiswa dengan nama yang bermakna."""
    nama = {}
    for c in profil.columns:
        kol = profil[c]
        kuat = kol.reindex(kol.abs().sort_values(ascending=False).index)[:2]
        nama[str(int(c))] = ', '.join(
            f'{n} {"tinggi" if v > 0 else "rendah"}' for n, v in kuat.items())
    return nama


def simpan_hasil(nomor: int, isi: dict) -> str:
    os.makedirs(FOLDER_HASIL, exist_ok=True)
    jalur = f'{FOLDER_HASIL}/varian{nomor}.json'
    with open(jalur, 'w', encoding='utf-8') as f:
        json.dump(isi, f, indent=2, ensure_ascii=False)
    return jalur


def baca_hasil(nomor: int) -> dict | None:
    jalur = f'{FOLDER_HASIL}/varian{nomor}.json'
    if not os.path.exists(jalur):
        return None
    with open(jalur, encoding='utf-8') as f:
        return json.load(f)
