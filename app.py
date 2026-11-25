"""
app.py  --  Aplikasi web segmentasi pelanggan (Tahap 3)
---------------------------------------------------------
Menjalankan di komputer sendiri:
    pip install -r requirements.txt
    streamlit run app.py

Tiga aturan aplikasi yang dipatuhi berkas ini:
  1. Satu kasus, bukan satu berkas. Pengguna mengisi nilai untuk satu baris
     data, lalu menerima satu jawaban. Tidak ada unggahan CSV.
  2. Memberi informasi yang belum diketahui pengguna: segmen mana yang paling
     mirip dengan polanya, dan apa ciri segmen itu.
  3. Memakai scaler yang disimpan, dengan transform. Model tidak dilatih
     ulang di dalam aplikasi.
"""
from __future__ import annotations

import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Segmentasi Pelanggan K-Means', layout='wide')


@st.cache_resource
def muat():
    model = joblib.load('model/model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    fitur = joblib.load('model/fitur.pkl')
    with open('model/segmen.json', encoding='utf-8') as f:
        info = json.load(f)
    return model, scaler, fitur, info


try:
    model, scaler, fitur, info = muat()
except FileNotFoundError:
    st.error('Berkas model belum ada. Jalankan dulu: '
             'python tahap3_model_final.py')
    st.stop()

NAMA = info['nama_segmen']
# Saran tindakan diisi di sel 6.3. Dipakai .get agar aplikasi tetap berjalan
# pada segmen.json lama yang belum memuatnya.
TINDAKAN = info.get('tindakan_disarankan', {})
PROFIL = pd.DataFrame(info['profil_relatif'])      # baris fitur, kolom klaster
MASUKAN = {m['kolom']: m for m in info['masukan']}

# ---------------------------------------------------------------- sidebar
st.sidebar.header('Tentang model ini')
st.sidebar.metric('Jumlah segmen', info['k'])
st.sidebar.metric('Silhouette Score', f"{info['silhouette']:.3f}")
st.sidebar.metric('Jumlah fitur', len(fitur))
st.sidebar.caption(
    f"Dilatih pada {info['jumlah_baris']:,} baris data. Jumlah segmen dan "
    f"himpunan fitur ({info.get('himpunan_fitur', '-')}) ditetapkan lewat "
    f"{info.get('asal_keputusan', 'keputusan kelompok')}; "
    f"scikit-learn {info.get('versi_sklearn', '-')}."
)

st.title('Segmentasi Pelanggan Kartu Kredit')
st.write(
    'Isi pola pemakaian kartu untuk **satu** pelanggan. Aplikasi akan '
    'menunjukkan segmen mana yang paling mirip dengan pola itu, ciri khas '
    'segmen tersebut, dan posisi nilai yang Anda isi dibanding rata-rata '
    'seluruh pelanggan.'
)

# ------------------------------------------------------------------- form
with st.form('satu_kasus'):
    st.subheader('Data satu pelanggan')
    kolom = st.columns(3)
    nilai = {}
    for i, f in enumerate(fitur):
        m = MASUKAN[f]
        judul = f"{m['label']} ({m['satuan']})" if m['satuan'] else m['label']
        langkah = max((m['maks'] - m['min']) / 100, 0.001)
        nilai[f] = kolom[i % 3].number_input(
            judul,
            min_value=float(m['min']),
            max_value=float(m['maks']),
            value=float(m['awal']),
            step=float(langkah),
            format='%.2f',
            help=f"Rentang pada data pelatihan: {m['min']:,.2f} "
                 f"sampai {m['maks']:,.2f}",
        )
    kirim = st.form_submit_button('Tentukan segmen')

# ------------------------------------------------------------------ hasil
if kirim:
    X = pd.DataFrame([nilai])[fitur]      # urutan kolom dari fitur.pkl
    X_skala = scaler.transform(X)         # transform, BUKAN fit_transform
    klaster = int(model.predict(X_skala)[0])

    st.success(f'Segmen {klaster} — {NAMA[str(klaster)]}')

    saran = TINDAKAN.get(str(klaster), '')
    if saran:
        st.markdown(f'**Saran tindakan:** {saran}')

    kiri, kanan = st.columns(2)

    with kiri:
        st.subheader('Ciri khas segmen ini')
        st.caption('Simpangan rata-rata segmen terhadap rata-rata seluruh '
                   'pelanggan, dalam satuan simpangan baku.')
        ciri = PROFIL[str(klaster)].sort_values(key=abs, ascending=False).head(6)
        st.dataframe(pd.DataFrame({
            'Ciri': [MASUKAN[f]['label'] for f in ciri.index],
            'Simpangan': ciri.to_numpy().round(2),
            'Arah': ['di atas rata-rata' if v > 0 else 'di bawah rata-rata'
                     for v in ciri],
        }), hide_index=True)

    with kanan:
        st.subheader('Posisi nilai yang Anda isi')
        st.caption('Berapa simpangan baku nilai Anda dari rata-rata '
                   'seluruh pelanggan.')
        posisi = []
        for f in fitur:
            m = MASUKAN[f]
            z = (nilai[f] - m['rata']) / m['sd'] if m['sd'] else 0.0
            posisi.append({'Ciri': m['label'], 'Nilai': nilai[f],
                           'Simpangan': round(z, 2)})
        pos = pd.DataFrame(posisi)
        pos = pos.reindex(pos['Simpangan'].abs().sort_values(
            ascending=False).index).head(6)
        st.dataframe(pos, hide_index=True)

    st.subheader('Seluruh segmen')
    st.caption('Kolom yang sedang terpilih ditandai pada judulnya.')
    tampil = PROFIL.copy()
    tampil.index = [MASUKAN[f]['label'] for f in tampil.index]
    tampil.columns = [f'Segmen {c}' + (' (terpilih)' if int(c) == klaster else '')
                      for c in tampil.columns]
    st.dataframe(tampil)

    for nomor, nm in NAMA.items():
        jml = info['jumlah_anggota'][nomor]
        baris = (f"**Segmen {nomor} — {nm}**  \n"
                 f"{jml:,} pelanggan "
                 f"({jml / info['jumlah_baris'] * 100:.1f}% dari data pelatihan)")
        if TINDAKAN.get(nomor):
            baris += f"  \nSaran tindakan: {TINDAKAN[nomor]}"
        st.markdown(baris)

    st.info(info['keterbatasan'])
else:
    st.caption('Isi formulir di atas, lalu tekan tombol untuk melihat hasilnya.')
