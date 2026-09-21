# Protocol lock — validasi prior terhadap endpoint anotasi ahli

Ditulis 2026-09-18, **sebelum** eksperimen dijalankan. Kriteria, kontrol, dan aturan tafsir di
bawah tidak boleh diubah setelah hasil terlihat. Hasil dilaporkan apa pun keluarannya.

## Pertanyaan

Apakah skor konflik dari prior pasangan lesi yang ditetapkan penulis memprediksi kesalahan kelas
yang ditentukan secara independen oleh anotasi ahli, dan apakah ia menambah informasi di atas
ambiguitas detektor biasa?

Ini berbeda dari Tabel 13 naskah, yang menurut naskah sendiri **sebagian definisional**: kategori
di sana dibentuk memakai ambang SDM lalu diuji dengan SDCA yang juga berasal dari SDM. Di sini
outcome berasal dari anotasi, tanpa menyentuh SDM.

## Populasi

Deteksi dengan `best_iou >= 0.5` dan `best_gt_class` tidak kosong, yaitu deteksi yang terlokalisasi
pada lesi beranotasi. Jumlah terverifikasi sebelum penguncian:

| Kohort | Terlokalisasi | Salah kelas | Rate | Citra |
|---|---:|---:|---:|---:|
| OIA-DDR valid (hanya untuk fitting) | 3.530 | 615 | 17,4% | 146 |
| OIA-DDR test | 6.800 | 1.309 | 19,2% | 224 |
| IDRiD | 945 | 118 | 12,5% | 27 |
| TJDR | 621 | 126 | 20,3% | 82 |
| FGADR | 15.046 | 1.941 | 12,9% | 1.214 |

## Outcome (hanya dari anotasi ahli)

Primer: `mismatch = 1` bila `pred_class != best_gt_class`, yaitu label yang diemisikan detektor
berbeda dari kelas ahli. Sensitivitas: outcome yang sama memakai `majority_class` (argmax ROI),
karena keduanya berbeda pada 4,41% deteksi.

SDM tidak dipakai sama sekali dalam mendefinisikan outcome.

## Prediktor (hanya dari keluaran model dan prior; tanpa GT)

| Kode | Sinyal | Peran |
|---|---|---|
| SDCA | `ev_sdca`, konflik prior penulis | metode utama |
| MARGIN | `unc_margin` | **kontrol kunci**: ambiguitas tanpa struktur pasangan |
| ENTROPY | `unc_entconf` | kontrol ambiguitas alternatif |
| MSP | `unc_msp` | kontrol kepercayaan |
| PERM | SDM dengan 6 nilai off-diagonal dipermutasi, 5 undian, benih 101-105 | kontrol struktur |
| EMP | relasi empiris dari anotasi: P(GT=b \| pred=a) dengan smoothing add-1 | kontrol relasi berbasis data |

EMP diestimasi **hanya pada OIA-DDR validasi**. Tidak ada CSV split train OIA-DDR; validasi adalah
split fitting di seluruh naskah dan terpisah dari test maupun kohort eksternal.

MARGIN, ENTROPY, dan MSP diorientasikan ke arah yang **memaksimalkan AUROC pada validasi**, lalu
tandanya dikunci dan dipakai apa adanya di test dan kohort eksternal. Orientasi ini sengaja
menguntungkan kontrol, bukan metode utama.

## Metrik

AUROC dan AUPRC untuk memprediksi mismatch; odds ratio deteksi konflik tinggi lawan rendah pada
ambang naskah sendiri (SDCA >= 0.30); bootstrap berklaster-citra 2.000 resample, benih 0.

Nilai tambah atas ambiguitas diuji dengan regresi logistik yang **dipasang pada OIA-DDR validasi**
lalu diterapkan pada test dan kohort eksternal: model MARGIN saja lawan model MARGIN + SDCA.

## Kontrol positif perangkat

`ev_sdca` tersimpan harus dapat direkonstruksi dari `SDM_norm[argmax ROI, runner-up]` pada
seluruh baris. Bila gagal, hasil tidak boleh ditafsirkan.

## Kriteria lolos (seluruhnya wajib)

- **A1** AUROC SDCA > 0.5 dengan CI bootstrap tidak memuat 0.5, pada OIA-DDR test **dan** pada
  sekurangnya 2 dari 3 kohort eksternal.
- **A2** SDCA mengungguli kontrol permutasi: selisih AUROC > 0 dengan CI berpasangan > 0 pada
  OIA-DDR test.
- **A3** Nilai tambah atas ambiguitas: AUROC(MARGIN + SDCA) > AUROC(MARGIN) dengan CI berpasangan
  > 0 pada OIA-DDR test **dan** pada sekurangnya 2 dari 3 kohort eksternal.

## Aturan tafsir (dikunci di muka)

1. Bila A1 sampai A3 terpenuhi **tetapi** AUROC EMP >= AUROC SDCA pada sekurangnya 2 kohort, klaim
   yang sah adalah relasi turunan anotasi sekurangnya sama informatifnya dengan prior penulis. Itu
   harus dilaporkan, dan mengganti prior dengan relasi empiris menjadi pilihan yang lebih defensible.
2. Bila A3 gagal, yang terbukti hanyalah "model yang ragu lebih sering salah". Itu **bukan** klaim
   tentang prior, dan tidak boleh ditulis sebagai validasi prior.
3. Klaim maksimum bila semuanya lolos: *the fixed lesion-relation signal has retrospective
   criterion validity against expert-derived lesion annotations across cohorts*. **Bukan** klaim
   tentang bahaya klinis atau keselamatan pasien.

## Isolasi

Skrip hanya membaca CSV bukti dan graf di lokasi aslinya. Seluruh keluaran ditulis di bawah
`_NATIVE/`. Naskah tidak disentuh sampai hasil keluar dan penulis memutuskan.
