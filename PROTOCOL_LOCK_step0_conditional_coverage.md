# Protocol lock — Langkah 0: cakupan bersyarat per stratum korroborasi

Ditulis 2026-09-18, **sebelum** eksperimen dijalankan. Kriteria di bawah tidak boleh diubah
setelah hasil terlihat. Hasil dilaporkan apa pun keluarannya.

## Latar

Seluruh jaminan conformal yang diukur sejauh ini bersifat **marginal**. Ensemble lima detektor
mencapai selisih cakupan +0,000 secara keseluruhan pada OIA-DDR test. Jaminan marginal dikenal
dapat menyembunyikan kegagalan subkelompok: cakupan dapat tinggi pada kotak yang dikorroborasi
banyak detektor dan rendah pada kotak tanpa korroborasi, sementara rata-ratanya tetap nominal.

Laju positif menurut jumlah detektor yang setuju terukur 0,086 (0 anggota) sampai 0,844
(5 anggota), rentang hampir sepuluh kali lipat. Satu ambang global tidak mungkin melayani kedua
ujung itu sama baiknya. Langkah 0 menguji dugaan tersebut.

## Pertanyaan

Apakah ambang conformal yang dikalibrasi secara marginal pada OIA-DDR validasi memberikan
cakupan yang mendekati nominal **di dalam setiap stratum korroborasi** pada OIA-DDR test?

## Populasi

Seluruh deteksi kandidat detektor primer pada OIA-DDR test, target `governance_retain`,
identik dengan protokol `tools/eval_fp_reduction_benchmark.py`.

## Taksonomi stratum

`k` = jumlah anggota dari lima detektor detcmp (`fasterrcnn`, `fasterrcnn_v2`, `fcos`,
`retinanet`, `retinanet_v2`) yang memiliki kotak tercocokkan pada IoU minimal 0,15 terhadap
kotak primer. `k` bernilai 0 sampai 5. Taksonomi ini **tidak memakai anotasi**.

## Metode yang diuji

| Kode | Skor |
|---|---|
| S | skor detektor primer |
| S+E | gate penuh, lima benih, dipasang pada validasi |
| E5 | rata-rata lima detektor detcmp |
| E6 | rata-rata primer ditambah lima detcmp |

## Prosedur

Untuk tiap metode: hitung ambang split-conformal **marginal** pada OIA-DDR validasi dengan
miscoverage `alpha = 0,20`, memakai kuantil `ceil((n+1)(1-alpha))/n` dari nonkonformitas positif,
persis seperti `tools/eval_selective_conformal_baselines.py`. Terapkan ambang itu tanpa perubahan
ke test, lalu ukur cakupan empiris **di dalam tiap stratum**.

Selisih per stratum = cakupan empiris dikurangi 0,80.

## Kriteria, dikunci di muka

- **MATI.** Bila setiap metode memiliki `|selisih| <= 0,05` pada **setiap** stratum, jaminan
  marginal sudah cukup, gagasan cakupan bersyarat tidak punya ruang, dan pekerjaan dihentikan.
- **LANJUT.** Bila sekurangnya satu metode memiliki `|selisih| > 0,15` pada sekurangnya satu
  stratum yang memuat minimal 200 deteksi positif.
- **ABU-ABU.** Bila selisih maksimum berada di antara 0,05 dan 0,15, sinyalnya lemah; hasil
  dilaporkan dan keputusan diserahkan kepada penulis tanpa melanjutkan otomatis.

Stratum dengan kurang dari 200 deteksi positif dilaporkan tetapi **tidak** dipakai untuk memicu
kriteria LANJUT, karena cakupan pada sampel kecil tidak stabil.

## Keluaran sekunder, BUKAN bagian kriteria

Sebagai pratinjau Langkah 1, skrip juga menghitung ambang conformal **per stratum** (Mondrian)
yang dikalibrasi pada validasi dan diterapkan ke test. Angka ini dilaporkan dengan label jelas
dan **tidak boleh** dipakai untuk menilai kriteria di atas, karena kriteria hanya menyangkut
perilaku metode marginal.

## Isolasi

Skrip hanya membaca cache dan CSV di lokasi aslinya. Seluruh keluaran ditulis di bawah
`_LGOVNET_NEW_SEP/`. Naskah `_FINAL_Latex/` tidak disentuh.
