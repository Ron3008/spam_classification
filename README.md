# Enail Spam Classification
Proyek ini adalah aplikasi web berbasis machine learning yang mampu mengklasifikasikan isi email menjadi spam atau non-spam. Aplikasi ini dibangun menggunakan metode ensemble learning dari dua model: Naive Bayes dan Random Forest, dengan implementasi full-stack menggunakan backend Flask dan frontend yang sederhana namun responsif. Proyek ini dibuat untuk memenuhi tugas Mata Kuliah Machine Learning COMP6577001 LK01 Tahun Ajaran GENAP Semester 4 yang beranggotakan: -HUGO SACHIO WIJAYA 2702261151 Computer Science: Bertugas sebagai penulis utama code, backend, serta frontend (tetapi bukan penullis model backend pickle) BRIAN ALEXANDER 2702282351 : Bertugas menguji dan menciptakan model pickle Naive Bayes, Random Forest dan stacking serta menguji akurasi model EZEKIEL AARON SETIAWAN 2702288600 : Bertugas mendeploy model ke frontend streamlit 

## Fitur
- Klasifikasi isi email melalui input manual atau unggah file (.txt/.pdf)
- Pembelajaran ensemble (Naive Bayes + Random Forest)
- Menampilkan skor kepercayaan (confidence score) dari model
- Autentikasi pengguna (register & login)
- Riwayat klasifikasi tersimpan untuk setiap pengguna
- Tampilan frontend dengan opsi mode gelap/terang (uji coba)
- Teknologi yang Digunakan

Backend: Python, Flask, scikit-learn
Frontend: HTML, CSS, JavaScript
Model ML: Naive Bayes, Random Forest (Model Stacking)
Lainnya: pdfplumber, Flask-Login

Description of the app ...

## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://spamemail-classification.streamlit.app/)

## Cara Set-up Streamlit

1. Buat akun di Streamlit cloud menggunakan akun GitHub
2. Tekan tombol Create App di pojok kanan atas
3. Pencet deploy a public app from GitHub
4. Gunakan repository ini untuk di bagian Repository (
5. Pencet Deploy dan aplikasi bisa dipakai


