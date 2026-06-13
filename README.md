```# Titanic Survival Prediction CI/CD Pipeline

Proyek ini adalah implementasi pipeline CI/CD untuk model machine learning menggunakan MLflow, GitHub Actions, dan Docker. Model dilatih secara otomatis setiap kali ada perubahan kode (push) ke repositori, kemudian di-build menjadi Docker image yang siap untuk dideploy.

## Project Structure
```text
└── MLProject
    ├── modelling.py              # Script training & logging model
    ├── requirements.txt          # Library dependencies untuk model
    ├── Tautan ke Docker Hub.txt  # Link ke Docker Image
    └── titanic_clean.csv         # Dataset yang digunakan

```

## Teknologi yang Digunakan

* **Python 3.12**
* **MLflow 2.19.0**: Untuk tracking eksperimen dan packaging model.
* **Scikit-Learn**: Untuk algoritma Random Forest.
* **GitHub Actions**: Sebagai automation server (CI/CD).
* **Docker**: Untuk containerization model.

## Pipeline Otomatisasi (GitHub Actions)

Setiap kali ada `push` ke branch `main` atau `master`:

1. **Setup Environment**: Menginstal Python dan library yang diperlukan.
2. **Training**: Script `modelling.py` dijalankan untuk melatih model dan mencatat run_id.
3. **Artifact Logging**: Hasil training disimpan di folder `mlruns`.
4. **Docker Build & Push**: Model di-package menjadi Docker image dengan `env-manager=local` dan di-push ke Docker Hub secara otomatis.

## Cara Penggunaan

1. Pastikan Docker Hub credentials (`DOCKER_USERNAME` & `DOCKER_PASSWORD`) sudah terpasang di GitHub Secrets repositori ini.
2. Setiap kali kode di-push, cek tab **Actions** di GitHub untuk memantau proses build.
3. Docker image yang berhasil akan muncul di Docker Hub (link tertera di file `Tautan ke Docker Hub.txt`).

```
