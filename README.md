# CNN, RNN, and LSTM From Scratch - IF3270 Machine Learning

Repositori ini merupakan hasil dari Tugas Besar II mata kuliah IF3270 Pembelajaran Mesin. Proyek ini bertujuan untuk mengimplementasikan **Convolutional Neural Network (CNN), Recurrent Neural Network (RNN), dan Long Short Term Memory (LSTM)** *from scratch*.

## 🚀 Deskripsi Singkat

CNN, RNN, dan LSTM yang kami implementasikan memiliki fitur-fitur sebagai berikut:
- Arsitektur fleksibel (jumlah layer dan neuron dapat disesuaikan)
- Forward propagation untuk batch input
- Performa sesuai dengan model dari library Keras
- RNN dan LSTM yang bisa dijalankan dengan Unidirectional atau Bidirectional
- CNN yang bisa dijalankan dengan menggunakan Max Pooling atau Average Pooling
- Jenis-jenis layer lainnya yang mendukung jalannya CNN, RNN, dan LSTM seperti Fully Connected (FFNN), Dropout, Embedded, ReLU, dll

---

## 🔠 Cara Setup dan Menjalankan Program

1. **Clone repository**
   ```bash
   git clone https://github.com/scifo04/CNN-RNN-LSTM.git
   cd src
   ```

2. **Install dependencies**
   ```bash
   pip install numpy scikit-learn keras matplotlib
   ```

3. **Gunakan Model dalam notebook Anda**
   Usage
   ```python
   # CNN
   cnn_model = [
    Conv2D(...),
    ReLU(...),
    Pooling(...),
    .
    .
    .
   ]

   # Menjalankan Program CNN (Transpose seperti contoh dibawah ini untuk Conv2D dan Pooling untuk menyamakan dengan Keras)
   # Pastikan set training status dari Dropout adalah False jika ingin membandingkan model dengan Keras
   x = cnn_model[0].forward(x.transpose(2,0,1)).transpose(1,2,0)
   x = cnn_model[1].forward(x)
   x = cnn_model[2].forward(x.transpose(2,0,1)).transpose(1,2,0)
   .
   .
   .

   # RNN
   rnn_model = [
    RNN(...),
    .
    .
    FullyConnected(...),\
   ]
   
   # Menjalankan RNN
   for model in rnn_model:
    x = model.forward(x)

   # LSTM
   lstm_model = [
    LSTM(...),
    .
    .
    FullyConnected(...),
   ]
   
   # Menjalankan LSTM
   for model in lstm_model:
    x = model.forward(x)
    

4. **Jalankan Notebook Anda**

   Jika menggunakan Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
   Lalu buka file berikut:
   ```
   src/cnn.ipynb
   src/rnn.ipynb
   src/lstm.ipynb
   ```

---

## 👥 Pembagian Tugas Anggota Kelompok

| Nama | NIM | Tugas |
|------|-----|-------|
| Suthasoma Mahardhika Munthe | 13522098 | Laporan |
| Marvin Scifo Hutahaean | 13522110 | Implementasi CNN, RNN, dan LSTM, Laporan |
| Berto Richardo Togatorop | 13522118 | Laporan |

---

