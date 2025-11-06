import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt

st.set_page_config(page_title="Operasi Citra Monokrom", layout="centered")
st.title("Operasi Dasar Citra Monokrom")

st.write("Aplikasi ini menghitung dan menampilkan hasil operasi dasar pada citra monokrom.")

# -----------------------------------
# Bagian 1 – Efek operasi skalar
# -----------------------------------
st.header("6.1–6.4 Operasi Skalar pada Citra Monokrom")

# Citra contoh 3x3
img = np.array([[50, 100, 150],
                [200, 220, 240],
                [10, 30, 60]], dtype=np.uint8)

st.write("Citra Asli (nilai piksel):")
st.write(img)

# Operasi skalar
add_const = cv2.add(img, 50)
sub_const = cv2.subtract(img, 50)
mul_gt1 = cv2.multiply(img, 1.5)
mul_lt1 = cv2.multiply(img, 0.5)

# Tampilkan hasil
col1, col2 = st.columns(2)
with col1:
    st.image(add_const, caption="(6.1) Tambah konstanta positif (+50)", clamp=True, channels="GRAY")
    st.image(sub_const, caption="(6.2) Kurang konstanta positif (-50)", clamp=True, channels="GRAY")
with col2:
    st.image(mul_gt1, caption="(6.3) Kali konstanta >1.0 (×1.5)", clamp=True, channels="GRAY")
    st.image(mul_lt1, caption="(6.4) Kali konstanta <1.0 (×0.5)", clamp=True, channels="GRAY")

# -----------------------------------
# Bagian 2 – Operasi Logika
# -----------------------------------
st.header("6.5 Operasi Logika AND, OR, XOR")

X = np.array([[200, 100, 100],
              [0,   10,  50],
              [50, 250, 120]], dtype=np.uint8)

Y = np.array([[100, 220, 230],
              [45,   95, 120],
              [50,  250, 120]], dtype=np.uint8)  # baris ke-3 disesuaikan agar 3x3

# Konversi ke biner
X_bin = (X > 0).astype(np.uint8)
Y_bin = (Y > 0).astype(np.uint8)

and_img = cv2.bitwise_and(X_bin, Y_bin)
or_img  = cv2.bitwise_or(X_bin, Y_bin)
xor_img = cv2.bitwise_xor(X_bin, Y_bin)

st.write("Matriks X (biner):")
st.write(X_bin)
st.write("Matriks Y (biner):")
st.write(Y_bin)

col3, col4, col5 = st.columns(3)
with col3:
    st.image(and_img*255, caption="(a) X AND Y", clamp=True, channels="GRAY")
with col4:
    st.image(or_img*255, caption="(b) X OR Y", clamp=True, channels="GRAY")
with col5:
    st.image(xor_img*255, caption="(c) X XOR Y", clamp=True, channels="GRAY")

# -----------------------------------
# Bagian 3 – Operasi Tambahan (soal lanjutan)
# -----------------------------------
st.header("Operasi Tambahan (Soal Lanjutan)")

img_uint8 = np.array([[0, 50, 100, 150, 200, 255]], dtype=np.uint8)
add_self = cv2.add(img_uint8, img_uint8)
mul_self = cv2.multiply(img_uint8, img_uint8, scale=1/255)

binary_img = np.array([[0, 1, 0]], dtype=np.uint8)
binary_mul = cv2.multiply(binary_img, binary_img)

A = np.array([[50, 100, 150]], dtype=np.uint8)
B = np.array([[60, 80, 140]], dtype=np.uint8)
diff_sub = cv2.subtract(A, B)
diff_abs = cv2.absdiff(A, B)
diff_xor = cv2.bitwise_xor(A, B)
B_safe = B.astype(np.float32) + 1e-5
diff_div = (A.astype(np.float32) / B_safe)

st.subheader("(1) Tambah citra dengan dirinya sendiri")
st.write(add_self)

st.subheader("(2) Kali citra dengan dirinya sendiri")
st.write(mul_self)

st.subheader("(3 & 4) Kali citra biner [0,1,0] dengan dirinya sendiri")
st.write(binary_mul)

st.subheader("(5) Bandingkan cara mencari selisih antar citra")
st.write("A =", A)
st.write("B =", B)
col6, col7 = st.columns(2)
with col6:
    st.write("Pengurangan (A - B):")
    st.write(diff_sub)
    st.write("Selisih absolut |A - B|:")
    st.write(diff_abs)
with col7:
    st.write("XOR (bitwise):")
    st.write(diff_xor)
    st.write("Pembagian A/B:")
    st.write(diff_div)

st.success("✅ Semua operasi selesai dihitung dan divisualisasikan.")
