import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from transformers import pipeline
import torch
import os

# ======================================================================================
# --- KONFIGURASI APLIKASI ---
# ======================================================================================
st.set_page_config(
    page_title="Dashboard Optimasi IndoBERT",
    page_icon="🤖",
    layout="wide"
)

# ======================================================================================
# --- LOGIKA TEMA (DARK/LIGHT MODE) DENGAN TOGGLE ---
# ======================================================================================
st.sidebar.markdown("## 🧭 Navigasi")
mode = st.sidebar.toggle("Theme: ☀️ / 🌙")
page_selection = st.sidebar.selectbox(
    "Pilih Halaman:",
    ["Home", "Penelitian", "Demo Model", "Analisis Sentimen"]
)


# Tentukan tema & warna font untuk Plotly berdasarkan mode
theme_plotly = "plotly_dark" if mode else "plotly_white"
plotly_font_color = "#FAFAFA" if mode else "#31333F"

if mode:
    # CSS KHUSUS DARK MODE
    dark_css = """
    <style>
        /* Background Utama dan Sidebar */
        [data-testid="stAppViewContainer"] { background-color: #0E1117; color: #FAFAFA; }
        [data-testid="stSidebar"] { background-color: #262730; color: #FAFAFA; }
        [data-testid="stHeader"] { background-color: rgba(0,0,0,0); }
        h1, h2, h3, h4, h5, h6, p, span, div, label { color: #FAFAFA !important; }
        
        /* Selectbox Input */
        div[data-baseweb="select"] > div { background-color: #31333F !important; color: white !important; border-color: #555 !important; }
        
        /* Dropdown Menu Popover (List item saat selectbox diklik) */
        div[data-baseweb="popover"] > div { background-color: #262730 !important; border: 1px solid #444 !important;}
        div[data-baseweb="popover"] ul li { background-color: #262730 !important; color: #FAFAFA !important; }
        div[data-baseweb="popover"] ul li:hover { background-color: #4C4E5F !important; }
        
        /* Text Area */
        .stTextArea textarea { background-color: #31333F !important; color: white !important; }
        
        /* Tabel Statis (st.table) */
        table { color: white !important; background-color: #262730 !important; border: 1px solid #444 !important; }
        thead tr th { background-color: #0E1117 !important; color: white !important; border-bottom: 1px solid #444 !important; }
        tbody tr td { background-color: #262730 !important; color: white !important; border-bottom: 1px solid #444 !important; }
        
        /* Kotak Metrik (F1-Score, dll) */
        div[data-testid="stMetric"] { background-color: #262730; padding: 15px; border-radius: 10px; border: 1px solid #444; }
        
        /* st.code() Background */
        pre, code { background-color: #1E1E1E !important; color: #D4D4D4 !important; }
        
        /* st.expander() Background */
        [data-testid="stExpander"] { background-color: #262730 !important; border: 1px solid #444 !important; border-radius: 8px; }
        [data-testid="stExpander"] summary { background-color: #262730 !important; color: #FAFAFA !important; border-radius: 8px;}
        [data-testid="stExpander"] summary:hover { background-color: #31333F !important; }
        [data-testid="stExpander"] details[open] summary { border-bottom: 1px solid #444 !important; }
        
        /* st.button() Background */
        button[kind="secondary"] { background-color: #31333F !important; color: white !important; border: 1px solid #555 !important; }
        button[kind="secondary"]:hover { border-color: #ff4b4b !important; color: #ff4b4b !important; }

        /* Membalik urutan Toggle (Teks di kiri, Switch di kanan) */
        div[data-testid="stCheckbox"] label { flex-direction: row-reverse; justify-content: space-between; }
    </style>
    """
    st.markdown(dark_css, unsafe_allow_html=True)
else:
    # CSS KHUSUS LIGHT MODE (Reset ke default Streamlit)
    light_css = """
    <style>
        [data-testid="stAppViewContainer"] { background-color: #FFFFFF; color: #31333F; }
        [data-testid="stSidebar"] { background-color: #F0F2F6; color: #31333F; }
        [data-testid="stHeader"] { background-color: rgba(0,0,0,0); }
        h1, h2, h3, h4, h5, h6, p, span, div, label { color: #31333F !important; }
        div[data-testid="stMetric"] { background-color: #FFFFFF; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border: 1px solid #ddd; }

        /* Membalik urutan Toggle (Teks di kiri, Switch di kanan) */
        div[data-testid="stCheckbox"] label { flex-direction: row-reverse; justify-content: space-between; }
    </style>
    """
    st.markdown(light_css, unsafe_allow_html=True)

# ======================================================================================
# --- FUNGSI DATA & MODEL ---
# ======================================================================================

@st.cache_resource
def load_sentiment_pipeline():
    model_name = "uyahhh/indobert-coretax" 
    
    try:
        device = 0 if torch.cuda.is_available() else -1
        sentiment_task = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=device
        )
        return sentiment_task
    except Exception as e:
        # Menampilkan detail asli dari error kenapa gagal download
        st.error(f"⚠️ GAGAL MENDOWNLOAD MODEL: {str(e)}")
        return None

def load_hp_data():
    try:
        df_grid = pd.read_csv("grid-search.csv")
        df_random = pd.read_csv("random-search.csv")
        df_bayesian = pd.read_csv("bayesian.csv")
        return df_grid, df_random, df_bayesian
    except:
        return None, None, None

# ======================================================================================
# --- HALAMAN: HOME ---
# ======================================================================================

def page_home():
    st.title("Optimasi Hyperparameter IndoBERT")
    st.subheader("Analisis Sentimen Masyarakat terhadap Website Coretax")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Ringkasan Penelitian
        Penelitian ini bertujuan untuk mengoptimasi proses *fine-tuning* model **IndoBERT** menggunakan tiga metode pencarian hyperparameter yang berbeda. 
        Fokus utama adalah membandingkan efektivitas **Grid Search**, **Random Search**, dan **Bayesian Optimization** dalam meningkatkan performa klasifikasi sentimen (Positif, Netral, Negatif).

        1. **Model Dasar:** `indobenchmark/indobert-base-p1`
        2. **Dataset:** 30.181 tweet berlabel.
        3. **Optimasi:** 24 trial untuk setiap metode menggunakan library Optuna.
        4. **Hyperparameter:** Learning Rate, Epoch, Batch Size, dan Weight Decay.
        """)
    with col2:
        st.info("""
        **Info Penelitian:**\n
        - Dataset: 30.181 tweet ✅\n
        - Model: IndoBERT ✅\n
        - Dashboard: Streamlit ✅
        """)

# ======================================================================================
# --- HALAMAN: PENELITIAN ---
# ======================================================================================

def page_penelitian():
    st.title("🔬 Detail Eksperimen & Hasil")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset", "🏗️ Baseline Model", "📈 Proses Optimasi", "🏆 Perbandingan Akhir"])

    with tab1:
        st.subheader("Visualisasi Dataset")
        data_sentimen = pd.DataFrame({'Sentimen': ['Positive', 'Neutral', 'Negative'], 'Jumlah': [2086, 17632, 10474]})
        st.markdown("Berikut merupakan Visualisasi Dataset hasil Web Scraping pada Platform X mengenai Website Coretax berlabel dan telah dibersihkan yang akan dijadikan bahan pelatihan model. Jumlah keseluruhan data pada dataset yang digunakan berjumlah **30.181 tweet** dengan label Negative **10.474 tweet**, Neutral **17.632 tweet**, dan Positive **2086 tweet**.")

        col_a, col_b = st.columns(2)
        with col_a:
            # PLOTLY: Menambahkan font=dict(color=plotly_font_color)
            fig_pie = px.pie(data_sentimen, values='Jumlah', names='Sentimen', title="Distribusi Kelas Sentimen",
                             color_discrete_sequence=['#2ecc71', '#f1c40f', '#e74c3c'], template=theme_plotly)
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=plotly_font_color))
            st.plotly_chart(fig_pie)
            
        with col_b:
            st.write("<hr>", unsafe_allow_html=True)
            st.write("**Contoh Dataset:**")
            example_data = pd.DataFrame({
                'No': ['1', '2', '...', '30.180', '30.181'],
                'Tweet': ["lah iya sama gak mau kebuka coretax nya.", "selamat pagi dunia tipu tipu dimulai dengan coretax", "...","mjb aktifin lewat situs coretax bisa kak masuk ke akun dulu", "setidaknya liat coretax sambil joget dikit lah yah."],
                'Label': ['Negative', 'Negative','...', 'Neutral', 'Positive']
            })
            st.dataframe(example_data, hide_index='true')

    with tab2:
        st.markdown("Langkah selanjutnya adalah melakukan pelatihan Baseline model pre-trained `indobenchmark/indobert-base-p1` menggunakan dataset sebelumnya dengan metode fine-tuning untuk menambahkan lapisan klasifikasi untuk melakukan tugas analisis sentimen masyarakat pada platform X terhadap website pajak Coretax dengan konfigurasi Hyperparameter rekomendasi (Devlin dkk., 2019) tanpa optimasi.")
        st.markdown("Berikut merupakan Performa dari Baseline Model yang dijadikan dijadikan sebagai standar acuan untuk mengukur peningkatan performa setelah proses optimasi:")
        st.subheader("Performa Baseline Model")
        st.markdown("")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("F1-Score", "89,62%")
        c2.metric("Akurasi", "94,23%")
        c3.metric("Presisi", "89,34%")
        c4.metric("Recall", "89,90%")
        st.code("""
            # Konfigurasi Baseline
            learning_rate = 2e-5
            num_train_epochs = 3
            per_device_train_batch_size = 16
            weight_decay = 0.01
        """)

    with tab3:
        st.markdown("Proses Optimasi dilakukan dengan mencari kombinasi Hyperparameter terbaik menggunakan metode Grid Search, Random Search, dan Bayesian Optimization sebanyak 24 trial. Kemudian hasil Hyperparameter Terbaik setiap metode nya digunakan untuk fine-tuning terhadap model pre-trained untuk dibandingkan performanya.")
        st.markdown("Visualisasi Distribusi Pencarian dan Hyperparameter Terbaik untuk setiap metodenya ditunjukkan dibawah ini.")
        st.subheader("Visualisasi Pencarian Hyperparameter")
        df_grid, df_random, df_bayesian = load_hp_data()

        if df_grid is not None:
            method = st.selectbox("Pilih Metode untuk Visualisasi:", ["Grid Search", "Random Search", "Bayesian Optimization"])
            target_df = {"Grid Search": df_grid, "Random Search": df_random, "Bayesian Optimization": df_bayesian}[method]
            
            st.markdown(f"Hyperparameter Terbaik Metode {method}: ")
            try:
                target_column = 'F1-Score' 
                
                # Mengambil baris dengan nilai tertinggi
                best_row = target_df.loc[target_df[target_column].idxmax()]
                
                # Menampilkan nilai dalam bentuk kotak metrik
                col_hp2, col_hp3, col_hp4, col_hp5 = st.columns(4)
                col_hp2.metric("Learning Rate", f"{best_row['Learning_Rate']:.2e}")
                col_hp3.metric("Batch Size", int(best_row['Batch_Size']))
                col_hp4.metric("Epoch", int(best_row['Epoch']))
                col_hp5.metric("Weight Decay", f"{best_row['Weight_Decay']:.2f}")
                
            except KeyError:
                st.warning("⚠️ Gagal menampilkan data terbaik. Pastikan nama kolom target (misalnya 'F1-Score') pada kodingan sudah sesuai dengan nama kolom yang ada di dalam file CSV Anda.")

            fig_lr1 = px.line(target_df, x='trial', y='Learning_Rate', markers=True, title=f"Distribusi Pencarian Learning Rate ({method})", template=theme_plotly)
            fig_lr2 = px.line(target_df, x='trial', y='Batch_Size', markers=True, title=f"Distribusi Pencarian Batch Size ({method})", template=theme_plotly)
            fig_lr3 = px.line(target_df, x='trial', y='Epoch', markers=True, title=f"Distribusi Pencarian Jumlah Epoch ({method})", template=theme_plotly)
            fig_lr4 = px.line(target_df, x='trial', y='Weight_Decay', markers=True, title=f"Distribusi Pencarian Weight Decay ({method})", template=theme_plotly)
            
            for fig in [fig_lr1, fig_lr2, fig_lr3, fig_lr4]:
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=plotly_font_color))
                st.plotly_chart(fig, width='stretch')

    with tab4:
        st.subheader("Perbandingan Hasil Terbaik")
        st.markdown("Performa Model Hasil masing-masing Metode dibandingkan dengan Baseline Model.")
        st.markdown("Didapatkan bahwa model dengan nilai F1-Score terbesar adalah Model Hasil Metode **Bayesian Optimization** dengan nilai 91.72%")
        col_x, col_y = st.columns([2, 1])
        comparison_data = pd.DataFrame({
            'Metode': ['Baseline', 'Grid Search', 'Random Search', 'Bayesian'],
            'Best F1-Score': [89.62, 90.42, 90.64, 91.72],
        })
    
        with col_x:
            fig_comp = px.bar(comparison_data, x='Metode', y='Best F1-Score', range_y=[80, 95], color='Metode', text_auto=True, template=theme_plotly)
            fig_comp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=plotly_font_color))
            st.plotly_chart(fig_comp)
            
        with col_y:
            st.write("<br><br><hr>", unsafe_allow_html=True)
            st.table(comparison_data)

# ======================================================================================
# --- HALAMAN: DEMO MODEL ---
# ======================================================================================

def page_demo():
    st.title("🚀 Demo Analisis Sentimen")
    st.subheader("🏆 Performa & Konfigurasi Model Terbaik")
    st.markdown("Berikut merupakan keseluruhan performa dan detail kombinasi hyperparameter yang digunakan oleh Model Terbaik Hasil Optimasi Metode **Bayesian Optimization**")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("F1-Score", "91.72%")
    m2.metric("Akurasi", "95.00%")
    m3.metric("Presisi", "91.86%")
    m4.metric("Recall", "91.59%")

    with st.expander("🔍 Lihat Detail Hyperparameter Terpilih", expanded=True):
        h1, h2, h3, h4 = st.columns(4)
        h1.write("**Learning Rate**"); h1.code("1.04e-05")
        h2.write("**Epoch**"); h2.code("4")
        h3.write("**Batch Size**"); h3.code("16")
        h4.write("**Weight Decay**"); h4.code("0.22")

    st.markdown("---")
    st.write("### Uji Coba Model")
    st.markdown("Masukkan teks tweet mengenai website Coretax untuk dianalisis oleh model terbaik.")

    pipe = load_sentiment_pipeline()
    
    if pipe is not None:
        text_input = st.text_area("Input Tweet:", placeholder="Contoh: Coretax keren banget!")
        if st.button("Analisis Sentimen"):
            if text_input and text_input.strip():
                with st.spinner("Menganalisis..."):
                    try:
                        result = pipe(text_input.strip())[0]
                        label, score = result['label'], result['score']
                        if label.lower() == 'positive': st.success(f"**Hasil: Positif** (Keyakinan: {score:.2%})")
                        elif label.lower() == 'negative': st.error(f"**Hasil: Negatif** (Keyakinan: {score:.2%})")
                        else: st.info(f"**Hasil: Netral** (Keyakinan: {score:.2%})")
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat analisis: {str(e)}")

# ======================================================================================
# --- HALAMAN: ANALISIS SENTIMEN ---
# ======================================================================================

def page_sentimen():
    st.title("📈 Analisis Sentimen Publik (Coretax)")
    st.markdown("Halaman ini menampilkan analisis sentimen menggunakan model terbaik terhadap dataset tweet masyarakat mengenai Coretax pada platform X untuk mengetahui anggapan masyarakat terhadap Website Coretax")

    tab11, tab22 = st.tabs(["📊 Dataset", "🏗️ Hasil Sentimen"])
    with tab11:
        st.write("### Contoh Dataset")
        st.markdown("""
        Berikut merupakan contoh tweet pada dataset yang akan digunakan untuk memprediksi sentimen masyarakat pada aplikasi X. Dataset berjumlah **22196 tweet** setelah mengahapus tweet dari akun resmi pemerintah. 
        """)

        tweet_data = pd.DataFrame({
            'No': ['1','2','3','...','22195','22196'],
            'Created_at': ['Wed Jan 01 23:57:13 2025', 'Wed Jan 01 10:49:43 2025', 'Wed Jan 01 00:57:05 2025', '...', 'Thu Jan 02 14:42:17 2025', 'Thu Jan 02 14:18:17 2025'],
            'Tweet': ['Lah iya sama gak mau kebuka coretax nya', 'after deeply research coretax saya menyimpulkan tax consultant 2025 bakal laku keras dan naik daun bgt', 'Lho katanya pake coretax? Kok masih pake app e faktur?','...', 'Besok masih bergelut ama login coretax direktur yang belom bisa hhh dah mana draft invoice diubah juga', 'Sampe detik ini coretax msh blm normal jg kah ?'],
            'User_id_str': ['8427162818','12826609','119705358','...','1172503','1114044705']
        })
        st.dataframe(tweet_data, hide_index='true')

    with tab22:
        st.markdown("Berikut ini ditampilkan grafik distribusi kelas dan tabel jumlah tweet masyarakat mengenai Website Coretax")
        st.markdown("Dari grafik dan tabel tersebut dapat disimpulkan bahwa masyarakat pada media sosial X lebih banyak yang beranggapan **negatif** terhadap website Coretax dengan persentase **47%** dan tweet sebanyak **10428 tweet** dari total **22197 tweet**.")
        col1, col2 = st.columns([1, 1])
        summary_stats = pd.DataFrame({'Kategori': ['Negatif', 'Netral', 'Positif'], 'Jumlah': [10428, 9682, 2086]})
        
        with col1:
            fig_pie2 = px.pie(summary_stats, values='Jumlah', names='Kategori', title="Distribusi Kelas",
                              color_discrete_sequence=['#e74c3c', '#f1c40f', '#2ecc71'], template=theme_plotly)
            fig_pie2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=plotly_font_color))
            st.plotly_chart(fig_pie2)
        
        with col2:
            st.write("<br><hr>", unsafe_allow_html=True)
            st.write("**Jumlah Tweet Masyarakat Mengenai Website Coretax**")
            st.table(summary_stats)

# ======================================================================================
# --- EKSEKUSI HALAMAN UTAMA ---
# ======================================================================================
if page_selection == "Home":
    page_home()
elif page_selection == "Penelitian":
    page_penelitian()
elif page_selection == "Demo Model":
    page_demo()
elif page_selection == "Analisis Sentimen":

    page_sentimen()

