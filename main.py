import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import chi2
import io
from adjustText import adjust_text
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.metrics.pairwise import euclidean_distances

# -------------------------------------------------------------------
# Konfigurasi Halaman Streamlit
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Analisis Cluster K-Means Interaktif",
    page_icon="✨",
    layout="wide"
)

# -------------------------------------------------------------------
# Fungsi-Fungsi Bantuan
# -------------------------------------------------------------------

@st.cache_data
def load_data(uploaded_file):
    """Memuat data dari file Excel yang diunggah."""
    try:
        df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Gagal memproses file Excel: {e}")
        return None

@st.cache_data
def run_pca_and_elbow(data_selected, file_identifier):
    """Menjalankan PCA dan menghitung inertia untuk Elbow Method."""
    data_transposed = data_selected.T
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_transposed)
    
    pca = PCA(n_components=2, random_state=42)
    data_pca = pca.fit_transform(data_scaled)
    explained_variance = sum(pca.explained_variance_ratio_) * 100
    
    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(data_pca)
        inertia.append(kmeans.inertia_)
        
    pca_df = pd.DataFrame(data_pca, columns=['PC1_Effectiveness', 'PC2_Satisfaction'], index=data_transposed.index)
        
    return pca_df, explained_variance, inertia, k_range

def draw_confidence_ellipse(ax, points, color):
    """Menggambar elips kepercayaan di sekitar cluster."""
    if len(points) < 2: return
    centroid = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * np.sqrt(chi2.ppf(0.95, df=2) * eigenvalues)
    ellipse = patches.Ellipse(xy=centroid, width=width, height=height, angle=angle, facecolor=color, alpha=0.15, edgecolor=color, lw=1.5, linestyle='--')
    ax.add_patch(ellipse)

def to_excel(df):
    """Mengonversi DataFrame ke format Excel (in-memory)."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=True, sheet_name='Hasil Analisis')
    return output.getvalue()

# -------------------------------------------------------------------
# Tampilan Utama Aplikasi Streamlit
# -------------------------------------------------------------------

st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>✨ Aplikasi Analisis K-Means Clustering ✨</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- Sidebar untuk Unggah Data ---
st.sidebar.title("Langkah 1: Unggah Data")
uploaded_file = st.sidebar.file_uploader("Pilih file Excel Anda", type=["xlsx", "xls"])

st.sidebar.markdown("---")
st.sidebar.write("Tidak punya data? Unduh contoh format di bawah ini.")
try:
    with open("data.xlsx", "rb") as file:
        st.sidebar.download_button(
            label="📥 Unduh Contoh Dataset",
            data=file,
            file_name="data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
except FileNotFoundError:
    st.sidebar.warning("File contoh 'data.xlsx' tidak ditemukan.")

# --- Tampilan Utama Setelah Unggah ---
if uploaded_file is None:
    st.info("👋 Selamat datang! Silakan mulai dengan mengunggah file Excel Anda melalui sidebar di sebelah kiri.")
else:
    df_raw = load_data(uploaded_file)
    
    try:
        df_numeric = df_raw.select_dtypes(include=np.number)

        if df_numeric.empty:
            st.error("❌ Tidak ditemukan kolom numerik dalam file yang diunggah. Pastikan data Anda berisi angka untuk analisis.")
        else:
            missing_values_count = df_numeric.isnull().sum().sum()
            if missing_values_count > 0:
                # --- PERUBAHAN: Mengganti Missing Value dengan Modus ---
                # Menghitung modus untuk setiap kolom. .iloc[0] digunakan untuk mengambil nilai pertama jika ada lebih dari satu modus.
                modes = df_numeric.mode().iloc[0]
                data_selected = df_numeric.fillna(modes)
                st.success(f"✅ Ditemukan dan diganti **{int(missing_values_count)}** nilai yang hilang dengan **modus** dari masing-masing kolom.")
            else:
                data_selected = df_numeric.copy()
                st.info("ℹ️ Tidak ada nilai yang hilang (missing values) pada kolom numerik.")

            if 'pca_results' not in st.session_state or st.session_state.get('last_file_name') != uploaded_file.name:
                with st.spinner('Menjalankan analisis PCA dan menghitung Elbow Method untuk file baru...'):
                    pca_df, explained_variance, inertia, k_range = run_pca_and_elbow(data_selected, uploaded_file.name)
                    st.session_state.pca_results = {
                        "pca_df": pca_df,
                        "explained_variance": explained_variance,
                        "inertia": inertia,
                        "k_range": k_range
                    }
                st.session_state.last_file_name = uploaded_file.name
            
            pca_df = st.session_state.pca_results["pca_df"]
            explained_variance = st.session_state.pca_results["explained_variance"]
            inertia = st.session_state.pca_results["inertia"]
            k_range = st.session_state.pca_results["k_range"]

            st.header("🎯 Pra-Analisis & Penentuan Cluster")
            
            with st.expander("📂 Klik untuk melihat pratinjau data dan hasil PCA"):
                st.write(f"**Nama File:** `{uploaded_file.name}`")
                st.write(f"**Jumlah Baris:** `{data_selected.shape[0]}` | **Jumlah Kolom (Numerik):** `{data_selected.shape[1]}`")
                st.write(f"**Total Varians yang dijelaskan oleh PCA:** `{explained_variance:.2f}%`")
                st.dataframe(pca_df, use_container_width=True)
                excel_data_pca = to_excel(pca_df.round(4))
                st.download_button(
                    label="📥 Unduh Hasil PCA sebagai Excel",
                    data=excel_data_pca,
                    file_name='hasil_reduksi_dimensi_pca.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )

            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("Metode Elbow")
                fig_elbow, ax_elbow = plt.subplots()
                ax_elbow.plot(k_range, inertia, marker='o', linestyle='--')
                ax_elbow.set_xlabel('Jumlah Cluster (k)')
                ax_elbow.set_ylabel('SSE (Inertia)')
                ax_elbow.set_title('Elbow Method untuk Menentukan k Optimal')
                ax_elbow.grid(True)
                st.pyplot(fig_elbow, use_container_width=True)
                
            with col2:
                st.subheader("Pilih Jumlah Cluster")
                st.write("Berdasarkan grafik metode elbow di sebelah kiri, tentukan jumlah 'siku' atau titik di mana penurunan tidak lagi signifikan.")
                optimal_k = st.number_input("Masukkan jumlah cluster (k) pilihan Anda:", min_value=2, max_value=10, value=3, step=1)
                st.info(f"Anda memilih untuk menganalisis dengan **k = {optimal_k}** cluster.")
                
                st.sidebar.markdown("---")
                st.sidebar.title("Langkah 2: Pengaturan Grafik")
                plot_width = st.sidebar.slider("Lebar Grafik", 5, 20, 10)
                plot_height = st.sidebar.slider("Tinggi Grafik", 3, 20, 8)

            st.markdown("---")
            st.header(f"🚀 Hasil Akhir Analisis Clustering (k = {optimal_k})")

            kmeans_model = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42)
            cluster_labels = kmeans_model.fit_predict(pca_df)
            results_df = pca_df.copy()
            results_df['Cluster'] = cluster_labels
            
            final_centroids = kmeans_model.cluster_centers_

            tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualisasi Cluster", "📋 Data Profiling", "📈 Performa Cluster", "⭐ Evaluasi Model"])

            with tab1:
                fig_cluster, ax_cluster = plt.subplots(figsize=(plot_width, plot_height))
                palette = sns.color_palette('deep', n_colors=optimal_k)
                
                sns.scatterplot(x='PC1_Effectiveness', y='PC2_Satisfaction', hue='Cluster', data=results_df, palette=palette, s=150, alpha=0.7, ax=ax_cluster, zorder=2)
                ax_cluster.scatter(final_centroids[:, 0], final_centroids[:, 1], s=500, c='yellow', marker='*', edgecolor='black', linewidth=1.5, label='Centroid', zorder=10)
                
                for cluster_num in range(optimal_k):
                    cluster_points = results_df.loc[results_df['Cluster'] == cluster_num, ['PC1_Effectiveness', 'PC2_Satisfaction']]
                    draw_confidence_ellipse(ax_cluster, cluster_points, color=palette[cluster_num])

                texts = []
                for i, row in results_df.iterrows():
                    short_txt = str(i).split('_')[0]
                    texts.append(ax_cluster.text(row['PC1_Effectiveness'], row['PC2_Satisfaction'], short_txt, fontsize=8, ha='center', va='center'))
                
                adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

                ax_cluster.set_title(f'Visualisasi {optimal_k} Cluster', fontsize=18)
                ax_cluster.set_xlabel('Effectiveness (PC1)', fontsize=14)
                ax_cluster.set_ylabel('Satisfaction (PC2)', fontsize=14)
                ax_cluster.legend(title='Kluster')
                plt.tight_layout()
                st.pyplot(fig_cluster)

            with tab2:
                st.dataframe(results_df.sort_values(by='Cluster'), use_container_width=True)
                excel_data_profiling = to_excel(results_df.sort_values(by='Cluster').round(4))
                st.download_button(
                    label="📥 Unduh Hasil Profiling sebagai Excel",
                    data=excel_data_profiling,
                    file_name=f'hasil_profiling_k{optimal_k}.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )

            with tab3:
                cluster_performance = results_df.groupby('Cluster')[['PC1_Effectiveness', 'PC2_Satisfaction']].mean()
                fig_bar, ax_bar = plt.subplots(figsize=(plot_width, plot_height))

                cluster_performance.columns = ['Effectiveness (PC1)', 'Satisfaction (PC2)']
                
                cluster_performance.plot(kind='bar', ax=ax_bar, color=['#4B8BBE', '#F08A5D'], rot=0)
                
                ax_bar.set_title('Perbandingan Rata-Rata Performa per Cluster', fontsize=16)
                ax_bar.set_ylabel('Skor Rata-rata PCA', fontsize=12)
                ax_bar.set_xlabel('Nomor Cluster', fontsize=12)
                ax_bar.axhline(0, color='black', linewidth=0.8)
                
                ax_bar.legend(fontsize=10)
                
                for p in ax_bar.patches:
                    ax_bar.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 9), textcoords='offset points')
                plt.tight_layout()
                st.pyplot(fig_bar)
            
            with tab4:
                st.subheader(f"Evaluasi Model dengan Silhouette Plot (k={optimal_k})")
                st.markdown("""
                Diagram Silhouette menampilkan seberapa mirip sebuah titik data dengan clusternya sendiri dibandingkan dengan cluster lain.
                - **Lebar bilah (koefisien silhouette)**: Menunjukkan seberapa baik titik data cocok dalam clusternya. Semakin mendekati +1, semakin baik.
                - **Garis putus-putus merah**: Menunjukkan rata-rata Silhouette Score untuk semua data.
                - **Ketebalan bilah yang seragam**: Menandakan ukuran cluster yang seimbang dan baik.
                """)

                # --- Membuat Visualisasi Silhouette Menggunakan Yellowbrick ---
                fig_sil, ax_sil = plt.subplots(figsize=(plot_width, plot_height))
                visualizer = SilhouetteVisualizer(kmeans_model, colors='yellowbrick', ax=ax_sil)
                
                # Fit visualizer ke data
                visualizer.fit(pca_df)        
                visualizer.finalize() # Menggambar plot
                
                ax_sil.set_title(f"Diagram Silhouette untuk {optimal_k} Cluster", fontsize=16)
                st.pyplot(fig_sil)

                # --- Menghitung dan Menampilkan Skor Akhir Menggunakan Scikit-learn ---
                silhouette_avg = silhouette_score(pca_df, cluster_labels)
                
                st.markdown("---")
                st.subheader("Hasil Akhir Skor Silhouette")
                st.metric(label="Rata-rata Skor Silhouette (Keseluruhan)", value=f"{silhouette_avg:.4f}")

                if silhouette_avg > 0.5:
                    st.success("✅ Kualitas cluster **Baik**. Cluster padat dan terpisah dengan baik.")
                elif silhouette_avg > 0.25:
                    st.info("ℹ️ Kualitas cluster **Cukup Baik**. Ada struktur yang jelas dalam data.")
                else:
                    st.warning("⚠️ Kualitas cluster **Lemah**. Cluster mungkin tumpang tindih atau tidak signifikan.")


    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data: {e}")
        st.warning("Pastikan file Excel Anda memiliki format yang benar dan berisi kolom-kolom numerik untuk dianalisis.")