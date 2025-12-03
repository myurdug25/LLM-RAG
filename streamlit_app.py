import streamlit as st
import os
from search_engine import SearchEngine
import time
import psutil
import gc
from contextlib import contextmanager

def check_memory_usage():
    """Bellek kullanımını kontrol eder ve yüksekse uyarı verir"""
    try:
        process = psutil.Process(os.getpid())
        memory_percent = process.memory_percent()
        if memory_percent > 80:  # %80 üzeri bellek kullanımı
            st.warning(f"Yüksek bellek kullanımı: %{memory_percent:.1f}")
            # Bellek temizle
            gc.collect()
            return False
        return True
    except:
        return True  # Hata durumunda devam et

@contextmanager
def timeout(seconds):
    # Basitleştirilmiş timeout - sadece zaman ölçümü yapar
    start_time = time.time()
    yield
    elapsed = time.time() - start_time
    if elapsed > seconds:
        raise TimeoutError(f"İşlem {seconds} saniyeyi aştı (Geçen süre: {elapsed:.2f}s)")

# Sayfa yapılandırması
st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="🔍",
    layout="wide"
)

# Başlık
st.title("🔍 EKIP AI")

# Açıklama
st.markdown("""
Bu uygulama, belgelerde semantik arama yapmanızı sağlar. 
Aşağıya aramak istediğiniz ifadeyi yazın ve benzer içerikleri bulun.
""")

# Arama motorunu başlat
@st.cache_resource
def get_search_engine():
    engine = SearchEngine()
    return engine

engine = get_search_engine()

# Index durumunu kontrol et
if os.path.exists(engine.index_path) and os.path.exists(engine.metadata_path):
    try:
        if engine.load_index():
            st.success("✅ FAISS index başarıyla yüklendi.")
        else:
            st.warning("⚠️ Index yüklenemedi.")
    except Exception as e:
        st.warning("⚠️ Index yüklenirken hata oluştu.")
else:
    st.warning("⚠️ Index henüz oluşturulmamış.")
    st.info("Önce belgelerinizi ekleyin ve index oluşturun.")

# Sidebar ayarları
st.sidebar.header("⚙️ Ayarlar")
top_k = st.sidebar.slider("Gösterilecek Sonuç Sayısı", 1, 20, 5)

# Dosya yükleme bölümü
st.sidebar.header("📁 Belgeleri Yükle")
uploaded_files = st.sidebar.file_uploader(
    "PDF veya TXT dosyalarını yükleyin", 
    accept_multiple_files=True, 
    type=['pdf', 'txt']
)

if uploaded_files:
    st.sidebar.success(f"✅ {len(uploaded_files)} dosya yüklendi")
    
    # Belgeleri işle
    documents = []
    doc_names = []
    for uploaded_file in uploaded_files:
        doc_name = uploaded_file.name
        doc_names.append(doc_name)
        
        if uploaded_file.name.endswith('.txt'):
            content = uploaded_file.read().decode('utf-8')
            documents.append(content)
        elif uploaded_file.name.endswith('.pdf'):
            # PDF dosyayı geçici olarak kaydet
            temp_filename = f"temp_{uploaded_file.name}"
            with open(temp_filename, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # PDF'den metin çıkar
            try:
                text = engine.load_pdf(temp_filename)
                documents.append(text)
            finally:
                # Geçici dosyayı sil
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
    
    if st.sidebar.button("Index Oluştur"):
        with st.spinner("Index oluşturuluyor..."):
            try:
                engine.build_index(documents, doc_names)
                st.sidebar.success("✅ Index başarıyla oluşturuldu!")
                # Cache'i temizle
                get_search_engine.clear()
                # Streamlit cache'ini temizle
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"❌ Hata oluştu: {str(e)}")

# Belge seçimi
engine.load_index()  # Meta verileri yüklemek için
try:
    doc_list = engine.get_document_list()
    doc_options = {"Tüm Belgeler": None}
    if doc_list:
        for doc in doc_list:
            doc_options[f"{doc['name']} ({doc['chunk_count']} parça)"] = doc['doc_id']
    
    selected_doc = st.selectbox("Belge Seçin (Opsiyonel):", list(doc_options.keys()))
    selected_doc_id = doc_options[selected_doc] if selected_doc != "Tüm Belgeler" else None
except:
    selected_doc_id = None

# Ana arama bölümü
query = st.text_input("Soru veya arama ifadesi girin:", placeholder="Ne aramak istiyorsunuz?")

# İşlem tipi seçimi
operation_type = st.radio("İşlem Tipi:", ["Semantik Arama", "Soru Cevaplama", "Özet Çıkart"])

if st.button("İşlemi Gerçekleştir") or query:
    if query:
        with st.spinner("İşleniyor..."):
            try:
                if operation_type == "Semantik Arama":
                    if selected_doc_id is not None:
                        results = engine.search_with_document_filter(query, doc_id=selected_doc_id, k=top_k)
                        st.subheader(f"🔎 '{query}' için {selected_doc} belgesinde arama sonuçları:")
                    else:
                        results = engine.search(query, k=top_k)
                        st.subheader(f"🔎 '{query}' için arama sonuçları:")
                    
                    if results:
                        for i, result in enumerate(results, 1):
                            doc_name = result.get('doc_name', 'Bilinmiyor')
                            with st.expander(f"Sonuç #{i} (Skor: {result['score']:.4f}) - Belge: {doc_name}"):
                                st.write(result['text'])
                    else:
                        st.info("Sonuç bulunamadı.")
                
                elif operation_type == "Soru Cevaplama":
                    # İlk olarak ilgili içeriği bul
                    if selected_doc_id is not None:
                        results = engine.search_with_document_filter(query, doc_id=selected_doc_id, k=1)
                    else:
                        results = engine.search(query, k=1)
                    
                    if results:
                        context = results[0]['text']
                        
                        # Bağlam çok uzunsa kısalt
                        if len(context) > 1024:
                            st.warning("Bağlam çok uzun, ilk 1024 karakter kullanılacak")
                            context = context[:1024]
                        
                        # Timeout ile soru-cevap
                        start_time = time.time()
                        try:
                            with timeout(30):  # 30 saniye timeout
                                answer, score = engine.answer_question(context, query)
                            elapsed = time.time() - start_time
                            
                            st.subheader("❓ Soru:")
                            st.write(query)
                            
                            st.subheader("💬 Cevap:")
                            st.write(answer)
                            st.caption(f"Güven skoru: {score:.4f} | Süre: {elapsed:.2f}s")
                            
                            with st.expander("Bağlam (Context)"):
                                st.write(context)
                        except TimeoutError as te:
                            st.error(f"Soru cevaplama işlemi zaman aşımına uğradı: {str(te)}")
                        except Exception as e:
                            st.error(f"Soru cevaplama hatası: {str(e)}")
                            st.exception(e)  # Detaylı hata bilgisi
                    else:
                        st.info("İlgili içerik bulunamadı.")
                
                elif operation_type == "Özet Çıkart":
                    # Bellek kontrolü
                    if not check_memory_usage():
                        st.error("Yüksek bellek kullanımı nedeniyle işlem iptal edildi. Lütfen uygulamayı yeniden başlatın.")
                    else:
                        # Tüm belgeleri birleştir
                        if hasattr(engine, 'docs') and engine.docs:
                            try:
                                # Belge bazlı özetleme yap
                                st.info("Belgeler özetleniyor...")
                                all_summaries = []
                                
                                for i, doc in enumerate(engine.docs):
                                    doc_text = doc['text']
                                    doc_name = doc.get('doc_name', f'Belge {i+1}')
                                    
                                    # Her belge için daha uzun metin kullan
                                    max_chars = 4000  # Önceki 2000 yerine
                                    if len(doc_text) > max_chars:
                                        st.info(f"{doc_name} belgesi kısaltılıyor...")
                                        doc_text = doc_text[:max_chars]
                                    
                                    if len(doc_text.strip()) > 0:
                                        # Her belgeyi ayrı özetle
                                        summary = engine.summarize(doc_text, max_length=300, min_length=100)
                                        if summary and not "hata" in summary.lower():
                                            all_summaries.append(f"**{doc_name}:**\n{summary}\n")
                                        else:
                                            # Basit özet
                                            words = doc_text.split()
                                            if len(words) > 100:
                                                simple_summary = " ".join(words[:100]) + "..."
                                                all_summaries.append(f"**{doc_name} (Basit Özet):**\n{simple_summary}\n")
                                
                                # Tüm özetleri birleştir
                                if all_summaries:
                                    st.subheader("📋 Belgelerin Özetleri:")
                                    for summary in all_summaries:
                                        st.write(summary)
                                        st.divider()  # Özetler arasında çizgi
                                else:
                                    st.warning("Özet oluşturulamadı.")
                                    
                                # Bellek temizle
                                gc.collect()
                                
                            except Exception as e:
                                st.error(f"Özetleme hatası: {str(e)}")
                                st.info("Not: Uygulama çok büyük metinlerde kararsız çalışabilir. Daha küçük belgeler deneyin.")
                        else:
                            st.info("Özetlenecek içerik bulunamadı. Önce belgeleri yükleyin ve index oluşturun.")
                        
            except Exception as e:
                st.error(f"❌ Hata oluştu: {str(e)}")
    else:
        st.warning("Lütfen bir soru veya arama ifadesi girin.")

# Bilgi kutusu
st.sidebar.header("💡 Nasıl Çalışır?")
st.sidebar.markdown("""
1. **Belge Yükleme**: PDF veya TXT dosyalarınızı yükleyin
2. **Index Oluşturma**: Belgelerden vektör index'i oluşturun
3. **Arama Yapma**: İlgili içerikleri semantik olarak bulun
""")