Semantic Search Engine
Bu proje, belgelerde semantik arama yapmayı sağlayan bir sistemdir. FAISS vektör veritabanı ve sentence-transformers kullanarak belgelerinizi işler ve anlamsal olarak benzer içerikleri bulmanızı sağlar.

Özellikler
PDF ve TXT dosyalarını destekler
Semantik arama (anlam bazlı)
Soru-cevap yeteneği
Otomatik özetleme
Streamlit ile kullanıcı dostu arayüz
FastAPI ile RESTful API
Kurulum
Gerekli paketleri yükleyin:
pip install -r requirements.txt
Belgelerinizi data/ dizinine ekleyin

Index oluşturun:

python build_index.py
Uygulamayı başlatın:
streamlit run streamlit_app.py
Veya API sunucusunu başlatmak için:

uvicorn app:app --reload
Kullanım
Streamlit uygulamasını açın
PDF veya TXT dosyalarınızı yükleyin
"Index Oluştur" butonuna tıklayın
Üç farklı işlem seçeneği vardır:
Semantik Arama: Anahtar kelimeye göre benzer içerikleri bulur
Soru Cevaplama: Yüklediğiniz belgelerdeki bilgilerle sorularınızı cevaplar
Özet Çıkart: Tüm belgelerinizi özetler
Teknolojiler
FAISS: Vektör benzerliği araması
sentence-transformers: Metin gömme (embedding)
Transformers: Özetleme ve soru-cevap modelleri
FastAPI: REST API
Streamlit: Web arayüzü
PyPDF2: PDF işleme
Proje Yapısı
.
├── app.py              # FastAPI sunucu
├── build_index.py      # Index oluşturma scripti
├── search_engine.py    # Arama motoru
├── streamlit_app.py    # Streamlit arayüzü
├── requirements.txt    # Bağımlılıklar
├── data/               # Belgelerin bulunduğu dizin
└── index/              # Oluşturulan index dosyaları
