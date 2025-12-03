# Semantic Search Engine

Bu proje, belgelerde semantik arama yapmayı sağlayan bir sistemdir. FAISS vektör veritabanı ve sentence-transformers kullanarak belgelerinizi işler ve anlamsal olarak benzer içerikleri bulmanızı sağlar.

## Özellikler

- PDF ve TXT dosyalarını destekler
- Semantik arama (anlam bazlı)
- Soru-cevap yeteneği
- Otomatik özetleme
- Streamlit ile kullanıcı dostu arayüz
- FastAPI ile RESTful API

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Belgelerinizi `data/` dizinine ekleyin

3. Index oluşturun:
```bash
python build_index.py
```

4. Uygulamayı başlatın:
```bash
streamlit run streamlit_app.py
```

Veya API sunucusunu başlatmak için:
```bash
uvicorn app:app --reload
```

## Kullanım

1. Streamlit uygulamasını açın
2. PDF veya TXT dosyalarınızı yükleyin
3. "Index Oluştur" butonuna tıklayın
4. Üç farklı işlem seçeneği vardır:
   - **Semantik Arama**: Anahtar kelimeye göre benzer içerikleri bulur
   - **Soru Cevaplama**: Yüklediğiniz belgelerdeki bilgilerle sorularınızı cevaplar
   - **Özet Çıkart**: Tüm belgelerinizi özetler

## Teknolojiler

- FAISS: Vektör benzerliği araması
- sentence-transformers: Metin gömme (embedding)
- Transformers: Özetleme ve soru-cevap modelleri
- FastAPI: REST API
- Streamlit: Web arayüzü
- PyPDF2: PDF işleme

## Proje Yapısı

```
.
├── app.py              # FastAPI sunucu
├── build_index.py      # Index oluşturma scripti
├── search_engine.py    # Arama motoru
├── streamlit_app.py    # Streamlit arayüzü
├── requirements.txt    # Bağımlılıklar
├── data/               # Belgelerin bulunduğu dizin
└── index/              # Oluşturulan index dosyaları
```