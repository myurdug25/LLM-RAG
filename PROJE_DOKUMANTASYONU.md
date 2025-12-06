# 📚 Proje Dokümantasyonu - Semantic Search Engine

Bu dokümantasyon, projedeki tüm dosyaların ne işe yaradığını, hangi fonksiyonların ne yaptığını ve sistemin nasıl çalıştığını detaylı olarak açıklar.

---

## 📁 Proje Yapısı

```
advanced_ir/
├── app.py                  # FastAPI REST API sunucusu
├── build_index.py          # Index oluşturma scripti
├── search_engine.py        # Ana arama motoru sınıfı
├── streamlit_app.py        # Streamlit web arayüzü
├── requirements.txt        # Python bağımlılıkları
├── README.md              # Genel proje açıklaması
├── data/                  # Belgelerin bulunduğu dizin
│   ├── makine_ogrenmesi.txt
│   ├── ornek_belge.txt
│   └── yapay_zeka.txt
└── index/                 # Oluşturulan index dosyaları
    ├── faiss.index        # FAISS vektör index'i
    ├── metadata.json      # Chunk metadata
    └── doc_metadata.json  # Belge metadata
```

---

## 📖 TEMEL KAVRAMLAR - Detaylı Açıklama

### 🔸 CHUNK (Parça) Nedir?

**Basit Açıklama:** Chunk, büyük bir belgenin daha küçük parçalara bölünmüş halidir. Tıpkı bir kitabı bölümlere ayırmak gibi!

**Neden Gerekli?**

1. **Model Sınırlamaları:**
   - Transformer modelleri (BERT, vb.) genellikle maksimum 512 token (yaklaşık 400-500 kelime) kabul eder
   - Uzun belgeleri tek seferde işleyemezler

2. **Hassas Arama:**
   - Belgenin sadece ilgili kısmını bulmak için
   - Örnek: "Makine öğrenmesi nedir?" sorusuna 1000 sayfalık kitaptan sadece ilgili paragrafı döndürmek

3. **Hızlı İşleme:**
   - Küçük parçalar daha hızlı işlenir
   - Gereksiz bilgileri filtreler

**Gerçek Örnek:**

Diyelim ki elimizde şu belge var (makine_ogrenmesi.txt):

```
Makine Öğrenmesine Giriş

1. Makine Öğrenmesi Nedir?
Makine öğrenmesi (ML), bilgisayarların açıkça programlanmadan veriden öğrenmesini sağlayan algoritmalar bütünüdür...

2. Makine Öğrenmesi Türleri
Makine öğrenmesi genel olarak üç ana kategoriye ayrılır...

3. Gözetimli Öğrenme
Bu yöntemde algoritma, hem girdileri hem de çıktıları içeren etiketli verilerle eğitilir...
```

**200 kelimelik chunk'lara bölündüğünde:**

```
CHUNK 1 (doc_id: 0, chunk_id: 0):
"Makine Öğrenmesine Giriş 1. Makine Öğrenmesi Nedir? Makine öğrenmesi (ML)... [200 kelime]"

CHUNK 2 (doc_id: 0, chunk_id: 1):
"... [kalan kelimeler, 200 kelime]"

CHUNK 3 (doc_id: 0, chunk_id: 2):
"... [devamı, 200 kelime]"
```

**Metadata'da Nasıl Görünür?**

```json
{
    "doc_id": 0,                          // Hangi belge?
    "chunk_id": 0,                        // Belgenin kaçıncı parçası?
    "text": "Makine öğrenmesi (ML)...",   // Parçanın içeriği
    "doc_name": "makine_ogrenmesi.txt",   // Belge adı
    "doc_hash": "eea0f046..."             // Belgenin benzersiz kodu
}
```

**Chunk Boyutu Neden 200 Kelime?**

- **Çok küçük (50-100 kelime):** Çok fazla parça oluşur, arama yavaşlar
- **Orta (200 kelime):** İyi dengeli, yeterli context + hızlı arama
- **Büyük (500+ kelime):** Model sınırlarını zorlar, daha az hassas

---

### 🔸 FAISS INDEX Nedir ve İçinde Ne Var?

**Basit Açıklama:** FAISS index, metinlerin matematiksel gösterimlerini (vektörler) saklayan ve hızlı arama yapmamızı sağlayan bir veritabanıdır.

**Analoji:** 
- Normal arama = Kitapta kelime kelime aramak (yavaş)
- FAISS index = Her sayfanın özetini numaralı kartlarda saklamak, kart numarasına göre hızlı bulmak (çok hızlı)

**FAISS Index İçeriği - Detaylı Açıklama:**

FAISS index dosyası (`index/faiss.index`) bir **binary (ikili) dosyadır**. İnsanlar tarafından doğrudan okunamaz, sadece FAISS kütüphanesi ile okunabilir.

**1. Dosya Formatı:**
- **Tip:** Binary (ikili)
- **Açıklama:** Normal metin dosyası değil, özel format
- **Okuma:** Sadece `faiss.read_index()` ile okunabilir
- **Düzenleme:** Doğrudan düzenlenemez, yeniden oluşturulmalı

**2. İçeriği - Ne Saklanıyor?**

FAISS index dosyası içinde şunlar saklanır:

**A) Vektör Verileri:**
- Her chunk'ın 384 boyutlu sayısal gösterimi
- Örnek vektör: `[0.234, -0.567, 0.891, 0.123, ..., -0.456]` (384 sayı)
- Her sayı `float32` formatında (4 byte)

**B) Index Başlık Bilgileri:**
- Index tipi: `IndexFlatL2`
- Vektör boyutu: `384`
- Toplam vektör sayısı: `125` (örnek)

**C) Vektör Organizasyonu:**
- Vektörler sıralı şekilde saklanır
- Her vektörün pozisyonu (index numarası) kaydedilir
- Hızlı erişim için optimize edilmiş yapı

**3. Dosya Boyutu Hesaplama:**

```
Toplam Boyut = Vektör Sayısı × Vektör Boyutu × Byte Per Sayı
             = 125 × 384 × 4 byte
             = 192.000 byte
             ≈ 188 KB
```

**4. İçerik Örneği (Görsel Temsil):**

```
FAISS INDEX DOSYASI (faiss.index)
═══════════════════════════════════════════════════════════
[BAŞLIK BİLGİLERİ]
  Index Tipi: IndexFlatL2
  Vektör Boyutu: 384
  Toplam Vektör: 125
═══════════════════════════════════════════════════════════
[VEKTÖR 0] → [0.234, -0.567, 0.891, ..., 0.123] (384 sayı)
[VEKTÖR 1] → [0.245, -0.578, 0.902, ..., 0.134] (384 sayı)
[VEKTÖR 2] → [0.256, -0.589, 0.913, ..., 0.145] (384 sayı)
...
[VEKTÖR 124] → [0.890, -0.123, 0.456, ..., 0.789] (384 sayı)
═══════════════════════════════════════════════════════════
```

**5. FAISS Index ile Metadata.json İlişkisi:**

```
FAISS Index (faiss.index)        Metadata (metadata.json)
════════════════════════         ════════════════════════
Vektör 0 (sadece sayılar)    →   Chunk 0 (metin + bilgiler)
  [0.234, -0.567, ...]       →   {
                                    "text": "Makine öğrenmesi...",
                                    "doc_id": 0,
                                    "doc_name": "makine_ogrenmesi.txt"
                                  }

Vektör 1 (sadece sayılar)    →   Chunk 1 (metin + bilgiler)
  [0.245, -0.578, ...]       →   {
                                    "text": "Gözetimli öğrenme...",
                                    "doc_id": 0,
                                    "doc_name": "makine_ogrenmesi.txt"
                                  }
```

**Nasıl Birlikte Kullanılırlar?**

1. FAISS index sadece **sayısal arama** için kullanılır (hızlı)
2. Metadata.json **metin içeriği** için kullanılır (sonuçları göstermek için)
3. Arama sonucunda:
   - FAISS → Index numarasını verir (örnek: 5)
   - Metadata → Index 5'teki metni verir

**6. Görsel Açıklama - Akış:**

```
ORJİNAL METİN                    →    VEKÖR (384 sayı)

"Makine öğrenmesi nedir?"        →    [0.23, -0.56, 0.89, ..., 0.12]
                                      ↓
                                  FAISS INDEX (faiss.index)
                                      ↓ (Arama: En yakın 5 vektör)
                            Hızlı benzerlik araması
                                      ↓
                           Index numaraları: [2, 0, 5, 8, 12]
                                      ↓
                                  Metadata.json
                                      ↓ (Index 2'deki metni al)
                           "Makine öğrenmesi (ML)..."
                                      ↓
                                  Kullanıcıya göster
```

**7. FAISS Index Türleri:**

**IndexFlatL2 (Şu an kullanılan):**
- ✅ Kesin sonuç verir
- ✅ Basit ve anlaşılır
- ❌ Büyük veri setlerinde yavaş (10.000+ vektör)

**Alternatifler:**

**IndexIVFFlat:**
- ✅ Daha hızlı (büyük veri setleri için)
- ❌ Yaklaşık sonuçlar (biraz hata payı)

**IndexHNSW:**
- ✅ Çok hızlı
- ✅ Hassas sonuçlar
- ❌ Daha fazla bellek kullanır

**8. FAISS Index Okuma (Python ile):**

```python
import faiss

# Index'i yükle
index = faiss.read_index("index/faiss.index")

# Bilgileri gör
print(f"Toplam vektör: {index.ntotal}")      # 125
print(f"Vektör boyutu: {index.d}")           # 384

# Belirli bir vektörü oku
vektör_5 = index.reconstruct(5)  # 5. vektörü getir
print(f"Vektör 5: {vektör_5[:10]}...")  # İlk 10 sayısını göster
```

**9. FAISS Index Avantajları:**

✅ **Hız:** Milisaniyeler içinde milyonlarca vektör arasında arama
✅ **Bellek:** Verimli bellek kullanımı
✅ **Ölçeklenebilirlik:** Büyük veri setlerini destekler
✅ **Doğruluk:** Matematiksel olarak kesin sonuçlar

**10. Önemli Notlar:**

- ❌ FAISS index dosyasını doğrudan düzenleyemezsiniz
- ❌ Yeni belge eklediğinizde index'i yeniden oluşturmalısınız
- ✅ Metadata.json dosyasını okuyabilirsiniz (normal metin dosyası)
- ✅ Index'i silip yeniden oluşturabilirsiniz

**FAISS Index Dosyasının İçeriği:**

FAISS index dosyası binary (ikili) formattadır, bu yüzden doğrudan okuyamazsınız. Ancak içinde şunlar saklanır:

```
FAISS Index İçeriği:
├── Index Tipi: IndexFlatL2
├── Vektör Boyutu: 384
├── Toplam Vektör Sayısı: 125 (örnek)
└── Her Vektör:
    ├── 384 adet float32 sayısı
    ├── Örnek: [0.234, -0.567, ..., 0.123]
    └── Toplam: 384 × 4 byte = 1.536 byte per vektör
```

**Örnek Hesaplama:**
- 125 chunk var
- Her chunk = 384 boyutlu vektör
- Her sayı = 4 byte (float32)
- **Toplam boyut:** 125 × 384 × 4 = **192.000 byte ≈ 188 KB**

**Neden FAISS Kullanıyoruz?**

1. **Hız:**
   - Normal arama: Tüm metinleri karşılaştır (çok yavaş)
   - FAISS: Matematiksel mesafe hesaplaması (çok hızlı)
   - 10.000 chunk'ta bile milisaniyeler içinde sonuç

2. **Ölçeklenebilirlik:**
   - Milyonlarca vektörü saklayabilir
   - Bellek kullanımını optimize eder

3. **Doğruluk:**
   - Semantik (anlamsal) benzerliği yakalar
   - "Makine öğrenmesi" = "ML" = "machine learning" (aynı anlam)

---

### 🔸 Vektör (Embedding) Nedir?

**Basit Açıklama:** Vektör, bir metnin matematiksel gösterimidir. Sayısal bir dizi ile metnin anlamını temsil ederiz.

**Gerçek Örnek:**

Metin: `"Makine öğrenmesi, veriden öğrenen algoritmalardır."`

Bu metin SentenceTransformer modeli tarafından şu şekilde vektöre dönüştürülür:

```python
# 384 boyutlu vektör (ilk 10 boyutu gösteriliyor)
[ 0.234, -0.567,  0.891, -0.123,  0.456,
 -0.789,  0.321, -0.654,  0.987, -0.234,
 ... 374 tane daha sayı ...]
```

**Bu Sayılar Ne Anlama Geliyor?**

- Her sayı, metnin belirli bir özelliğini temsil eder
- Model, eğitim sırasında hangi sayıların ne anlama geldiğini öğrenir
- Benzer anlamlı metinler, benzer sayısal değerlere sahip olur

**Karşılaştırma Örneği:**

```python
# Metin 1: "Makine öğrenmesi nedir?"
vektör1 = [0.23, -0.56, 0.89, ...]

# Metin 2: "Machine learning ne demek?" (İngilizce ama aynı anlam)
vektör2 = [0.24, -0.55, 0.88, ...]

# Mesafe hesaplama (L2):
mesafe = sqrt((0.23-0.24)² + (-0.56-(-0.55))² + ...)
# Küçük mesafe = Benzer anlam!
```

**Neden 384 Boyut?**

- Model: `all-MiniLM-L6-v2`
- 384 boyut = İyi dengeli (hız + kalite)
- Daha az boyut (128): Daha hızlı ama daha az hassas
- Daha çok boyut (768): Daha hassas ama daha yavaş

---

### 🔸 Index Dosyalarının Birlikte Çalışması

Projede 3 önemli index dosyası var:

#### 1. `faiss.index` (Binary Dosya)
**İçerik:** Sadece sayılar (vektörler)
**Okuma:** FAISS kütüphanesi ile okunur
**Boyut:** ~188 KB (125 chunk için)

#### 2. `metadata.json` (Metin Dosyası)
**İçerik:** Her chunk'ın metin içeriği ve bilgileri
**Örnek:**
```json
[
    {
        "doc_id": 0,
        "chunk_id": 0,
        "text": "Makine Öğrenmesine Giriş...",
        "doc_name": "makine_ogrenmesi.txt",
        "doc_hash": "eea0f046..."
    },
    {
        "doc_id": 0,
        "chunk_id": 1,
        "text": "regresyon Kümeleme...",
        "doc_name": "makine_ogrenmesi.txt",
        "doc_hash": "eea0f046..."
    }
]
```

#### 3. `doc_metadata.json` (Metin Dosyası)
**İçerik:** Belgelerin genel bilgileri
**Örnek:**
```json
[
    {
        "doc_id": 0,
        "name": "makine_ogrenmesi.txt",
        "hash": "eea0f046...",
        "chunk_count": 2,
        "created_at": "2025-12-03T14:31:36"
    }
]
```

**Nasıl Birlikte Çalışırlar?**

```
1. Kullanıcı sorgu girer: "Makine öğrenmesi nedir?"

2. FAISS Index'te arama:
   - Sorgu vektöre dönüştürülür
   - FAISS en yakın 5 vektörü bulur (index: 0, 15, 23, 45, 67)

3. Metadata'dan metinleri al:
   - Index 0 → metadata.json[0]["text"] = "Makine öğrenmesi (ML)..."
   - Index 15 → metadata.json[15]["text"] = "..."

4. Belge bilgilerini al:
   - metadata.json[0]["doc_id"] = 0
   - doc_metadata.json[0]["name"] = "makine_ogrenmesi.txt"

5. Sonuç göster:
   - "makine_ogrenmesi.txt" belgesinden
   - "Makine öğrenmesi (ML)..." parçası
   - Benzerlik skoru: 0.85
```

---

## 1️⃣ search_engine.py - Ana Arama Motoru

**Dosyanın Amacı:** Projenin kalbi olan bu dosya, tüm arama, indeksleme ve belge işleme fonksiyonlarını içerir. `SearchEngine` sınıfı, FAISS vektör veritabanı ve sentence-transformers kullanarak semantik arama yapar.

### 📦 İçe Aktarılan Kütüphaneler

```python
import faiss                    # Facebook'un vektör benzerliği arama kütüphanesi
import numpy as np              # Sayısal işlemler için
import json                     # JSON dosya işlemleri
import os                       # Dosya sistemi işlemleri
from sentence_transformers import SentenceTransformer  # Metin → Vektör dönüşümü
import logging                  # Log kayıtları
from transformers import pipeline  # BERT modeli için soru-cevap
from datetime import datetime    # Tarih işlemleri
import hashlib                  # MD5 hash hesaplama
import time                     # Zaman ölçümü
```

### 🔧 SearchEngine Sınıfı

#### `__init__(self, index_path, metadata_path, doc_metadata_path)`
**Satırlar:** 30-44

**Ne Yapar:**
- SearchEngine nesnesini başlatır
- Gerekli dosya yollarını ayarlar
- Sentence Transformer modelini yükler (`all-MiniLM-L6-v2`)
- Index ve belgeleri tutacak listeleri hazırlar
- Gerekli dizinleri oluşturur

**Parametreler:**
- `index_path`: FAISS index dosyasının yolu (varsayılan: "index/faiss.index")
- `metadata_path`: Metadata JSON dosyasının yolu (varsayılan: "index/metadata.json")
- `doc_metadata_path`: Belge metadata JSON dosyasının yolu (varsayılan: "index/doc_metadata.json")

**İç Değişkenler:**
- `self.model`: Sentence Transformer modeli (384 boyutlu vektörler üretir)
- `self.index`: FAISS index nesnesi (None başlangıçta)
- `self.docs`: Tüm chunk'ların metadata listesi
- `self.doc_metadata`: Belgelerin genel bilgileri
- `self.summarizer`: Özetleme modeli (henüz kullanılmıyor)
- `self.qa_pipeline`: Soru-cevap modeli (lazy loading ile yüklenir)

---

#### `_ensure_directories(self)`
**Satırlar:** 46-51

**Ne Yapar:**
- Index dosyalarının kaydedileceği dizini oluşturur
- Eğer `index/` dizini yoksa oluşturur

**Kullanım:** Otomatik olarak `__init__` içinde çağrılır.

---

#### `_save_doc_metadata(self, doc_metadata)`
**Satırlar:** 53-57

**Ne Yapar:**
- Belge metadata'sını JSON dosyasına kaydeder
- UTF-8 encoding kullanır, Türkçe karakterleri destekler
- İndentli (4 boşluk) JSON formatında kaydeder

**Parametreler:**
- `doc_metadata`: Belge bilgilerini içeren liste

**Örnek Metadata Yapısı:**
```json
{
    "doc_id": 0,
    "name": "makine_ogrenmesi.txt",
    "hash": "eea0f04690dda0320fed866cfe1335f6",
    "chunk_count": 15,
    "created_at": "2024-01-15T10:30:00"
}
```

---

#### `_load_doc_metadata(self)`
**Satırlar:** 59-68

**Ne Yapar:**
- Kaydedilmiş belge metadata'sını yükler
- Eğer dosya yoksa uyarı verir ve False döndürür

**Dönüş Değeri:**
- `True`: Başarılı yükleme
- `False`: Dosya bulunamadı

---

#### `chunk_text(self, text, chunk_size=200)`
**Satırlar:** 71-77

**Ne Yapar:**
- Uzun metinleri daha küçük parçalara (chunk) böler
- Her parça 200 kelimelik olur (varsayılan)
- Basit kelime bazlı bölme yapar (cümle sınırlarını dikkate almaz)

**Parametreler:**
- `text`: Bölünecek metin (string)
- `chunk_size`: Her chunk'taki kelime sayısı (varsayılan: 200)

**Detaylı Açıklama:**

**Adım Adım Nasıl Çalışır:**

1. **Metni Kelimelere Ayır:**
   ```python
   words = text.split()  # Boşluklardan ayırır
   ```
   - Örnek: `"Makine öğrenmesi nedir? Çok önemli bir konu."`
   - Sonuç: `["Makine", "öğrenmesi", "nedir?", "Çok", "önemli", "bir", "konu."]`

2. **200'şer Kelimelik Gruplar Oluştur:**
   ```python
   for i in range(0, len(words), chunk_size):
       chunk = " ".join(words[i:i + chunk_size])
   ```
   - İlk 200 kelime → Chunk 0
   - Sonraki 200 kelime → Chunk 1
   - Devam eder...

3. **Chunk'ları Listeye Ekle:**
   - Her chunk bir string olarak listeye eklenir

**Gerçek Örnek:**

Diyelim ki elimizde 550 kelimelik bir metin var:

```python
text = """
Makine öğrenmesi (ML), bilgisayarların açıkça programlanmadan 
veriden öğrenmesini sağlayan algoritmalar bütünüdür. 
Amaç, geçmiş verilere bakarak yeni örnekler üzerinde tahmin 
veya karar verebilen modeller geliştirmektir. Makine öğrenmesi 
genel olarak üç ana kategoriye ayrılır: Gözetimli öğrenme, 
gözetimsiz öğrenme ve pekiştirmeli öğrenme. Gözetimli öğrenmede 
algoritma, hem girdileri hem de çıktıları içeren etiketli 
verilerle eğitilir. Gözetimsiz öğrenmede ise veriler etiketli 
değildir ve algoritma verideki yapıları keşfetmeye çalışır. 
Pekiştirmeli öğrenmede model, bir ortam içinde kararlar alır 
ve her aksiyon sonrası ödül veya ceza alır. [550 kelime toplam]
"""

chunks = engine.chunk_text(text, chunk_size=200)

# Sonuç:
# chunks[0] = İlk 200 kelime (200 kelime)
# chunks[1] = Sonraki 200 kelime (200 kelime)  
# chunks[2] = Kalan 150 kelime (150 kelime)
# Toplam: 3 chunk
```

**Neden Cümle Sınırlarını Dikkate Almıyor?**

- **Basitlik:** Daha hızlı ve anlaşılır kod
- **Yeterlilik:** 200 kelime genellikle birkaç cümle içerir
- **Hız:** Cümle analizi daha yavaş olur

**Geliştirme Önerisi:**
Daha iyi chunk'lar için cümle sınırlarını dikkate alabilirsiniz:
```python
# Örnek geliştirme (şu an kullanılmıyor)
def chunk_text_smart(text, chunk_size=200):
    sentences = text.split('.')
    chunks = []
    current_chunk = []
    current_words = 0
    
    for sentence in sentences:
        words = sentence.split()
        if current_words + len(words) <= chunk_size:
            current_chunk.append(sentence)
            current_words += len(words)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_words = len(words)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks
```

**Neden Önemli:**
- FAISS ve transformer modelleri uzun metinleri işlemekte zorlanır
- Chunk'lar ayrı ayrı indekslenir, böylece daha hassas arama yapılır
- Sadece ilgili kısım döndürülür, tüm belge değil

---

#### `load_pdf(self, path)`
**Satırlar:** 80-91

**Ne Yapar:**
- PDF dosyasından metni çıkarır
- PyPDF2 kütüphanesini kullanır
- Tüm sayfaları birleştirir
- Hata durumunda uyarı verir ama devam eder

**Parametreler:**
- `path`: PDF dosyasının yolu

**Dönüş Değeri:**
- Çıkarılan metin (string)

**Hata Yönetimi:**
- Eğer bir sayfa okunamazsa, o sayfayı atlar ve devam eder
- Logger ile uyarı kaydeder

---

#### `build_index(self, documents, doc_names=None)`
**Satırlar:** 94-138

**Ne Yapar:**
- Belgeleri chunk'lara böler
- Her chunk'ı vektöre dönüştürür (embedding)
- FAISS index'i oluşturur
- Metadata'ları JSON dosyalarına kaydeder

**Parametreler:**
- `documents`: Metin listesi (her eleman bir belge)
  - Örnek: `["Makine öğrenmesi metni...", "Yapay zeka metni..."]`
- `doc_names`: Belge isimleri listesi (opsiyonel)
  - Örnek: `["makine_ogrenmesi.txt", "yapay_zeka.txt"]`
  - Verilmezse: `["Belge_1", "Belge_2", ...]` otomatik oluşturulur

**Detaylı İşlem Adımları:**

**1. ADIM: Belgeleri Chunk'lara Bölme**

```python
for doc_id, doc in enumerate(documents):
    chunks = self.chunk_text(doc)  # 200 kelimelik parçalara böler
```

**Örnek:**
- Belge 0: 550 kelime → 3 chunk (200, 200, 150 kelime)
- Belge 1: 800 kelime → 4 chunk (200, 200, 200, 200 kelime)
- **Toplam:** 7 chunk

**2. ADIM: Hash (Benzersiz Kod) Hesaplama**

```python
doc_hash = hashlib.md5(doc.encode('utf-8')).hexdigest()
```

**Ne İşe Yarar?**
- Aynı belgenin tekrar yüklenip yüklenmediğini kontrol eder
- Örnek hash: `"eea0f04690dda0320fed866cfe1335f6"` (32 karakter)

**3. ADIM: Belge Metadata Oluşturma**

Her belge için şu bilgiler kaydedilir:

```python
doc_info = {
    "doc_id": 0,                              # Belge numarası (0'dan başlar)
    "name": "makine_ogrenmesi.txt",          # Dosya adı
    "hash": "eea0f046...",                   # MD5 hash
    "chunk_count": 3,                        # Kaç parçaya bölündü?
    "created_at": "2025-12-03T14:31:36"     # Oluşturulma tarihi
}
```

**4. ADIM: Chunk Metadata Oluşturma**

Her chunk için şu bilgiler kaydedilir:

```python
metadata.append({
    "doc_id": 0,                             # Hangi belgeye ait?
    "chunk_id": 0,                           # Belgenin kaçıncı parçası?
    "text": "Makine öğrenmesi (ML)...",     # Parçanın içeriği
    "doc_name": "makine_ogrenmesi.txt",     # Belge adı
    "doc_hash": "eea0f046..."               # Belgenin hash'i
})
```

**Örnek Metadata.json:**
```json
[
    {
        "doc_id": 0,
        "chunk_id": 0,
        "text": "Makine öğrenmesi (ML)... [200 kelime]",
        "doc_name": "makine_ogrenmesi.txt",
        "doc_hash": "eea0f046..."
    },
    {
        "doc_id": 0,
        "chunk_id": 1,
        "text": "... [kalan 200 kelime]",
        "doc_name": "makine_ogrenmesi.txt",
        "doc_hash": "eea0f046..."
    }
]
```

**5. ADIM: Vektörleştirme (Embedding)**

```python
embeddings = self.model.encode(all_chunks).astype("float32")
```

**Ne Oluyor?**
- Tüm chunk'lar bir seferde vektöre dönüştürülür (batch processing)
- Her chunk → 384 boyutlu vektör
- 7 chunk varsa → 7 × 384 = 2.688 sayı

**Örnek:**
```
Chunk 0: "Makine öğrenmesi..." 
  → [0.234, -0.567, 0.891, ..., 0.123] (384 sayı)

Chunk 1: "Gözetimli öğrenme..."
  → [0.245, -0.578, 0.902, ..., 0.134] (384 sayı)

... (tüm chunk'lar)
```

**Neden Float32?**
- FAISS float32 formatı bekler
- Daha az bellek kullanır (float64 yerine)
- Yeterince hassas

**6. ADIM: FAISS Index Oluşturma**

```python
dim = embeddings.shape[1]  # 384 (vektör boyutu)
index = faiss.IndexFlatL2(dim)  # L2 mesafesi kullanır
index.add(embeddings)  # Vektörleri index'e ekler
```

**Ne Oluyor?**
- `IndexFlatL2`: Düz L2 (Öklid) mesafesi kullanan index tipi
- `index.add()`: Tüm vektörleri index'e ekler
- Index içinde vektörler özel formatta saklanır

**FAISS Index İçeriği:**
```
Index Tipi: IndexFlatL2
Vektör Boyutu: 384
Vektör Sayısı: 7

Vektör 0: [0.234, -0.567, ..., 0.123]
Vektör 1: [0.245, -0.578, ..., 0.134]
...
Vektör 6: [0.256, -0.589, ..., 0.145]
```

**7. ADIM: Dosyalara Kaydetme**

```python
faiss.write_index(index, self.index_path)  # index/faiss.index
```

**faiss.index dosyası:**
- Binary (ikili) format
- İçinde sadece sayılar (vektörler)
- Yaklaşık boyut: vektör_sayısı × 384 × 4 byte

**metadata.json dosyası:**
- JSON format (okunabilir metin)
- Her chunk'ın bilgileri
- İnsan tarafından okunabilir

**doc_metadata.json dosyası:**
- JSON format
- Belge genel bilgileri
- Belge listesini göstermek için kullanılır

**Çıktı Örneği:**
```
Index oluşturuldu → 7 chunk
Belge sayısı: 2
```

**Zamanlama:**
- Küçük veri seti (100 chunk): ~2-3 saniye
- Orta veri seti (1000 chunk): ~10-15 saniye
- Büyük veri seti (10000 chunk): ~2-3 dakika
- Model ilk yüklemede indirilir (yaklaşık 90MB)

---

#### `load_index(self)`
**Satırlar:** 141-157

**Ne Yapar:**
- Kaydedilmiş FAISS index'ini yükler
- Metadata dosyalarını okur
- Index'i belleğe alır (hızlı arama için)

**Dönüş Değeri:**
- `True`: Başarılı yükleme
- `False`: Dosyalar bulunamadı

**Kontroller:**
- Üç dosyanın da (`faiss.index`, `metadata.json`, `doc_metadata.json`) varlığını kontrol eder
- Eksik dosya varsa yükleme yapmaz

---

#### `search(self, query, k=5)`
**Satırlar:** 160-193

**Ne Yapar:**
- Kullanıcı sorgusunu vektöre dönüştürür
- FAISS ile en yakın k chunk'ı bulur
- Sonuçları benzerlik skoruna göre sıralar

**Parametreler:**
- `query`: Arama sorgusu (string)
  - Örnek: `"Makine öğrenmesi nedir?"`
- `k`: Döndürülecek sonuç sayısı (varsayılan: 5)
  - En yakın 5 chunk bulunur

**Detaylı İşlem Adımları:**

**1. ADIM: Dosya Kontrolü**

```python
if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
    return []  # Index yoksa boş liste döndür
```

**Ne Kontrol Edilir?**
- `index/faiss.index` dosyası var mı?
- `index/metadata.json` dosyası var mı?
- Eksikse arama yapılamaz

**2. ADIM: Index Yükleme**

```python
if self.index is None:
    if not self.load_index():
        return []  # Yüklenemezse boş liste
```

**Ne Oluyor?**
- Index bellekte yoksa (ilk kullanım veya yeniden başlatma)
- `load_index()` fonksiyonu çağrılır
- FAISS index ve metadata'lar belleğe alınır

**3. ADIM: Sorgu Vektörleştirme**

```python
q_vec = self.model.encode([query]).astype("float32")
```

**Ne Oluyor?**

Kullanıcı sorgusu: `"Makine öğrenmesi nedir?"`

Bu sorgu aynı SentenceTransformer modeli ile vektöre dönüştürülür:

```python
# Sorgu metni
query = "Makine öğrenmesi nedir?"

# Vektöre dönüştürme
q_vec = model.encode([query])

# Sonuç: 384 boyutlu vektör
q_vec = [[0.234, -0.567, 0.891, ..., 0.123]]  # 1 × 384 boyutlu array
```

**Neden Liste İçinde?**
- Model batch (toplu) işleme bekler
- `[query]` = 1 elemanlı liste
- Sonuç da 2D array: `[[...]]`

**4. ADIM: FAISS ile Arama**

```python
distances, indices = self.index.search(q_vec, min(k, self.index.ntotal))
```

**FAISS.search() Ne Yapar?**

FAISS, sorgu vektörünü index'teki tüm vektörlerle karşılaştırır:

```
Sorgu Vektörü:     [0.234, -0.567, ..., 0.123]

Index'teki Vektörler:
  Vektör 0:        [0.240, -0.560, ..., 0.125]  → Mesafe: 0.85
  Vektör 1:        [0.100, -0.200, ..., 0.050]  → Mesafe: 2.34
  Vektör 2:        [0.235, -0.568, ..., 0.124]  → Mesafe: 0.12  ← EN YAKIN!
  Vektör 3:        [0.500, -0.800, ..., 0.300]  → Mesafe: 4.56
  ...
```

**L2 Mesafesi (Öklid Mesafesi) Nasıl Hesaplanır?**

```
Mesafe = √[(0.234-0.235)² + (-0.567-(-0.568))² + ... + (0.123-0.124)²]
       = √[0.000001 + 0.000001 + ... + 0.000001]
       = 0.012 (yaklaşık)
```

**Dönen Değerler:**

```python
distances = [[0.12, 0.85, 1.23, 1.45, 1.67]]  # En yakın 5'in mesafeleri
indices = [[2, 0, 5, 8, 12]]                  # Hangi chunk'lar? (index numaraları)
```

**Mesafe Anlamı:**
- **0.0 - 1.0:** Çok benzer (yüksek benzerlik)
- **1.0 - 2.0:** Benzer (orta benzerlik)
- **2.0+:** Farklı (düşük benzerlik)

**5. ADIM: Metadata'dan Metinleri Alma**

```python
for idx, dist in zip(indices[0], distances[0]):
    if idx < len(self.docs):
        results.append({
            "text": self.docs[idx]["text"],        # Chunk'ın metnini al
            "score": float(dist),                  # Mesafe skoru
            "doc_name": self.docs[idx]["doc_name"], # Belge adı
            "doc_id": self.docs[idx]["doc_id"]     # Belge ID
        })
```

**Ne Oluyor?**

Index 2 → `metadata.json`'daki 2. elemana bak:
```json
{
    "doc_id": 0,
    "chunk_id": 1,
    "text": "Makine öğrenmesi (ML), bilgisayarların...",
    "doc_name": "makine_ogrenmesi.txt",
    "doc_hash": "..."
}
```

**Sonuç Formatı:**

```python
[
    {
        "text": "Makine öğrenmesi (ML), bilgisayarların açıkça programlanmadan veriden öğrenmesini sağlayan algoritmalar bütünüdür...",
        "score": 0.12,  # Mesafe (düşük = iyi)
        "doc_name": "makine_ogrenmesi.txt",
        "doc_id": 0
    },
    {
        "text": "...",
        "score": 0.85,
        "doc_name": "makine_ogrenmesi.txt",
        "doc_id": 0
    },
    ...
]
```

**Görsel Özet:**

```
1. Kullanıcı: "Makine öğrenmesi nedir?"
                    ↓
2. Vektöre dönüştür: [0.234, -0.567, ..., 0.123]
                    ↓
3. FAISS'te ara: En yakın 5 vektörü bul
                    ↓
4. Mesafeler: [0.12, 0.85, 1.23, 1.45, 1.67]
   Indexler:  [2,    0,    5,    8,    12]
                    ↓
5. Metadata'dan metinleri al:
   Index 2 → "Makine öğrenmesi (ML)..."
   Index 0 → "..."
   ...
                    ↓
6. Sonuçları döndür (skor sırasına göre)
```

**Performans:**
- Arama süresini ölçer: `time.time() - start_time`
- Genellikle 10-50 milisaniye arası
- Yazdırır: `"Arama 0.0234 saniyede tamamlandı."`

**Önemli Notlar:**

1. **Skor = Mesafe:**
   - Düşük skor = Yüksek benzerlik (iyi sonuç)
   - Yüksek skor = Düşük benzerlik (kötü sonuç)
   - Örnek: 0.12 < 0.85 (ilk sonuç daha iyi)

2. **Sonuçlar Otomatik Sıralı:**
   - FAISS en yakından en uzağa sıralar
   - En iyi sonuç ilk sırada

3. **Güvenlik Kontrolü:**
   ```python
   if idx < len(self.docs):  # Index sınırlarını kontrol et
   ```
   - Hatalı index numarasını önler

---

#### `get_document_list(self)`
**Satırlar:** 196-200

**Ne Yapar:**
- Yüklenen belgelerin listesini döndürür
- Belge metadata'sı yüklü değilse önce yükler

**Dönüş Değeri:**
- Belge bilgilerini içeren liste

**Kullanım:**
- Streamlit arayüzünde belge seçimi için kullanılır

---

#### `search_with_document_filter(self, query, doc_id=None, k=5)`
**Satırlar:** 202-270

**Ne Yapar:**
- Belirli bir belgede arama yapar (filtreli arama)
- Veya tüm belgelerde arama yapar (doc_id None ise)

**Parametreler:**
- `query`: Arama sorgusu
- `doc_id`: Belirli bir belgede arama yapmak için (None = tüm belgeler)
- `k`: Sonuç sayısı

**İşlem Mantığı:**

1. **Belge Filtreleme:**
   ```python
   filtered_indices = [i for i, doc in enumerate(self.docs) 
                       if doc.get('doc_id') == doc_id]
   ```
   - Sadece ilgili belgeye ait chunk'ları bulur

2. **Filtrelenmiş Vektörler:**
   ```python
   filtered_embeddings = np.array([self.index.reconstruct(i) 
                                    for i in filtered_indices])
   ```
   - FAISS'ten sadece ilgili vektörleri geri oluşturur

3. **Manuel Mesafe Hesaplama:**
   ```python
   dist = np.linalg.norm(q_vec[0] - emb)
   ```
   - Her vektör için L2 mesafesi hesaplanır

4. **Sıralama:**
   - Mesafelere göre sıralanır
   - En yakın k sonuç seçilir

**Kullanım:**
- Streamlit'te kullanıcı belirli bir belge seçerse bu fonksiyon çağrılır

---

#### `summarize(self, text, max_length=300, min_length=100)`
**Satırlar:** 273-346

**Ne Yapar:**
- Metni algoritmik olarak özetler
- ML modeli kullanmaz, basit kurallara göre özet çıkarır

**Parametreler:**
- `text`: Özetlenecek metin
- `max_length`: Maksimum özet uzunluğu (kullanılmıyor)
- `min_length`: Minimum özet uzunluğu (kullanılmıyor)

**Özetleme Stratejisi:**

1. **Kısa Metinler:**
   - 200 karakterden kısa ise doğrudan döndürür

2. **Paragraf Bazlı:**
   - Metni paragraflara ayırır
   - İlk 2 ve son 2 paragrafı alır
   - Her paragraftan ilk ve son cümleyi seçer

3. **Cümle Bazlı:**
   - Paragraf yoksa cümle bazlı özetler
   - İlk 3, ortadan 2, son 3 cümleyi alır

4. **Kelime Bazlı (Yedek):**
   - Hata durumunda ilk 100 ve son 100 kelimeyi alır

**Not:**
- Bu basit bir özetleme yöntemidir
- ML tabanlı özetleme için `transformers` kütüphanesindeki özetleme modelleri kullanılabilir

---

#### `answer_question(self, context, question)`
**Satırlar:** 349-391

**Ne Yapar:**
- Verilen bağlam (context) içinde soruya cevap verir
- BERT tabanlı Türkçe soru-cevap modeli kullanır

**Parametreler:**
- `context`: Soruya cevap vermek için kullanılacak metin
- `question`: Sorulan soru

**Dönüş Değeri:**
- Tuple: `(cevap, güven_skoru)`
  - `cevap`: Bulunan cevap metni
  - `güven_skoru`: 0.0-1.0 arası skor (yüksek = güvenilir)

**İşlem Adımları:**

1. **Girdi Kontrolü:**
   - Boş context veya soru kontrolü yapar

2. **Bağlam Kısaltma:**
   - BERT modelleri maksimum 512 token kabul eder
   - 1024 karakterden uzunsa kısaltır

3. **Model Yükleme (Lazy Loading):**
   ```python
   if self.qa_pipeline is None:
       self.qa_pipeline = pipeline("question-answering", 
                                   model="savasy/bert-base-turkish-squad")
   ```
   - İlk çağrıda model yüklenir (yaklaşık 500MB)
   - Türkçe model yüklenemezse İngilizce yedek model dener

4. **Soru-Cevap:**
   ```python
   result = self.qa_pipeline(question=question, context=context)
   ```
   - BERT modeli bağlam içinde sorunun cevabını bulur
   - Cevabın başlangıç ve bitiş pozisyonlarını belirler

**Kullanılan Model:**
- `savasy/bert-base-turkish-squad`: Türkçe için eğitilmiş BERT modeli
- SQuAD veri seti formatında eğitilmiştir

**Hata Yönetimi:**
- Model yüklenemezse hata mesajı döndürür
- Pipeline hatası durumunda exception yakalanır

---

## 2️⃣ build_index.py - Index Oluşturma Scripti

**Dosyanın Amacı:** Komut satırından çalıştırılarak `data/` dizinindeki belgelerden index oluşturur.

### 📝 Fonksiyonlar

#### `load_documents_from_data_dir()`
**Satırlar:** 4-45

**Ne Yapar:**
- `data/` dizinindeki tüm TXT ve PDF dosyalarını okur
- Her dosyanın içeriğini ve ismini döndürür

**İşlem Adımları:**

1. **Dizin Kontrolü:**
   ```python
   if not os.path.exists("data"):
       os.makedirs("data")
   ```
   - `data/` dizini yoksa oluşturur

2. **Dosya Okuma:**
   - `.txt` dosyaları: Direkt UTF-8 olarak okunur
   - `.pdf` dosyaları: `SearchEngine.load_pdf()` ile metin çıkarılır

3. **Hata Yönetimi:**
   - Desteklenmeyen dosya tipleri için uyarı
   - Okuma hatalarında exception yakalama

**Dönüş Değeri:**
- Tuple: `(documents, doc_names)`
  - `documents`: Metin içeriklerinin listesi
  - `doc_names`: Dosya isimlerinin listesi

---

#### `if __name__ == "__main__":`
**Satırlar:** 47-55

**Ne Yapar:**
- Script doğrudan çalıştırıldığında:
  1. `data/` dizininden belgeleri yükler
  2. SearchEngine oluşturur
  3. Index'i oluşturur

**Kullanım:**
```bash
python build_index.py
```

**Çıktı:**
```
✅ makine_ogrenmesi.txt dosyası yüklendi. (15234 karakter)
✅ yapay_zeka.txt dosyası yüklendi. (8932 karakter)
Toplam 2 döküman yüklendi.
Index oluşturuldu → 125 chunk
Belge sayısı: 2
Index başarıyla oluşturuldu!
```

---

## 3️⃣ app.py - FastAPI REST API Sunucusu

**Dosyanın Amacı:** Web API sunucusu oluşturur. Uzak uygulamaların arama yapmasını sağlar.

### 📝 Kod Açıklaması

#### FastAPI Uygulaması
**Satırlar:** 1-23

**Ne Yapar:**
- RESTful API endpoint'leri sağlar
- CORS (Cross-Origin Resource Sharing) desteği ekler
- SearchEngine'i başlatır ve index yükler

**Kurulum:**
```python
app = FastAPI()
engine = SearchEngine()
engine.load_index()
```

**CORS Ayarları:**
```python
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],      # Tüm kaynaklardan izin
                   allow_methods=["*"],      # Tüm HTTP metodları
                   allow_headers=["*"])      # Tüm header'lar
```
- Tüm domain'lerden isteklere izin verir (geliştirme amaçlı)

---

#### Query Modeli
**Satırlar:** 17-18

**Ne Yapar:**
- API isteklerinin formatını tanımlar
- Pydantic ile veri doğrulama yapar

**Yapı:**
```python
class Query(BaseModel):
    text: str  # Arama sorgusu
```

---

#### `/search` Endpoint
**Satırlar:** 20-23

**Ne Yapar:**
- POST isteği ile arama yapar
- JSON formatında sorgu alır
- Sonuçları JSON formatında döndürür

**İstek Örneği:**
```bash
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"text": "makine öğrenmesi nedir?"}'
```

**Yanıt Örneği:**
```json
{
  "results": [
    {
      "text": "Makine öğrenmesi (ML), bilgisayarların...",
      "score": 0.85,
      "doc_name": "makine_ogrenmesi.txt",
      "doc_id": 0
    },
    ...
  ]
}
```

**Kullanım:**
```bash
uvicorn app:app --reload
```
- Varsayılan port: 8000
- `--reload`: Değişikliklerde otomatik yeniden başlatma

---

## 4️⃣ streamlit_app.py - Web Arayüzü

**Dosyanın Amacı:** Kullanıcı dostu web arayüzü sağlar. Kullanıcılar tarayıcıdan belge yükleyip arama yapabilir.

### 📝 Fonksiyonlar

#### `check_memory_usage()`
**Satırlar:** 9-21

**Ne Yapar:**
- Uygulamanın bellek kullanımını kontrol eder
- %80'den fazla kullanım varsa uyarı verir ve bellek temizler

**Kullanım:**
- Özetleme işleminden önce çağrılır
- Yüksek bellek kullanımını önlemek için

---

#### `timeout(seconds)`
**Satırlar:** 23-30

**Ne Yapar:**
- İşlemler için zaman aşımı kontrolü sağlar
- Windows'ta tam timeout değil, sadece süre ölçümü yapar

**Kullanım:**
- Soru-cevap işlemlerinde 30 saniye timeout

---

### 🎨 Streamlit Arayüz Bileşenleri

#### Sayfa Yapılandırması
**Satırlar:** 32-37

```python
st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="🔍",
    layout="wide"
)
```

---

#### SearchEngine Cache
**Satırlar:** 49-54

**Ne Yapar:**
- SearchEngine nesnesini cache'ler
- Her sayfa yenilemesinde yeniden oluşturmaz
- Performans için önemli

```python
@st.cache_resource
def get_search_engine():
    engine = SearchEngine()
    return engine
```

---

#### Index Durum Kontrolü
**Satırlar:** 56-67

**Ne Yapar:**
- Index dosyalarının varlığını kontrol eder
- Kullanıcıya durum hakkında bilgi verir
- Başarılı yükleme durumunda yeşil uyarı gösterir

---

#### Sidebar Ayarlar
**Satırlar:** 69-71

- **Sonuç Sayısı Slider:** 1-20 arası sonuç sayısı seçimi

---

#### Dosya Yükleme
**Satırlar:** 73-121

**Ne Yapar:**
- Kullanıcının PDF/TXT dosyalarını yüklemesini sağlar
- Yüklenen dosyaları işler
- "Index Oluştur" butonu ile index oluşturur

**İşlem Akışı:**

1. **Dosya Yükleme:**
   ```python
   uploaded_files = st.sidebar.file_uploader(...)
   ```

2. **İçerik Çıkarma:**
   - TXT: Direkt decode edilir
   - PDF: Geçici dosyaya kaydedilir, metin çıkarılır, silinir

3. **Index Oluşturma:**
   ```python
   engine.build_index(documents, doc_names)
   ```
   - Cache temizlenir
   - Sayfa yeniden yüklenir

---

#### Belge Seçimi
**Satırlar:** 123-135

**Ne Yapar:**
- Kullanıcının belirli bir belgede arama yapmasını sağlar
- Dropdown menü ile belge seçimi

**Kullanım:**
- "Tüm Belgeler" seçilirse tüm belgelerde arama
- Belirli belge seçilirse sadece o belgede arama

---

#### Ana Arama Bölümü
**Satırlar:** 137-257

**İşlem Tipleri:**

##### 1. Semantik Arama
**Satırlar:** 147-161

**Ne Yapar:**
- Kullanıcı sorgusuna en benzer chunk'ları bulur
- Sonuçları genişletilebilir (expandable) kutularda gösterir

**Görüntüleme:**
- Her sonuç için: Skor, belge adı, chunk metni

---

##### 2. Soru Cevaplama
**Satırlar:** 163-200

**Ne Yapar:**
1. Önce semantik arama ile en ilgili chunk'ı bulur (k=1)
2. Bu chunk'ı context olarak BERT modeline verir
3. Modelden cevabı alır

**Görüntüleme:**
- Soru
- Cevap
- Güven skoru (0-1 arası)
- İşlem süresi
- Kullanılan context (genişletilebilir)

**Hata Yönetimi:**
- Timeout kontrolü
- Exception yakalama ve gösterim

---

##### 3. Özet Çıkart
**Satırlar:** 202-252

**Ne Yapar:**
- Tüm belgelerin özetini çıkarır
- Her belge ayrı ayrı özetlenir

**İşlem Akışı:**

1. **Bellek Kontrolü:**
   - Yüksek bellek kullanımı varsa işlemi durdurur

2. **Belge İşleme:**
   - Her belge için:
     - 4000 karakterden uzunsa kısaltır
     - `summarize()` fonksiyonunu çağırır
     - Özeti listeye ekler

3. **Görüntüleme:**
   - Tüm özetleri belge isimleriyle gösterir
   - Özetler arasında çizgi (divider) koyar

**Hata Yönetimi:**
- Exception yakalama
- Kullanıcıya bilgilendirme mesajı

---

#### Bilgi Kutusu
**Satırlar:** 259-265

- Kullanıcıya sistemin nasıl çalıştığını açıklar
- Sidebar'da gösterilir

---

## 5️⃣ requirements.txt - Python Bağımlılıkları

**Dosyanın Amacı:** Projenin çalışması için gerekli Python paketlerini listeler.

### 📦 Paketler ve Açıklamaları

```
faiss-cpu              # FAISS'in CPU versiyonu (vektör arama)
sentence-transformers  # Metin → Vektör dönüşümü
numpy                  # Sayısal işlemler (FAISS bağımlılığı)
PyPDF2                 # PDF dosyalarından metin çıkarma
fastapi                # REST API framework
uvicorn                 # ASGI sunucu (FastAPI için)
python-multipart       # Form verileri için (FastAPI)
streamlit              # Web arayüzü framework
transformers           # BERT ve diğer ML modelleri
torch                  # PyTorch (transformers bağımlılığı)
psutil                 # Sistem kaynaklarını ölçme (bellek)
```

**Kurulum:**
```bash
pip install -r requirements.txt
```

**Not:**
- `faiss-cpu`: GPU desteği için `faiss-gpu` kullanılabilir
- `torch`: Transformers'ın çalışması için gerekli

---

## 🔄 Sistem Akış Diyagramı

### Index Oluşturma Akışı

```
1. build_index.py çalıştırılır
   ↓
2. data/ dizinindeki dosyalar okunur
   ↓
3. Her belge chunk'lara bölünür (200 kelime)
   ↓
4. Chunk'lar vektöre dönüştürülür (SentenceTransformer)
   ↓
5. FAISS index oluşturulur ve kaydedilir
   ↓
6. Metadata JSON dosyalarına kaydedilir
```

### Arama Akışı

```
1. Kullanıcı sorgu girer
   ↓
2. Sorgu vektöre dönüştürülür
   ↓
3. FAISS ile en yakın k vektör bulunur (L2 mesafesi)
   ↓
4. Sonuçlar metadata'dan metin bilgisiyle birleştirilir
   ↓
5. Skorlara göre sıralanır ve gösterilir
```

### Soru-Cevap Akışı

```
1. Kullanıcı soru girer
   ↓
2. Semantik arama ile en ilgili chunk bulunur (k=1)
   ↓
3. Chunk context olarak BERT modeline verilir
   ↓
4. BERT modeli context içinde cevabı bulur
   ↓
5. Cevap ve güven skoru gösterilir
```

---

## 🎯 Kullanım Senaryoları

### Senaryo 1: Yeni Belge Ekleme

1. Belgeleri `data/` dizinine kopyala
2. `python build_index.py` çalıştır
3. Index yeniden oluşturulur

### Senaryo 2: Streamlit ile Arama

1. `streamlit run streamlit_app.py` çalıştır
2. Tarayıcıda açılan sayfada:
   - Dosya yükle (opsiyonel)
   - Index oluştur (yeni dosya varsa)
   - Soru/arama ifadesi gir
   - İşlem tipini seç
   - "İşlemi Gerçekleştir" tıkla

### Senaryo 3: API ile Arama

1. `uvicorn app:app --reload` çalıştır
2. API'ye POST isteği gönder:
   ```python
   import requests
   response = requests.post(
       "http://localhost:8000/search",
       json={"text": "makine öğrenmesi"}
   )
   results = response.json()["results"]
   ```

---

## 🔍 Teknik Detaylar

### Vektör Boyutu
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Boyut:** 384 boyutlu vektörler
- **Dil Desteği:** Çok dilli (Türkçe dahil)

### FAISS Index Tipi
- **Tip:** `IndexFlatL2`
- **Açıklama:** Düz L2 (Öklid) mesafesi kullanır
- **Avantaj:** Kesin sonuç verir
- **Dezavantaj:** Büyük veri setlerinde yavaş olabilir

**Alternatifler:**
- `IndexIVFFlat`: Daha hızlı, yaklaşık sonuçlar
- `IndexHNSW`: Hızlı ve hassas (büyük veri setleri için)

### Chunk Boyutu
- **Varsayılan:** 200 kelime
- **Neden:** 
  - Transformer modellerinin maksimum input uzunluğu sınırlı
  - Daha küçük chunk'lar daha hassas arama sağlar
  - Daha büyük chunk'lar daha fazla context içerir

---

## 🐛 Bilinen Sınırlamalar

1. **Özetleme:**
   - ML modeli kullanmıyor, basit algoritmik yöntem
   - Çok uzun metinlerde kalite düşebilir

2. **PDF İşleme:**
   - Sadece metin çıkarır, görselleri desteklemez
   - Karmaşık layout'larda metin kaybolabilir

3. **Bellek:**
   - Büyük veri setlerinde yüksek bellek kullanımı
   - FAISS index tamamen bellekte tutulur

4. **Türkçe Destek:**
   - Sentence Transformer çok dilli, Türkçe'yi destekler
   - QA modeli Türkçe eğitilmiş (`savasy/bert-base-turkish-squad`)

---

## 📊 Performans İpuçları

1. **Index Boyutu:**
   - 10.000 chunk'a kadar: IndexFlatL2 iyi çalışır
   - Daha büyük veri setleri için IndexIVFFlat veya IndexHNSW kullanın

2. **Chunk Boyutu:**
   - Kısa chunk'lar (100-200 kelime): Daha hassas arama
   - Uzun chunk'lar (300-500 kelime): Daha fazla context

3. **K Değeri:**
   - Semantik arama için: 5-10 yeterli
   - Soru-cevap için: 1 yeterli (en ilgili chunk)

4. **Model Seçimi:**
   - `all-MiniLM-L6-v2`: Hızlı, iyi kalite
   - `paraphrase-multilingual-MiniLM-L12-v2`: Daha iyi çok dilli destek

---

## 🚀 Geliştirme Önerileri

1. **Hybrid Search:**
   - BM25 (keyword-based) + Semantic search birleştir
   - Hem anlamsal hem kelime eşleşmesi kullan

2. **Reranking:**
   - Cross-encoder model ile sonuçları yeniden sırala
   - Daha hassas sonuçlar için

3. **Metadata Filtreleme:**
   - Tarih, kategori gibi metadata ile filtreleme
   - FAISS index'ine metadata ekle

4. **Özetleme İyileştirme:**
   - `facebook/bart-large-cnn` gibi özetleme modelleri kullan
   - Daha kaliteli özetler için

---

## 📝 Sonuç

Bu proje, modern bilgi erişim tekniklerini kullanarak:
- ✅ Semantik arama yapar
- ✅ Soru-cevap yeteneği sunar
- ✅ Belge özetleme yapar
- ✅ Kullanıcı dostu arayüz sağlar
- ✅ REST API sunar

Tüm kodlar Türkçe karakterleri destekler ve Türkçe belgeler üzerinde çalışabilir.

---

---

## 📋 ÖZET - Chunk ve FAISS Index Kavramları

### 🔸 Chunk (Parça) - Kısa Özet

**Ne?** Büyük belgenin küçük parçalara bölünmüş hali

**Neden?** 
- Model sınırlamaları (512 token)
- Hassas arama (sadece ilgili kısım)
- Hızlı işleme

**Nasıl?**
- Her 200 kelime bir chunk
- `chunk_text()` fonksiyonu ile bölünür
- Her chunk ayrı ayrı vektöre dönüştürülür

**Örnek:**
```
550 kelimelik belge → 3 chunk (200, 200, 150 kelime)
```

**İçerik:**
```json
{
    "doc_id": 0,        // Hangi belge?
    "chunk_id": 0,      // Kaçıncı parça?
    "text": "...",      // Parçanın içeriği
    "doc_name": "..."
}
```

---

### 🔸 FAISS Index - Kısa Özet

**Ne?** Vektörlerin hızlı arama için saklandığı binary dosya

**İçinde Ne Var?**
- Her chunk'ın 384 boyutlu vektör gösterimi
- Index yapısı (IndexFlatL2)
- Vektör organizasyonu

**Dosya:**
- `index/faiss.index` (binary, okunamaz)
- Boyut: ~188 KB (125 chunk için)
- Okuma: Sadece `faiss.read_index()` ile

**Nasıl Çalışır?**
```
1. Sorgu → Vektöre dönüştür
2. FAISS → En yakın vektörleri bul (mesafe hesaplama)
3. Index numaralarını al
4. Metadata'dan metinleri getir
5. Sonuçları göster
```

**Örnek:**
```
Sorgu: "Makine öğrenmesi nedir?"
  ↓
Vektör: [0.234, -0.567, ..., 0.123]
  ↓
FAISS arama → Index 2 (mesafe: 0.12)
  ↓
Metadata[2] → "Makine öğrenmesi (ML)..."
  ↓
Kullanıcıya göster
```

---

### 🔸 Üç Dosyanın İş Birliği

```
┌─────────────────────┐
│   faiss.index       │  → Sayısal arama (hızlı)
│   (Binary)          │     Index numaralarını verir
└─────────────────────┘
         ↓
┌─────────────────────┐
│   metadata.json     │  → Metin içeriği
│   (Okunabilir)      │     Index numarasına göre metin verir
└─────────────────────┘
         ↓
┌─────────────────────┐
│ doc_metadata.json   │  → Belge bilgileri
│   (Okunabilir)      │     Belge listesi ve genel bilgiler
└─────────────────────┘
```

**Akış:**
1. FAISS → En yakın 5 vektör bul (index: 2, 0, 5, 8, 12)
2. Metadata → Her index için metni getir
3. Doc Metadata → Belge adlarını getir
4. Sonuçları birleştir ve göster

---

### 🔸 Temel Terimler Sözlüğü

| Terim | Açıklama | Örnek |
|-------|----------|-------|
| **Chunk** | Belgenin küçük parçası | 200 kelimelik metin parçası |
| **Embedding** | Metnin sayısal gösterimi | [0.234, -0.567, ..., 0.123] |
| **Vektör** | Embedding'in diğer adı | 384 boyutlu sayı dizisi |
| **Index** | FAISS arama veritabanı | faiss.index dosyası |
| **L2 Mesafesi** | İki vektör arasındaki uzaklık | 0.12 (düşük = benzer) |
| **Metadata** | Ek bilgiler (metin, belge adı vb.) | JSON dosyaları |
| **Doc ID** | Belge numarası | 0, 1, 2... |
| **Chunk ID** | Parça numarası (belge içinde) | 0, 1, 2... |
| **Hash** | Belgenin benzersiz kodu | "eea0f046..." |

---

**Son Güncelleme:** 2024
**Versiyon:** 2.0 (Detaylı Türkçe Açıklamalı)

