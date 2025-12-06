import faiss
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
import logging
from transformers import pipeline
from datetime import datetime
import hashlib
import time

# Windows uyumluluğu için timeout
import sys
if sys.platform != "win32":
    from timeout_decorator import timeout, TimeoutError
else:
    # Windows için mock timeout
    def timeout(seconds):
        def decorator(func):
            return func
        return decorator
    class TimeoutError(Exception):
        pass

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchEngine:
    def __init__(self, index_path="index/faiss.index", metadata_path="index/metadata.json", doc_metadata_path="index/doc_metadata.json"):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.doc_metadata_path = doc_metadata_path
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.index = None
        self.docs = []
        self.doc_metadata = []
        
        # Dizinleri oluştur
        self._ensure_directories()
        
        # Özetleme ve QA modellerini yükle
        self.summarizer = None
        self.qa_pipeline = None

    def _ensure_directories(self):
        """Gerekli dizinleri oluşturur"""
        index_dir = os.path.dirname(self.index_path)
        if index_dir and not os.path.exists(index_dir):
            os.makedirs(index_dir)
            logger.info(f"Dizin oluşturuldu: {index_dir}")
    
    def _save_doc_metadata(self, doc_metadata):
        """Belge meta verisini kaydeder"""
        with open(self.doc_metadata_path, "w", encoding="utf-8") as f:
            json.dump(doc_metadata, f, ensure_ascii=False, indent=4)
        logger.info("Belge meta verisi kaydedildi.")
    
    def _load_doc_metadata(self):
        """Belge meta verisini yükler"""
        if os.path.exists(self.doc_metadata_path):
            with open(self.doc_metadata_path, "r", encoding="utf-8") as f:
                self.doc_metadata = json.load(f)
            logger.info("Belge meta verisi yüklendi.")
            return True
        else:
            logger.warning("Belge meta verisi bulunamadı.")
            return False

    # ================= CHUNKING ================= #
    def chunk_text(self, text, chunk_size=200):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    # ================= LOAD TEXT ================= #
    def load_pdf(self, path):
        from PyPDF2 import PdfReader
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            try:
                text += page.extract_text()
            except Exception as e:
                logger.warning(f"Sayfa metni çıkarılırken hata oluştu: {str(e)}")
                # Boş satır ekleyerek devam et
                text += "\n"
        return text

    # ================= INDEX OLUŞTURMA ================= #
    def build_index(self, documents, doc_names=None):
        all_chunks = []
        metadata = []
        doc_metadata = []
        
        # Belge isimleri sağlanmamışsa varsayılan isimler oluştur
        if doc_names is None:
            doc_names = [f"Belge_{i+1}" for i in range(len(documents))]

        for doc_id, doc in enumerate(documents):
            # Eğer doc bir dict ise (PDF için), path'ten oku
            if isinstance(doc, dict) and doc.get("type") == "pdf":
                doc = self.load_pdf(doc["path"])
            
            chunks = self.chunk_text(doc)
            doc_hash = hashlib.md5(doc.encode('utf-8')).hexdigest()
            doc_info = {
                "doc_id": doc_id,
                "name": doc_names[doc_id],
                "hash": doc_hash,
                "chunk_count": len(chunks),
                "created_at": datetime.now().isoformat()
            }
            doc_metadata.append(doc_info)
            
            for chunk_id, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                metadata.append({
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "doc_name": doc_names[doc_id],
                    "doc_hash": doc_hash
                })

        embeddings = self.model.encode(all_chunks).astype("float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        # Kaydet
        faiss.write_index(index, self.index_path)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        self._save_doc_metadata(doc_metadata)

        print(f"Index oluşturuldu → {len(all_chunks)} chunk")
        print(f"Belge sayısı: {len(documents)}")

    # ================= İNDEX YÜKLE ================= #
    def load_index(self):
        if (os.path.exists(self.index_path) and 
            os.path.exists(self.metadata_path) and
            os.path.exists(self.doc_metadata_path)):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.docs = json.load(f)
            self._load_doc_metadata()
            print("📥 FAISS index yüklendi.")
            return True
        else:
            print("⚠️ Index dosyaları bulunamadı.")
            # Önceki verileri temizle
            self.index = None
            self.docs = []
            self.doc_metadata = []
            return False

    # ================= ARAMA ================= #
    def search(self, query, k=5):
        start_time = time.time()
        
        # Her seferinde index dosyalarının varlığını kontrol et
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            print("⚠️ Index dosyaları bulunamadı.")
            return []
        
        # Index yüklü değilse veya dosyalar değişmişse yeniden yükle
        if self.index is None:
            if not self.load_index():
                return []

        # Güvenlik kontrolü
        if self.index is None or not self.docs:
            print("⚠️ Index veya belgeler yüklenemedi.")
            return []

        q_vec = self.model.encode([query]).astype("float32")
        distances, indices = self.index.search(q_vec, min(k, self.index.ntotal))

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.docs):  # Bounds check
                results.append({
                    "text": self.docs[idx]["text"],
                    "score": float(dist),
                    "doc_name": self.docs[idx].get("doc_name", "Bilinmiyor"),
                    "doc_id": self.docs[idx].get("doc_id", -1)
                })
        
        elapsed_time = time.time() - start_time
        print(f"Arama {elapsed_time:.4f} saniyede tamamlandı.")
        return results
    
    # ================= BELGE İŞLEMLERİ ================= #
    def get_document_list(self):
        """Yüklenen belgelerin listesini döndürür"""
        if not self.doc_metadata:
            self._load_doc_metadata()
        return self.doc_metadata
    
    def search_with_document_filter(self, query, doc_id=None, k=5):
        """Belirli bir belgede arama yapar"""
        # Index yüklü değilse veya dosyalar değişmişse yeniden yükle
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            print("⚠️ Index dosyaları bulunamadı.")
            return []
        
        if self.index is None:
            if not self.load_index():
                return []
        
        # Güvenlik kontrolü
        if self.index is None or not self.docs:
            print("⚠️ Index veya belgeler yüklenemedi.")
            return []

        q_vec = self.model.encode([query]).astype("float32")
        
        # Eğer belirli bir belgede arama yapılacaksa
        if doc_id is not None:
            # Sadece ilgili belgeye ait chunk'ları filtrele
            filtered_indices = [i for i, doc in enumerate(self.docs) if doc.get('doc_id') == doc_id]
            if not filtered_indices:
                return []
                
            # Filtrelenmiş indekslerde arama yap
            try:
                filtered_embeddings = np.array([self.index.reconstruct(i) for i in filtered_indices])
                distances = []
                indices = []
                
                for i, emb in enumerate(filtered_embeddings):
                    dist = np.linalg.norm(q_vec[0] - emb)
                    distances.append(dist)
                    indices.append(filtered_indices[i])
                
                # En yakın k sonuç
                sorted_pairs = sorted(zip(distances, indices))[:min(k, len(distances))]
                
                results = []
                for dist, idx in sorted_pairs:
                    if idx < len(self.docs):  # Bounds check
                        results.append({
                            "text": self.docs[idx]["text"],
                            "score": float(dist),
                            "doc_name": self.docs[idx].get("doc_name", "Bilinmiyor"),
                            "doc_id": self.docs[idx].get("doc_id", -1)
                        })
                return results
            except Exception as e:
                print(f"Filtreli arama hatası: {e}")
                return []
        else:
            # Tüm belgelerde arama
            try:
                distances, indices = self.index.search(q_vec, min(k, self.index.ntotal))
                results = []
                for idx, dist in zip(indices[0], distances[0]):
                    if idx < len(self.docs):  # Bounds check
                        results.append({
                            "text": self.docs[idx]["text"],
                            "score": float(dist),
                            "doc_name": self.docs[idx].get("doc_name", "Bilinmiyor"),
                            "doc_id": self.docs[idx].get("doc_id", -1)
                        })
                return results
            except Exception as e:
                print(f"Genel arama hatası: {e}")
                return []
    
    # ================= ÖZETLEME ================= #
    def summarize(self, text, max_length=300, min_length=100):
        """Metni özetler - Daha ayrıntılı sürüm"""
        if not text or len(text.strip()) == 0:
            return "Özetlenecek metin bulunamadı."
        
        # Metin çok kısasa doğrudan döndür
        if len(text) < 200:
            return text
        
        # Daha ayrıntılı özetleme
        try:
            # Temel temizlik
            text = text.replace('\n', ' ').replace('\r', ' ').strip()
            
            # Paragraflara ayır
            paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 20]
            
            if len(paragraphs) >= 2:
                # Birden fazla paragraf varsa ilk ve son paragrafları al
                if len(paragraphs) <= 4:
                    # Az paragraf varsa hepsini kullan
                    summary_parts = paragraphs
                else:
                    # 5+ paragraf varsa ilk 2 ve son 2 paragrafı al
                    summary_parts = paragraphs[:2] + ["..."] + paragraphs[-2:]
                
                # Her paragraftan önemli cümleleri seç
                final_summary = []
                for part in summary_parts:
                    if part == "...":
                        final_summary.append(part)
                    else:
                        sentences = [s.strip() for s in part.split('.') if len(s.strip()) > 10]
                        if len(sentences) >= 2:
                            # İlk ve son cümleyi al
                            selected = [sentences[0]]
                            if len(sentences) > 2:
                                selected.append("...")
                            selected.append(sentences[-1])
                            final_summary.append(". ".join(selected) + ".")
                        else:
                            final_summary.append(part)
                
                return "\n\n".join(final_summary)
            
            # Paragraf yoksa cümle bazlı özetle
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 15]
            
            if len(sentences) <= 6:
                # Az cümle varsa hepsini döndür
                return ". ".join(sentences) + "."
            else:
                # Cümle sayısı fazlaysa daha fazlasını al
                mid_point = len(sentences) // 2
                selected_sentences = (
                    sentences[:3] +  # İlk 3 cümle
                    ["..."] +
                    sentences[mid_point-1:mid_point+1] +  # Ortadaki 2 cümle
                    ["..."] +
                    sentences[-3:]  # Son 3 cümle
                )
                return ". ".join(selected_sentences) + "."
                
        except Exception as e:
            logger.warning(f"Özetleme hatası: {str(e)}")
            # Yedek yöntem: Kelime bazlı daha uzun özet
            words = text.split()
            if len(words) <= 200:
                return text
            else:
                # İlk 100 ve son 100 kelimeyi al
                start_words = words[:100]
                end_words = words[-100:] if len(words) > 200 else words[len(words)//2:]
                return " ".join(start_words) + "..." + " ".join(end_words)
    
    # ================= SORU CEVAP ================= #
    def answer_question(self, context, question):
        """Verilen bağlamda soruya cevap verir"""
        if not context or not question:
            return "Bağlam veya soru eksik.", 0.0
        
        if len(context.strip()) == 0 or len(question.strip()) == 0:
            return "Boş bağlam veya soru.", 0.0
        
        # Bağlam çok uzunsa kısalt
        original_length = len(context)
        if len(context) > 1024:
            context = context[:1024]
            logger.info(f"Bağlam kısaltıldı: {original_length} -> 1024 karakter")
        
        if self.qa_pipeline is None:
            # Türkçe destekli QA modeli
            try:
                logger.info("QA modeli yükleniyor...")
                self.qa_pipeline = pipeline("question-answering", model="savasy/bert-base-turkish-squad")
                logger.info("QA modeli yüklendi")
            except Exception as e:
                logger.warning(f"Türkçe QA modeli yüklenemedi: {str(e)}")
                try:
                    # Yedek model
                    logger.info("Yedek QA modeli yükleniyor...")
                    self.qa_pipeline = pipeline("question-answering")
                    logger.info("Yedek QA modeli yüklendi")
                except Exception as e2:
                    logger.error(f"Yedek QA modeli yüklenemedi: {str(e2)}")
                    return "QA modelleri yüklenemedi.", 0.0
        
        try:
            logger.info(f"Soru-cevap işlemi başlatılıyor. Bağlam: {len(context)} karakter, Soru: {len(question)} karakter")
            result = self.qa_pipeline(question=question, context=context)
            if result and 'answer' in result and 'score' in result:
                logger.info("Soru-cevap işlemi tamamlandı")
                return result['answer'], result['score']
            else:
                logger.info("Cevap bulunamadı")
                return "Cevap bulunamadı.", 0.0
        except Exception as e:
            logger.error(f"Soru cevaplama hatası: {str(e)}")
            return f"Cevap bulunamadı. Hata: {str(e)}", 0.0

