"""File handling utilities for document loading."""

import os
from pathlib import Path


def load_documents_from_data_dir(data_dir="data"):
    """
    data dizininden belgeleri yükler
    
    Args:
        data_dir: Belgelerin bulunduğu dizin (varsayılan: "data")
    
    Returns:
        tuple: (documents, doc_names) - Metin listesi ve dosya isimleri listesi
    """
    documents = []
    doc_names = []
    
    # data dizinini kontrol et ve oluştur
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"{data_dir} dizini oluşturuldu. Belgelerinizi bu dizine ekleyin.")
        return documents, doc_names
    
    # Dosyaları oku
    files = os.listdir(data_dir)
    if not files:
        print(f"{data_dir} dizininde hiç dosya bulunamadı.")
        return documents, doc_names
    
    print(f"{len(files)} dosya bulundu: {files}")
    
    for file in files:
        path = os.path.join(data_dir, file)
        try:
            if file.endswith(".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    documents.append(content)
                    doc_names.append(file)
                    print(f"✅ {file} dosyası yüklendi. ({len(content)} karakter)")
            elif file.endswith(".pdf"):
                # PDF dosyası - işleme build_index.py'de yapılacak
                documents.append({"type": "pdf", "path": path})
                doc_names.append(file)
                print(f"✅ {file} dosyası bulundu (PDF, işleme için hazır).")
            else:
                print(f"⚠️  {file} dosyası desteklenmiyor (sadece .txt ve .pdf desteklenir)")
        except Exception as e:
            print(f"❌ {file} dosyası işlenirken hata oluştu: {e}")
    
    print(f"Toplam {len(documents)} döküman yüklendi.")
    return documents, doc_names

