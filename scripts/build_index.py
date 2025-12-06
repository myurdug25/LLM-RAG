#!/usr/bin/env python3
"""
Index oluşturma scripti.
data dizinindeki belgelerden FAISS index'i oluşturur.
"""

import sys
from pathlib import Path

# Proje root'unu Python path'ine ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.engine import SearchEngine
from src.utils import load_documents_from_data_dir
import os


def main():
    """Ana fonksiyon: Belgeleri yükler ve index oluşturur."""
    print("=" * 60)
    print("📚 Index Oluşturma Scripti")
    print("=" * 60)
    print()
    
    # Belgeleri yükle
    documents, doc_names = load_documents_from_data_dir()
    
    if documents:
        print(f"\n📦 {len(documents)} belge yüklendi. Index oluşturuluyor...")
        print("-" * 60)
        
        engine = SearchEngine()
        
        # PDF dosyalarını işle
        processed_documents = []
        for i, doc in enumerate(documents):
            if isinstance(doc, dict) and doc.get("type") == "pdf":
                # PDF dosyasını oku
                text = engine.load_pdf(doc["path"])
                processed_documents.append(text)
                print(f"✅ PDF işlendi: {doc_names[i]}")
            else:
                processed_documents.append(doc)
        
        # Index oluştur
        engine.build_index(processed_documents, doc_names)
        print("-" * 60)
        print("✅ Index başarıyla oluşturuldu!")
        print("=" * 60)
    else:
        print("❌ İşlenecek belge bulunamadı.")
        print("💡 Önce data/ dizinine belgelerinizi ekleyin.")
        print("=" * 60)


if __name__ == "__main__":
    main()

