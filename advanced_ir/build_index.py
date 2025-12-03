from search_engine import SearchEngine
import os

def load_documents_from_data_dir():
    """data dizininden belgeleri yükler"""
    documents = []
    doc_names = []
    
    # data dizinini kontrol et ve oluştur
    if not os.path.exists("data"):
        os.makedirs("data")
        print("data dizini oluşturuldu. Belgelerinizi bu dizine ekleyin.")
        return documents, doc_names
    
    # Dosyaları oku
    files = os.listdir("data")
    if not files:
        print("data dizininde hiç dosya bulunamadı.")
        return documents, doc_names
    
    print(f"{len(files)} dosya bulundu: {files}")
    
    engine = SearchEngine()
    
    for file in files:
        path = f"data/{file}"
        try:
            if file.endswith(".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    documents.append(content)
                    doc_names.append(file)
                    print(f"✅ {file} dosyası yüklendi. ({len(content)} karakter)")
            elif file.endswith(".pdf"):
                content = engine.load_pdf(path)
                documents.append(content)
                doc_names.append(file)
                print(f"✅ {file} dosyası yüklendi. ({len(content)} karakter)")
            else:
                print(f"⚠️  {file} dosyası desteklenmiyor (sadece .txt ve .pdf desteklenir)")
        except Exception as e:
            print(f"❌ {file} dosyası işlenirken hata oluştu: {e}")
    
    print(f"Toplam {len(documents)} döküman yüklendi.")
    return documents, doc_names

if __name__ == "__main__":
    documents, doc_names = load_documents_from_data_dir()
    
    if documents:
        engine = SearchEngine()
        engine.build_index(documents, doc_names)
        print("Index başarıyla oluşturuldu!")
    else:
        print("İşlenecek belge bulunamadı. Önce data dizinine belgelerinizi ekleyin.")
