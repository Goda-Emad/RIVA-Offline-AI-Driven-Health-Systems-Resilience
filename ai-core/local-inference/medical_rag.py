"""
medical_rag.py
==============
RIVA Health Platform - Medical RAG Module
لقراءة ملفات PDF الطبية والبحث فيها باستخدام LangChain و ChromaDB
"""

import logging
import os
from pathlib import Path
from typing import List

# استدعاء مكتبات LangChain والذكاء الاصطناعي
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

log = logging.getLogger("riva.medical_rag")

# مسار مجلد المعرفة الطبية (ملفات الـ PDF)
KNOWLEDGE_PATH = Path(__file__).parent.parent.parent / "data" / "medical_knowledge"
# مسار حفظ قاعدة البيانات (عشان منقراش الملفات كل مرة من الصفر)
CHROMA_DB_PATH = Path(__file__).parent.parent.parent / "data" / "chroma_db"

class MedicalRAG:
    """
    قاعدة المعرفة الطبية: تقرأ ملفات PDF وتوفر وظيفة البحث
    """
    
    def __init__(self):
        self.vectorstore = None
        self.is_loaded = False
        # موديل Embeddings بيدعم عربي وإنجليزي بكفاءة
        self.embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self._load_knowledge_base()
    
    def _load_knowledge_base(self):
        """تحميل قاعدة المعرفة من مجلد medical_knowledge أو من قاعدة البيانات المحفوظة"""
        try:
            log.info(f"[MedicalRAG] جاري تهيئة الذكاء الاصطناعي... مسار المعرفة: {KNOWLEDGE_PATH}")
            embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)

            # لو قاعدة البيانات موجودة ومبنية قبل كده، حملها فوراً (أسرع بكتير)
            if os.path.exists(CHROMA_DB_PATH) and os.listdir(CHROMA_DB_PATH):
                log.info("[MedicalRAG] تم العثور على قاعدة بيانات سابقة. جاري التحميل...")
                self.vectorstore = Chroma(persist_directory=str(CHROMA_DB_PATH), embedding_function=embeddings)
                self.is_loaded = True
                log.info("[MedicalRAG] ✅ تم تحميل قاعدة البيانات بنجاح!")
                return

            # لو مفيش قاعدة بيانات، هنبنيها من الـ 9 ملفات PDF بتوعنا
            log.info("[MedicalRAG] جاري قراءة الملفات الطبية وبناء قاعدة البيانات لأول مرة (قد يستغرق بعض الوقت)...")
            loader = PyPDFDirectoryLoader(str(KNOWLEDGE_PATH))
            documents = loader.load()
            
            if not documents:
                log.warning("[MedicalRAG] ⚠️ لم يتم العثور على أي ملفات PDF في المجلد!")
                self.is_loaded = False
                return

            log.info(f"[MedicalRAG] تم قراءة {len(documents)} صفحة. جاري التقسيم...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            
            log.info(f"[MedicalRAG] تم تقسيم النصوص إلى {len(chunks)} قطعة. جاري الحفظ في ChromaDB...")
            self.vectorstore = Chroma.from_documents(
                documents=chunks, 
                embedding=embeddings, 
                persist_directory=str(CHROMA_DB_PATH)
            )
            self.vectorstore.persist()
            
            self.is_loaded = True
            log.info("[MedicalRAG] 🎉 عااااش! تم بناء قاعدة البيانات بنجاح!")
            
        except Exception as e:
            log.error(f"[MedicalRAG] فشل التحميل: {e}")
            self.is_loaded = False
    
    def search(self, query: str, k: int = 5) -> List[str]:
        """البحث في قاعدة المعرفة"""
        if not self.is_loaded or not self.vectorstore:
            return []
        
        try:
            # البحث عن أقرب نصوص للسؤال
            docs = self.vectorstore.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            log.error(f"[MedicalRAG] خطأ أثناء البحث: {e}")
            return []
    
    def get_context(self, query: str) -> str:
        """الحصول على سياق للمساعدة في الرد"""
        results = self.search(query, k=3) # بنجيب أفضل 3 قطع نصوص
        if not results:
            return ""
        return "\n\n---\n\n".join(results)

# نسخة واحدة للتطبيق (Singleton)
_rag = None

def get_medical_rag():
    global _rag
    if _rag is None:
        _rag = MedicalRAG()
    return _rag
