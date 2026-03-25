"""
medical_rag.py
==============
RIVA Health Platform - Medical RAG Module
لقراءة ملفات PDF الطبية والبحث فيها باستخدام LangChain
"""

import logging
from pathlib import Path
from typing import List, Dict, Any

log = logging.getLogger("riva.medical_rag")

# مسار مجلد المعرفة الطبية
KNOWLEDGE_PATH = Path(__file__).parent.parent.parent / "data" / "medical_knowledge"

class MedicalRAG:
    """
    قاعدة المعرفة الطبية: تقرأ ملفات PDF وتوفر وظيفة البحث
    """
    
    def __init__(self):
        self.vectorstore = None
        self.is_loaded = False
        self._load_knowledge_base()
    
    def _load_knowledge_base(self):
        """تحميل قاعدة المعرفة من مجلد medical_knowledge"""
        try:
            # مؤقتًا: هنضيف الكود الحقيقي بعدين
            log.info(f"[MedicalRAG] Knowledge base path: {KNOWLEDGE_PATH}")
            self.is_loaded = True
        except Exception as e:
            log.error(f"[MedicalRAG] Failed to load: {e}")
            self.is_loaded = False
    
    def search(self, query: str, k: int = 5) -> List[str]:
        """البحث في قاعدة المعرفة"""
        if not self.is_loaded:
            return []
        
        # مؤقتًا: نرجع نص تجريبي
        return [f"معلومات عن: {query} - من قاعدة المعرفة الطبية"]
    
    def get_context(self, query: str) -> str:
        """الحصول على سياق للمساعدة في الرد"""
        results = self.search(query, k=3)
        if not results:
            return ""
        return "\n\n".join(results)

# نسخة واحدة للتطبيق
_rag = None

def get_medical_rag():
    global _rag
    if _rag is None:
        _rag = MedicalRAG()
    return _rag
