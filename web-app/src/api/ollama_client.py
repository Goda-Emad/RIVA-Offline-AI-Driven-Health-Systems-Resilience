"""
ollama_client.py
================
RIVA Health Platform - Ollama API Client
للتواصل مع موديلات Ollama المحلية
"""

import aiohttp
import logging
from typing import AsyncGenerator, Optional, Dict, Any

log = logging.getLogger("riva.ollama")

OLLAMA_URL = "http://ollama:11434"


async def generate_response(
    prompt: str,
    system_prompt: str = "",
    model: str = "mistral",
    stream: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_predict: int = 256
) -> AsyncGenerator[str, None] | str:
    """
    إرسال رسالة إلى Ollama والحصول على رد
    
    Args:
        prompt: رسالة المستخدم
        system_prompt: التعليمات النظامية
        model: اسم الموديل (mistral, llama2, etc.)
        stream: بث الرد كلمة بكلمة
        temperature: درجة الإبداع (0-1)
        top_p: تنوع الكلمات
        num_predict: الحد الأقصى لعدد الكلمات
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": stream,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": num_predict
        }
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload
        ) as response:
            if not stream:
                result = await response.json()
                return result.get("response", "")
            
            async for line in response.content:
                if line:
                    import json
                    try:
                        data = json.loads(line)
                        yield data.get("response", "")
                    except:
                        continue


async def generate_structured_response(
    prompt: str,
    system_prompt: str,
    model: str = "mistral"
) -> Dict[str, Any]:
    """
    إرسال رسالة والحصول على رد منظم (JSON)
    يستخدم للـ Actions (generate_qr, redirect, etc.)
    """
    full_response = ""
    async for chunk in generate_response(prompt, system_prompt, model, stream=True):
        full_response += chunk
    
    # محاولة تحويل الرد إلى JSON
    import json
    try:
        # البحث عن JSON في النص
        start = full_response.find('{')
        end = full_response.rfind('}') + 1
        if start != -1 and end > start:
            return json.loads(full_response[start:end])
    except:
        pass
    
    # رد افتراضي
    return {
        "status": "normal",
        "action": "chat",
        "message": full_response
    }


async def get_available_models() -> list:
    """الحصول على قائمة الموديلات المتاحة"""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{OLLAMA_URL}/api/tags") as response:
            if response.status == 200:
                data = await response.json()
                return [m["name"] for m in data.get("models", [])]
    return []


async def pull_model(model_name: str = "mistral") -> bool:
    """تحميل موديل من Ollama"""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{OLLAMA_URL}/api/pull",
            json={"name": model_name}
        ) as response:
            return response.status == 200


async def check_ollama_health() -> bool:
    """فحص إذا كان Ollama شغال"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{OLLAMA_URL}/api/tags") as response:
                return response.status == 200
    except:
        return False
