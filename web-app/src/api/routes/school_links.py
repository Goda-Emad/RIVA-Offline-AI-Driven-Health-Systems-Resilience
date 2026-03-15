from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import List, Optional
from functools import lru_cache
import json, os, sqlite3, uuid
from datetime import datetime

router = APIRouter()

# ================================================================
# CONFIG
# ================================================================
class Settings(BaseSettings):
    db_path: str = "data/databases/school.db"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

# ================================================================
# MODELS
# ================================================================
class SchoolLink(BaseModel):
    school_id:  str
    school_name: str
    student_id: str
    class_name: str

class SchoolLinkBulk(BaseModel):
    school_id:   str
    school_name: str
    class_name:  str
    student_ids: List[str]

# ================================================================
# DB HELPER
# ================================================================
def get_db(settings: Settings = Depends(get_settings)):
    if not os.path.exists(settings.db_path):
        raise HTTPException(status_code=404, detail="Database not found")
    return sqlite3.connect(settings.db_path)

# ================================================================
# ROUTES
# ================================================================
@router.post("/link")
def link_student_to_school(data: SchoolLink):
    try:
        conn = sqlite3.connect("data/databases/school.db")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS school_links (
                id          TEXT PRIMARY KEY,
                school_id   TEXT,
                school_name TEXT,
                student_id  TEXT,
                class_name  TEXT,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute(
            "INSERT OR IGNORE INTO school_links VALUES (?,?,?,?,?,?)",
            (str(uuid.uuid4()), data.school_id, data.school_name,
             data.student_id, data.class_name, datetime.now())
        )
        conn.commit()
        conn.close()
        return {"success": True, "message": "Student linked to school"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/link-bulk")
def link_bulk(data: SchoolLinkBulk):
    try:
        conn = sqlite3.connect("data/databases/school.db")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS school_links (
                id TEXT PRIMARY KEY, school_id TEXT,
                school_name TEXT, student_id TEXT,
                class_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        for student_id in data.student_ids:
            conn.execute(
                "INSERT OR IGNORE INTO school_links VALUES (?,?,?,?,?,?)",
                (str(uuid.uuid4()), data.school_id, data.school_name,
                 student_id, data.class_name, datetime.now())
            )
        conn.commit()
        conn.close()
        return {
            "success": True,
            "linked":  len(data.student_ids),
            "school":  data.school_name,
            "class":   data.class_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/school/{school_id}")
def get_school_students(school_id: str):
    try:
        conn = sqlite3.connect("data/databases/school.db")
        rows = conn.execute(
            "SELECT * FROM school_links WHERE school_id = ?",
            (school_id,)
        ).fetchall()
        conn.close()
        return {
            "school_id": school_id,
            "students":  [{"id": r[0], "student_id": r[3],
                           "class": r[4]} for r in rows],
            "total":     len(rows)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/class/{school_id}/{class_name}")
def get_class_students(school_id: str, class_name: str):
    try:
        conn = sqlite3.connect("data/databases/school.db")
        rows = conn.execute(
            "SELECT * FROM school_links WHERE school_id=? AND class_name=?",
            (school_id, class_name)
        ).fetchall()
        conn.close()
        return {
            "school_id":  school_id,
            "class_name": class_name,
            "students":   [{"student_id": r[3]} for r in rows],
            "total":      len(rows)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/unlink/{school_id}/{student_id}")
def unlink_student(school_id: str, student_id: str):
    try:
        conn = sqlite3.connect("data/databases/school.db")
        conn.execute(
            "DELETE FROM school_links WHERE school_id=? AND student_id=?",
            (school_id, student_id)
        )
        conn.commit()
        conn.close()
        return {"success": True, "message": "Student unlinked"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
