"""
Database Schemas for Deepneumoscan

Each Pydantic model represents a MongoDB collection. The collection name is the
lowercase of the class name by default (handled when calling database helpers).
"""
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Literal, Dict, Any
from datetime import datetime


class User(BaseModel):
    name: str = Field(..., description="Full name")
    email: EmailStr = Field(..., description="Email address")
    password: str = Field(..., min_length=6, description="Plain text for demo only")
    language: Literal["en", "kn"] = Field("en", description="Preferred language code")
    age: Optional[int] = Field(None, ge=1, le=120)
    gender: Optional[Literal["male", "female", "other"]] = None


class SelfAssessment(BaseModel):
    user_id: str
    answers: List[Dict[str, Any]]
    predicted_condition: Literal[
        "bacterial_pneumonia",
        "viral_pneumonia",
        "fungal_pneumonia",
        "aspiration_pneumonia",
        "rsv",
        "normal",
        "other"
    ]
    confidence: float = Field(..., ge=0, le=1)


class ScanMeta(BaseModel):
    name: str
    age: int = Field(..., ge=1, le=120)
    gender: Literal["male", "female", "other"]
    medical_condition: Optional[str] = None


class ScanResult(BaseModel):
    user_id: str
    meta: ScanMeta
    filename: str
    prediction: Literal[
        "bacterial_pneumonia",
        "viral_pneumonia",
        "fungal_pneumonia",
        "aspiration_pneumonia",
        "rsv",
        "normal",
        "other"
    ]
    confidence: float = Field(..., ge=0, le=1)
    annotated_image_b64: Optional[str] = None


class CureAssessment(BaseModel):
    user_id: str
    symptoms: List[Dict[str, Any]]
    evaluation: Literal["improving", "stable", "worsening"]
    score_change: float


class HistoryItem(BaseModel):
    user_id: str
    type: Literal["self_assessment", "scan", "cure_assessment"]
    data: Dict[str, Any]
    created_at: Optional[datetime] = None
