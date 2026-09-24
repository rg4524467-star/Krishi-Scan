"""
SQLAlchemy ORM models - the server-side (cloud) schema.

Minimum required models are all present:
  Farmer, Diagnosis, SoilRecord, SyncQueue (+ the 'synced' flag on Diagnosis).

JSON blobs (soil_data_json, weather_data_json) store the exact unmodified
inputs the farmer confirmed, so downstream audits always reflect what was
actually used - never a silently-mutated copy.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, ForeignKey, Integer,
                        String, Text)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Farmer(Base):
    __tablename__ = "farmers"

    id = Column(Integer, primary_key=True)
    name = Column(String(120), nullable=True)
    phone = Column(String(20), nullable=True, index=True)
    preferred_language = Column(String(8), default="hi")
    location = Column(String(200), nullable=True)   # village/town/district text

    diagnoses = relationship("Diagnosis", back_populates="farmer")
    soil_records = relationship("SoilRecord", back_populates="farmer")


class Diagnosis(Base):
    __tablename__ = "diagnoses"

    id = Column(Integer, primary_key=True)
    farmer_id = Column(Integer, ForeignKey("farmers.id"), nullable=True, index=True)
    # Local filename/uuid of the captured leaf photo.
    image_ref = Column(String(200), nullable=True)
    crop = Column(String(80), nullable=True)
    predicted_disease = Column(String(120), nullable=False)
    confidence = Column(Float, nullable=False)
    # JSON of the confirmed inputs (may be None when skipped).
    soil_data_json = Column(Text, nullable=True)
    weather_data_json = Column(Text, nullable=True)
    gatekeeper_failed = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    synced = Column(Boolean, default=True, index=True)

    farmer = relationship("Farmer", back_populates="diagnoses")
    sync_entries = relationship("SyncQueue", back_populates="diagnosis")


class SoilRecord(Base):
    __tablename__ = "soil_records"

    id = Column(Integer, primary_key=True)
    farmer_id = Column(Integer, ForeignKey("farmers.id"), nullable=True, index=True)
    n = Column(Float, nullable=True)
    p = Column(Float, nullable=True)
    k = Column(Float, nullable=True)
    ph = Column(Float, nullable=True)
    source = Column(String(12), default="manual")   # manual | ocr
    timestamp = Column(DateTime, default=datetime.utcnow)

    farmer = relationship("Farmer", back_populates="soil_records")


class SyncQueue(Base):
    __tablename__ = "sync_queue"

    id = Column(Integer, primary_key=True)
    diagnosis_id = Column(Integer, ForeignKey("diagnoses.id"), nullable=False, index=True)
    created_offline_at = Column(DateTime, default=datetime.utcnow)
    synced_at = Column(DateTime, nullable=True)

    diagnosis = relationship("Diagnosis", back_populates="sync_entries")
