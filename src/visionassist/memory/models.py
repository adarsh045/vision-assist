from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, Text, func
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_path = Column(String, nullable=False)
    timestamp = Column(DateTime, server_default=func.now())

    objects = relationship("DetectedObject", back_populates="detection", cascade="all, delete")


class DetectedObject(Base):
    __tablename__ = "detection_objects"

    id = Column(Integer, primary_key=True, autoincrement=True)
    detection_id = Column(Integer, ForeignKey("detections.id"), nullable=False)

    object_name = Column(String, nullable=False)
    confidence = Column(Float, nullable=True)
    bbox = Column(Text, nullable=True)

    timestamp = Column(DateTime, server_default=func.now())

    detection = relationship("Detection", back_populates="objects")
