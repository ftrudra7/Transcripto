from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.sql import func
from app.core.database import Base

class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True, default="Untitled Session")
    transcript = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    input_type = Column(String, default="audio") # audio, sign, text
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
