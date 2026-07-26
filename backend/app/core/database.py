import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# SQLite database for portable session history
DATABASE_URL = "sqlite:///./transcripto.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
