import os
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wardrobe.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{_DB_PATH}"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Item(Base):
    __tablename__ = "items"

    id                  = Column(Integer, primary_key=True, index=True)
    item_id_str         = Column(String, unique=True)
    category            = Column(String)
    color               = Column(String)
    fabric              = Column(String)
    occasion            = Column(String)
    total_wear_count    = Column(Integer, default=0)
    current_cycle_wears = Column(Integer, default=0)
    max_wears_before_wash = Column(Integer, default=2)
    status              = Column(String, default="Clean")
    sustainability_score = Column(Float)

Base.metadata.create_all(bind=engine)
