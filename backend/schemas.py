# Pydantic models (data validation)
from pydantic import BaseModel

class ItemBase(BaseModel):
    item_id_str: str
    category: str
    status: str
    total_wear_count: int

class ItemUpdate(BaseModel):
    status: str
    current_cycle_wears: int

class InsightResponse(BaseModel):
    message: str