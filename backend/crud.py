from sqlalchemy.orm import Session
import database

def get_items(db: Session):
    return db.query(database.Item).all()

def get_item_by_str_id(db: Session, item_id_str: str):
    return db.query(database.Item).filter(database.Item.item_id_str == item_id_str).first()

def record_wear(db: Session, item_id_str: str):
    item = get_item_by_str_id(db, item_id_str)
    if not item:
        return None
    item.total_wear_count    += 1
    item.current_cycle_wears += 1
    if item.current_cycle_wears >= item.max_wears_before_wash:
        item.status = "Dirty"
    db.commit()
    db.refresh(item)
    return item

def wash_item(db: Session, item_id_str: str):
    item = get_item_by_str_id(db, item_id_str)
    if not item:
        return None
    item.current_cycle_wears = 0
    item.status = "Clean"
    db.commit()
    db.refresh(item)
    return item
