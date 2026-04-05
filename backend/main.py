from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import pandas as pd
import os
import threading

import database, crud, ml_engine

@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_csv_data()          # fast — just CSV insert
    threading.Thread(target=_run_accuracy_report, daemon=True).start()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

def _load_csv_data():
    csv_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), '..', 'data', 'sri_lanka_smart_wardrobe_dataset.csv')
    )
    db = database.SessionLocal()
    try:
        if db.query(database.Item).count() == 0:
            print(f"Loading CSV from: {csv_path}")
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                db.add(database.Item(
                    item_id_str          = str(row['item_id']),
                    category             = str(row['category']),
                    color                = str(row['color']),
                    fabric               = str(row['fabric']),
                    occasion             = str(row['occasion']),
                    total_wear_count     = int(row['total_wear_count']),
                    current_cycle_wears  = int(row['current_cycle_wears']),
                    max_wears_before_wash= int(row['max_wears_before_wash']),
                    status               = str(row['status']),
                    sustainability_score = float(row['sustainability_score']),
                ))
            db.commit()
            print(f"Loaded {len(df)} items into the database.")
        else:
            print("Database already populated, skipping CSV load.")
    except Exception as e:
        print(f"ERROR loading CSV: {e}")
    finally:
        db.close()


def _run_accuracy_report():
    db = database.SessionLocal()
    try:
        all_items = db.query(database.Item).all()
        ml_engine.print_model_accuracy(all_items)
    except Exception as e:
        print(f"ERROR in accuracy report: {e}")
    finally:
        db.close()


@app.get("/items/organized")
def get_organized_items(db: Session = Depends(get_db)):
    items = crud.get_items(db)
    clusters = ml_engine.SmartWardrobeEngine.get_clusters(items)

    result = []
    for item in items:
        item_dict = {c.name: getattr(item, c.name) for c in item.__table__.columns}
        item_dict['layout_group'] = clusters.get(item.id, 0)
        result.append(item_dict)
    return result


@app.post("/items/wear/{item_id_str}")
def wear_item(item_id_str: str, db: Session = Depends(get_db)):
    item = crud.record_wear(db, item_id_str)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"message": "Wear recorded", "status": item.status}


@app.post("/items/wash/{item_id_str}")
def wash_item(item_id_str: str, db: Session = Depends(get_db)):
    item = crud.wash_item(db, item_id_str)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"message": "Item marked as washed", "status": item.status}


@app.get("/items/insights")
def get_insights(db: Session = Depends(get_db)):
    items = crud.get_items(db)
    anomalies   = ml_engine.SmartWardrobeEngine.detect_anomalies(items)
    dirty_count = db.query(database.Item).filter(database.Item.status == "Dirty").count()
    return {
        "dirty_count": dirty_count,
        "underused":   anomalies["underused"],
        "overused":    anomalies["overused"],
    }
