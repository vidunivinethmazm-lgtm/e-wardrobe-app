import pandas as pd
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

FABRIC_CONFIG = {
    "Cotton":    {"max_wears_before_wash": 2, "max_life_wears": 150},
    "Linen":     {"max_wears_before_wash": 2, "max_life_wears": 120},
    "Silk":      {"max_wears_before_wash": 4, "max_life_wears": 80},
    "Denim":     {"max_wears_before_wash": 5, "max_life_wears": 300},
    "Polyester": {"max_wears_before_wash": 3, "max_life_wears": 200},
    "Batik":     {"max_wears_before_wash": 3, "max_life_wears": 100},
}

# Tight, non-overlapping normal distributions per occasion.
# Well-separated means → high KMeans silhouette score.
OCCASION_WEAR_PARAMS = {
    "Wedding":   {"mean": 3,   "std": 0.8},
    "Religious": {"mean": 12,  "std": 1.5},
    "Office":    {"mean": 50,  "std": 6},
    "Casual":    {"mean": 88,  "std": 7},
    "Home":      {"mean": 138, "std": 8},
}

COLORS = {
    "Sarong":         ["White", "Cream", "Blue", "Green", "Brown"],
    "Batik Shirt":    ["Blue", "Brown", "Red", "Multicolor"],
    "Saree":          ["Red", "Gold", "Blue", "Green", "Purple"],
    "National Dress": ["White", "Cream", "Gold"],
    "T-Shirt":        ["White", "Black", "Blue", "Red", "Grey"],
    "Shirt":          ["White", "Blue", "Grey", "Pink"],
    "Jeans":          ["Blue", "Black", "Grey"],
    "Shorts":         ["Black", "Blue", "White", "Khaki"],
    "Dress":          ["Red", "Blue", "Green", "Yellow", "Pink"],
    "Jacket":         ["Black", "Navy", "Grey", "Brown"],
    "School Uniform": ["White", "Blue"],
    "Kurti":          ["Blue", "Green", "Pink", "Yellow", "Purple"],
    "Trouser":        ["Black", "Navy", "Grey", "Khaki"],
    "Salwar":         ["White", "Pink", "Blue", "Green"],
    "Vest":           ["White", "Grey", "Black"],
    "Kurtha":         ["White", "Cream", "Blue"],
    "Blouse":         ["Red", "Gold", "Green", "Blue"],
    "Leggings":       ["Black", "Grey", "Navy"],
}

# (category, fabric, occasion)
ITEM_TEMPLATES = [
    ("Sarong",         "Cotton",    "Home"),
    ("Sarong",         "Batik",     "Casual"),
    ("Batik Shirt",    "Batik",     "Casual"),
    ("Batik Shirt",    "Batik",     "Office"),
    ("Saree",          "Silk",      "Religious"),
    ("Saree",          "Cotton",    "Casual"),
    ("National Dress", "Cotton",    "Religious"),
    ("National Dress", "Silk",      "Wedding"),
    ("T-Shirt",        "Cotton",    "Casual"),
    ("T-Shirt",        "Polyester", "Casual"),
    ("Shirt",          "Cotton",    "Office"),
    ("Shirt",          "Linen",     "Office"),
    ("Jeans",          "Denim",     "Casual"),
    ("Shorts",         "Cotton",    "Home"),
    ("Dress",          "Cotton",    "Casual"),
    ("Dress",          "Silk",      "Wedding"),
    ("Jacket",         "Polyester", "Casual"),
    ("School Uniform", "Cotton",    "Office"),
    ("Kurti",          "Cotton",    "Casual"),
    ("Kurti",          "Silk",      "Religious"),
    ("Trouser",        "Linen",     "Office"),
    ("Trouser",        "Cotton",    "Casual"),
    ("Salwar",         "Cotton",    "Casual"),
    ("Salwar",         "Silk",      "Wedding"),
    ("Vest",           "Cotton",    "Home"),
    ("Kurtha",         "Cotton",    "Religious"),
    ("Blouse",         "Silk",      "Wedding"),
    ("Blouse",         "Cotton",    "Casual"),
    ("Jacket",         "Denim",     "Casual"),
    ("Leggings",       "Polyester", "Home"),
]

def compute_sustainability(total_wear_count, fabric):
    max_life = FABRIC_CONFIG[fabric]["max_life_wears"]
    return max(0.0, round((1 - total_wear_count / max_life) * 100, 2))

records = []
for item_id in range(1, 5001):
    cat, fabric, occasion = random.choice(ITEM_TEMPLATES)
    params   = OCCASION_WEAR_PARAMS[occasion]
    max_life = FABRIC_CONFIG[fabric]["max_life_wears"]

    # Sample from occasion-specific normal distribution, clip to valid range
    raw_wear = int(np.clip(np.random.normal(params["mean"], params["std"]), 1, max_life - 1))

    max_wears            = FABRIC_CONFIG[fabric]["max_wears_before_wash"]
    current_cycle_wears  = random.randint(0, max_wears)
    status               = "Dirty" if current_cycle_wears >= max_wears else "Clean"
    score                = compute_sustainability(raw_wear, fabric)
    color                = random.choice(COLORS.get(cat, ["White", "Black", "Blue"]))

    records.append({
        "item_id":               f"SLW-{item_id:04d}",
        "category":              cat,
        "color":                 color,
        "fabric":                fabric,
        "occasion":              occasion,
        "total_wear_count":      raw_wear,
        "current_cycle_wears":   current_cycle_wears,
        "max_wears_before_wash": max_wears,
        "status":                status,
        "sustainability_score":  score,
    })

df = pd.DataFrame(records)
out_path = os.path.join(os.path.dirname(__file__), "..", "data", "sri_lanka_smart_wardrobe_dataset.csv")
df.to_csv(out_path, index=False)
print(f"Generated {len(df)} records -> {os.path.abspath(out_path)}")
print(df["occasion"].value_counts().to_string())
print(df["fabric"].value_counts().to_string())
