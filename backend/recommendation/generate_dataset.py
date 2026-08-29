from pathlib import Path
import pandas as pd
import random

# Configuration
NUM_ROWS = 5500
FILE_NAME = Path(__file__).resolve().parent / "wardrobe_data.csv"

# Data Pools based on Sri Lankan context [cite: 11, 26]
events = ['Traditional Wedding', 'Corporate Meeting', 'Beach Party', 'Pola/Market', 'Temple Visit', 'Dinner Gala', 'Gym/Workout']
fabrics = ['Silk', 'Linen', 'Cotton', 'Viscose', 'Polyester', 'Georgette']
colors = ['Navy Blue', 'Earth Tone', 'Emerald Green', 'White', 'Black', 'Maroon']
personas = ['Minimalist', 'Bold', 'Professional'] # [cite: 13]
climates = ['Indoor (AC)', 'Outdoor (Humid)'] # [cite: 15]

data = []

for i in range(NUM_ROWS):
    event = random.choice(events)
    fabric = random.choice(fabrics)
    color = random.choice(colors)
    persona = random.choice(personas)
    climate = random.choice(climates)
    
    # Logic for "Suitability Score" (The Label our models will predict) [cite: 9, 16]
    # We create a basic rule: High score if Fabric matches Climate and Event
    score = random.uniform(0.1, 0.5) # Base random score
    
    if event in ['Traditional Wedding', 'Dinner Gala'] and fabric in ['Silk', 'Georgette']:
        score += 0.4
    if climate == 'Outdoor (Humid)' and fabric in ['Linen', 'Cotton']:
        score += 0.4
    if event == 'Corporate Meeting' and persona == 'Professional':
        score += 0.3
    if climate == 'Indoor (AC)' and fabric == 'Silk':
        score += 0.2
        
    # Normalize score to be between 0 and 1
    final_score = round(min(score, 1.0), 2)
    
    # Label for Classification models (Match or No Match)
    match_status = 1 if final_score >= 0.75 else 0
    
    data.append([event, fabric, color, persona, climate, final_score, match_status])

# Create DataFrame
df = pd.DataFrame(data, columns=['Event', 'Fabric', 'Color', 'Style_Persona', 'Micro_Climate', 'Suitability_Score', 'Match_Status'])

# Save to CSV
df.to_csv(FILE_NAME, index=False)
print(f"✅ Dataset with {NUM_ROWS} rows created: {FILE_NAME}")