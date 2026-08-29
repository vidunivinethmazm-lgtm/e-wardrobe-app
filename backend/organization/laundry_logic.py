# State management & wear-count rules
def update_laundry_status(item):
    # NOVELTY: Laundry Logic Integration
    # Rules based on Sri Lankan climate and fabric type
    # Cotton/Linen in heat = wash every 1-2 wears
    # Denim/Jackets = wash every 4-5 wears
    
    if item['Current_Cycle_Wears'] >= item['Max_Wears_Before_Wash']:
        return "Dirty"
    return "Clean"

def wear_item_logic(item):
    # NOVELTY: Wear Count Tracking
    item['Total_Wear_Count'] += 1
    item['Current_Cycle_Wears'] += 1
    item['Status'] = update_laundry_status(item)
    return item