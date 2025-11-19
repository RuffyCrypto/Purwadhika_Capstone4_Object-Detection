# src/calorie_map.py
# Simple mapping from class index to calorie and label.
# Customize this mapping based on your data.yaml or dataset.
CALORIE_MAP = {
    0: {'label': 'Ayam Goreng', 'cal': 260, 'unit': 'per100g'},
    1: {'label': 'Capcay', 'cal': 67, 'unit': 'per100g'},
    2: {'label': 'Nasi', 'cal': 129, 'unit': 'per100g'},
    3: {'label': 'Sayur Bayam', 'cal': 36, 'unit': 'per100g'},
    4: {'label': 'Sayur Kangkung', 'cal': 98, 'unit': 'per100g'},
    5: {'label': 'Sayur Sop', 'cal': 22, 'unit': 'per100g'},
    6: {'label': 'Tahu', 'cal': 80, 'unit': 'per100g'},
    7: {'label': 'Telur Dadar', 'cal': 93, 'unit': 'per100g'},
    8: {'label': 'Telur Mata Sapi', 'cal': 110, 'unit': 'per_item'},
    9: {'label': 'Telur Rebus', 'cal': 78, 'unit': 'per_item'},
    10: {'label': 'Tempe', 'cal': 225, 'unit': 'per100g'},
    11: {'label': 'Tumis Buncis', 'cal': 65, 'unit': 'per100g'}
}

def get_calorie_info(cls_idx):
    return CALORIE_MAP.get(cls_idx, {'label': f'class_{cls_idx}', 'cal': 0, 'unit': 'per100g'})
