import json
import os
from utils.dataset_loader import load_datasets

_, _, class_names = load_datasets()

os.makedirs("models", exist_ok=True)

with open("models/class_names.json", "w") as f:
    json.dump(class_names, f, indent=4)

print("Class names saved.")