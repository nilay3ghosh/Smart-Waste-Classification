import os
import json

# 1. Create the folder 'mapping' if it doesn't exist
os.makedirs("mapping", exist_ok=True)

# 2. Define your classes
class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# 3. Create a dictionary mapping indices to class names
class_indices = {str(i): name for i, name in enumerate(class_names)}

# 4. Save it as JSON
with open("mapping/classes.json", "w") as f:
    json.dump(class_indices, f, indent=4)

print("classes.json created successfully!")
