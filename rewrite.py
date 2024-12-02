import os
import pandas as pd

# Paths
image_folder = r"D:\activity_surveillance\dataset_path\Normal"
csv_file_path = r"D:\activity_surveillance\dataset_path\normal_keypoints.csv"

# Load the CSV file without assuming headers
df = pd.read_csv(csv_file_path, header=None)

# Rename columns: first column as 'image_name', rest as 'coord_x'
df.columns = ['image_name'] + [f'coord_{i}' for i in range(1, len(df.columns))]

# Print updated column names for verification
print("Renamed CSV columns:", df.columns)

# List of images in the folder
existing_images = set(os.listdir(image_folder))

# Filter rows where the image file exists
df_cleaned = df[df['image_name'].apply(lambda x: x in existing_images)]

# Save the cleaned DataFrame back to the CSV file
df_cleaned.to_csv(csv_file_path, index=False, header=False)  # Save without header for consistency

print(f"Cleaned CSV file saved to {csv_file_path}. Removed {len(df) - len(df_cleaned)} rows.")
