import kagglehub
import shutil 
import os

tmp_path = kagglehub.dataset_download("furkanark/premier-league-2024-2025-data") 

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
target_folder = os.path.join(project_root, "data")

if not os.path.exists(target_folder):
    os.makedirs(target_folder)
    print(f"Created folder: {target_folder}")

files = os.listdir(tmp_path)
for file_name in files: 
    source = os.path.join(tmp_path, file_name) 
    destination = os.path.join(target_folder, file_name) 

    if os.path.exists(destination):
        os.remove(destination)
        
    shutil.move(source, destination)
    print(f"Moved: {file_name}") 

print(f"\nSuccess! Files are now in: {target_folder}")