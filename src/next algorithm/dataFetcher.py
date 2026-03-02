import kagglehub
import shutil
import os

# =========================
# Resolve project root
# =========================
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# =========================
# Download dataset
# =========================
tmp_path = kagglehub.dataset_download(
    "ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset"
)

# =========================
# Ensure data directory exists
# =========================
os.makedirs(DATA_DIR, exist_ok=True)

# =========================
# Move files
# =========================
for file_name in os.listdir(tmp_path):
    source = os.path.join(tmp_path, file_name)
    destination = os.path.join(DATA_DIR, file_name)

    if os.path.exists(destination):
        os.remove(destination)

    shutil.move(source, destination)
    print(f"Moved: {file_name}")

print(f"\nSuccess! Files are now in: {DATA_DIR}")