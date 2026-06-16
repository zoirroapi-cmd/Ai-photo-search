from pathlib import Path

photos_dir = Path("photos")

for file in photos_dir.iterdir():
    print(file)