import os

# ディレクトリリスト
dirs = [
    "./code",
    "./data",
    "./images/test",
    "./images/train",
    "./models",
    "./outputs"
]

for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"Created: {d}")

