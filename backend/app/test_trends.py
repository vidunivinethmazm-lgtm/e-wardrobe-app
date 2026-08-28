import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.app.trends import get_fashion_trends

print(get_fashion_trends("women"))
print(get_fashion_trends("men"))
print(get_fashion_trends("general"))
