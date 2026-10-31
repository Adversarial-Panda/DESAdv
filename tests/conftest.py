from pathlib import Path
import sys
r=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(r/'src'),str(r/'experiments')]
