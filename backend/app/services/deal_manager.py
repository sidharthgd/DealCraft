import os
from pathlib import Path
from app.core.config import settings

_DEFAULT_FOLDERS = [
    "Deal Files/Income Statement",
    "Deal Files/Balance Sheet", 
    "Deal Files/Cash Flow",
    "Deal Files/LOI",
    "Deal Files/CIM",
    "Deal Files/Diligence Tracker",
    "Deal Files/Customer List",
    "Deal Files/Operations",
    "Deal Files/Site Visit Notes",
    "Deal Files/Internal Notes",
    "Deal Files/Seller Materials",
    "Deal Files/Deal Materials",
    "Deal Files/Contacts",
    "Reports/Generated PPM",
    "Reports/Memo",
]


def create_default_folder_tree(deal_id: str):
    """Create canonical directory layout on disk for a new deal."""
    root = Path(settings.UPLOAD_DIR) / deal_id
    for rel in _DEFAULT_FOLDERS:
        path = root / rel
        path.mkdir(parents=True, exist_ok=True) 