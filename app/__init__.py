"""Mini Speech-to-Text Service Package."""

import os
import sys
from pathlib import Path

# Ensure Windows C-extension DLLs in .libs packages (av.libs, ctranslate2.libs) can be loaded
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    site_packages_dirs = [p for p in sys.path if "site-packages" in p]
    for sp in site_packages_dirs:
        sp_path = Path(sp)
        if sp_path.exists():
            for lib_dir in sp_path.glob("*.libs"):
                try:
                    os.add_dll_directory(str(lib_dir))
                except OSError:
                    pass

__version__ = "1.0.0"
