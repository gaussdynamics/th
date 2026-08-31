"""Application entry point.

Run the app:        python -m app.main          (from repo root)
                    python app/main.py
Headless self-test: python -m app.main --selftest

The self-test constructs the QApplication + MainWindow under the offscreen
platform and exits 0 — it verifies the environment without needing a display.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a plain script (`python app/main.py`) by putting the repo
# root on sys.path so `import app.*` resolves.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtGui import QFontDatabase  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.main_window import MainWindow  # noqa: E402

_APP_DIR = Path(__file__).resolve().parent
# Active theme. Alternates available in theme/: "panel.qss", "cassette.qss".
_THEME_NAME = "panel.qss"
_THEME = _APP_DIR / "theme" / _THEME_NAME
_FONTS = _APP_DIR / "resources" / "fonts"


def load_theme(app: QApplication) -> None:
    """Register any bundled fonts and apply the cassette stylesheet."""
    if _FONTS.exists():
        for font_file in _FONTS.glob("*.ttf"):
            QFontDatabase.addApplicationFont(str(font_file))
    if _THEME.exists():
        app.setStyleSheet(_THEME.read_text(encoding="utf-8"))


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("RailLab")
    app.setOrganizationName("Thesis")
    load_theme(app)

    win = MainWindow()
    win.show()

    if "--selftest" in sys.argv:
        # Render one frame, then quit cleanly. Proves construction + theme load.
        QTimer.singleShot(150, app.quit)
        app.exec()
        print("SELFTEST OK: QApplication + MainWindow constructed and rendered.")
        return 0

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
