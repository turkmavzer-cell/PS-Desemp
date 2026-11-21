"""MeChi Studio giris noktasi (single-instance IPC ve ozel pencere ikonu)."""
from __future__ import annotations

import os
import sys

_MECHI_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_MECHI_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PySide6.QtGui import QIcon
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication

from mechi.config import ASSETS_DIR, ensure_dirs
from mechi.core.module_registry import ModuleRegistry
from mechi.modules import register_builtin_modules
from mechi.theme import build_stylesheet
from mechi.ui.main_window import MeChiMainWindow

IPC_SERVER_NAME = "MeChiStudioSingleInstance_v1"


def _notify_running_instance() -> bool:
    sock = QLocalSocket()
    sock.connectToServer(IPC_SERVER_NAME)
    if sock.waitForConnected(400):
        sock.write(b"RAISE")
        sock.waitForBytesWritten(400)
        sock.disconnectFromServer()
        return True
    return False


def main() -> int:
    ensure_dirs()
    app = QApplication(sys.argv)
    app.setApplicationName("MeChi")
    app.setQuitOnLastWindowClosed(True)

    ico_path = os.path.join(ASSETS_DIR, "mechi.ico")
    if os.path.exists(ico_path):
        app_icon = QIcon(ico_path)
        app.setWindowIcon(app_icon)
    else:
        app_icon = QIcon()

    if _notify_running_instance():
        return 0

    app.setStyleSheet(build_stylesheet())

    registry = ModuleRegistry()
    register_builtin_modules(registry)

    win = MeChiMainWindow(registry)
    if not app_icon.isNull():
        win.setWindowIcon(app_icon)

    QLocalServer.removeServer(IPC_SERVER_NAME)
    server = QLocalServer(win)
    if server.listen(IPC_SERVER_NAME):
        def _on_new_conn() -> None:
            conn = server.nextPendingConnection()
            if conn:
                conn.waitForReadyRead(300)
                conn.close()
            win.raise_from_ipc()

        server.newConnection.connect(_on_new_conn)

    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
