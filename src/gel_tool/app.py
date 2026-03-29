"""
PySide6 desktop GUI for gel image processing.

Load a raw gel photo, preview it, switch between three output modes,
and export the result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QImage, QPixmap, QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QComboBox,
    QPushButton,
    QSlider,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QScrollArea,
    QSizePolicy,
    QMessageBox,
    QGroupBox,
)

from gel_tool.processor import process_gel, GelResult

MODE_GREEN = "Green Channel"
MODE_BOOSTED = "Green + Band Boost"
MODE_RESIDUAL = "Residual (Detrended)"

MODES = [MODE_GREEN, MODE_BOOSTED, MODE_RESIDUAL]


def _ndarray_to_qpixmap(img: np.ndarray) -> QPixmap:
    """Convert a numpy image (grayscale or BGR) to QPixmap."""
    if img.ndim == 2:
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, w * ch, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


class ImageViewer(QScrollArea):
    """Scrollable, zoomable image display."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
        )
        self.setWidget(self._label)
        self.setWidgetResizable(True)
        self._pixmap: QPixmap | None = None

    def set_pixmap(self, pm: QPixmap):
        self._pixmap = pm
        self._label.setPixmap(
            pm.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pixmap:
            self._label.setPixmap(
                self._pixmap.scaled(
                    self.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gel Image Tool")
        self.resize(1200, 800)

        self._result: GelResult | None = None
        self._current_mode: str = MODE_GREEN
        self._gain: float = 4.0

        self._build_ui()
        self._build_menu()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # --- Left: controls ---
        ctrl_panel = QWidget()
        ctrl_panel.setFixedWidth(280)
        ctrl_layout = QVBoxLayout(ctrl_panel)
        ctrl_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        load_btn = QPushButton("Load Image...")
        load_btn.setMinimumHeight(40)
        load_btn.clicked.connect(self._on_load)
        ctrl_layout.addWidget(load_btn)

        # Preview of raw image
        raw_group = QGroupBox("Raw Preview")
        raw_layout = QVBoxLayout(raw_group)
        self._raw_thumb = QLabel("No image loaded")
        self._raw_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._raw_thumb.setMinimumHeight(160)
        self._raw_thumb.setStyleSheet("background: #1a1a1a; color: #888;")
        raw_layout.addWidget(self._raw_thumb)
        ctrl_layout.addWidget(raw_group)

        # Output mode selector
        mode_group = QGroupBox("Output Mode")
        mode_layout = QVBoxLayout(mode_group)
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(MODES)
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self._mode_combo)
        ctrl_layout.addWidget(mode_group)

        # Gain slider (for band boost)
        gain_group = QGroupBox("Band Boost Gain")
        gain_layout = QVBoxLayout(gain_group)
        self._gain_label = QLabel(f"Gain: {self._gain:.1f}")
        gain_layout.addWidget(self._gain_label)
        self._gain_slider = QSlider(Qt.Orientation.Horizontal)
        self._gain_slider.setRange(10, 100)
        self._gain_slider.setValue(int(self._gain * 10))
        self._gain_slider.valueChanged.connect(self._on_gain_changed)
        gain_layout.addWidget(self._gain_slider)
        self._gain_group = gain_group
        ctrl_layout.addWidget(gain_group)

        # Export
        export_btn = QPushButton("Export Current View...")
        export_btn.setMinimumHeight(36)
        export_btn.clicked.connect(self._on_export)
        ctrl_layout.addWidget(export_btn)

        ctrl_layout.addStretch()

        # --- Right: main preview ---
        self._viewer = ImageViewer()
        self._viewer.setStyleSheet("background: #111;")

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(ctrl_panel)
        splitter.addWidget(self._viewer)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._update_gain_visibility()

    def _build_menu(self):
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")

        open_act = QAction("&Open Image...", self)
        open_act.setShortcut(QKeySequence.StandardKey.Open)
        open_act.triggered.connect(self._on_load)
        file_menu.addAction(open_act)

        export_act = QAction("&Export...", self)
        export_act.setShortcut(QKeySequence("Ctrl+E"))
        export_act.triggered.connect(self._on_export)
        file_menu.addAction(export_act)

        file_menu.addSeparator()
        quit_act = QAction("&Quit", self)
        quit_act.setShortcut(QKeySequence.StandardKey.Quit)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

    def _update_gain_visibility(self):
        visible = self._current_mode == MODE_BOOSTED
        self._gain_group.setVisible(visible)

    def _current_output(self) -> np.ndarray | None:
        if self._result is None:
            return None
        if self._current_mode == MODE_GREEN:
            return self._result.green
        elif self._current_mode == MODE_BOOSTED:
            return self._result.boosted
        elif self._current_mode == MODE_RESIDUAL:
            return self._result.residual_u8
        return None

    def _refresh_main_view(self):
        img = self._current_output()
        if img is None:
            return
        self._viewer.set_pixmap(_ndarray_to_qpixmap(img))

    def _reprocess_boost(self):
        """Re-run only the band-boost with the current gain."""
        if self._result is None:
            return
        from gel_tool.processor import _band_boost

        self._result.boosted = _band_boost(self._result.green, gain=self._gain)
        self._result.gain = self._gain
        if self._current_mode == MODE_BOOSTED:
            self._refresh_main_view()

    @Slot()
    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Gel Image",
            "",
            "Images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp);;All (*)",
        )
        if not path:
            return

        self._status.showMessage(f"Processing {Path(path).name}...")
        QApplication.processEvents()

        try:
            self._result = process_gel(path, gain=self._gain)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self._status.clearMessage()
            return

        # Raw thumbnail
        thumb = _ndarray_to_qpixmap(self._result.crop_bgr)
        self._raw_thumb.setPixmap(
            thumb.scaled(
                260,
                160,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

        self._refresh_main_view()
        h, w = self._result.green.shape
        self._status.showMessage(
            f"{Path(path).name}  |  Gel crop: {w} x {h}  |  Mode: {self._current_mode}"
        )

    @Slot(str)
    def _on_mode_changed(self, mode: str):
        self._current_mode = mode
        self._update_gain_visibility()
        self._refresh_main_view()
        if self._result:
            self._status.showMessage(
                f"Mode: {mode}  |  Gel crop: {self._result.green.shape[1]} x {self._result.green.shape[0]}"
            )

    @Slot(int)
    def _on_gain_changed(self, val: int):
        self._gain = val / 10.0
        self._gain_label.setText(f"Gain: {self._gain:.1f}")
        self._reprocess_boost()

    @Slot()
    def _on_export(self):
        img = self._current_output()
        if img is None:
            QMessageBox.information(self, "Nothing to export", "Load an image first.")
            return

        mode_suffix = {
            MODE_GREEN: "green",
            MODE_BOOSTED: f"boosted_g{self._gain:.1f}",
            MODE_RESIDUAL: "residual",
        }
        suggested = f"gel_{mode_suffix.get(self._current_mode, 'output')}.png"

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Image",
            suggested,
            "PNG (*.png);;TIFF (*.tif *.tiff);;JPEG (*.jpg);;All (*)",
        )
        if not path:
            return

        cv2.imwrite(path, img)
        self._status.showMessage(f"Exported: {path}")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Gel Image Tool")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
