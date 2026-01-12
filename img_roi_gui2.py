#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMG ROI Averager
- Drag & drop *.img files
- Configure raw loader (width/height/dtype/endianness/header offset)
- Configure ROI1/ROI2 (X, Y, Width, Height)
- Computes ROI means asynchronously and shows per-file results
Tested with Python 3.9+. Requires: PyQt5, numpy, pandas
"""
import os
import sys
import struct
import traceback
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pandas as pd

DTYPE_MAP = {
    "uint8": np.uint8,
    "int16": np.int16,
    "uint16": np.uint16,
    "float32": np.float32,
    "float64": np.float64,
}

ENDIAN_MAP = {
    "native": "=",
    "little": "<",
    "big": ">",
}

@dataclass
class LoaderConfig:
    width: int
    height: int
    dtype: str   # keys of DTYPE_MAP
    endian: str  # keys of ENDIAN_MAP
    header_offset: int  # bytes to skip

@dataclass
class ROIConfig:
    x: int
    y: int
    w: int
    h: int

class WorkerSignals(QtCore.QObject):
    # (filepath, roi1_mean, roi2_mean, meta dict)
    result = QtCore.pyqtSignal(str, float, float, dict)
    error = QtCore.pyqtSignal(str, str)            # (filepath, error message)

class ComputeTask(QtCore.QRunnable):
    def __init__(self, filepath: str, loader: LoaderConfig, roi1: ROIConfig, roi2: ROIConfig):
        super().__init__()
        self.filepath = filepath
        self.loader = loader
        self.roi1 = roi1
        self.roi2 = roi2
        self.signals = WorkerSignals()
        self.setAutoDelete(True)

    def run(self):
        try:
            mean1, mean2, meta = self.compute_roi_means(self.filepath, self.loader, self.roi1, self.roi2)
            self.signals.result.emit(self.filepath, float(mean1), float(mean2), meta)
        except Exception as e:
            tb = traceback.format_exc()
            self.signals.error.emit(self.filepath, f"{e}\n{tb}")

    @staticmethod
    def compute_roi_means(filepath: str, loader: LoaderConfig, roi1: ROIConfig, roi2: ROIConfig) -> Tuple[float, float, dict]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)
        dtype = DTYPE_MAP[loader.dtype]
        endian_char = ENDIAN_MAP[loader.endian]
        dt = np.dtype(dtype).newbyteorder(endian_char)

        # read file in
        with open(filepath, "rb") as f:
            f.seek(loader.header_offset, os.SEEK_SET)
            expected_pixels = loader.width * loader.height
            arr = np.fromfile(f, dtype=dt, count=expected_pixels)

        if arr.size != expected_pixels:
            raise ValueError(
                f"File does not contain enough data for the specified shape. "
                f"Expected {expected_pixels} elements, got {arr.size}."
            )

        arr = arr.reshape((loader.height, loader.width))  # row-major (Y, X)

        def roi_mean(roi: ROIConfig, name: str) -> float:
            x0, y0 = roi.x, roi.y
            x1, y1 = x0 + roi.w, y0 + roi.h
            if x0 < 0 or y0 < 0 or x1 > loader.width or y1 > loader.height:
                raise ValueError(
                    f"{name} out of bounds: image {loader.width}x{loader.height}, "
                    f"{name}=({x0},{y0},{roi.w},{roi.h})"
                )
            roi_arr = arr[y0:y1, x0:x1]
            return float(roi_arr.mean())

        mean1 = roi_mean(roi1, "ROI1")
        mean2 = roi_mean(roi2, "ROI2")

        meta = {
            "width": loader.width,
            "height": loader.height,
            "dtype": loader.dtype,
            "endian": loader.endian,
            "header_offset": loader.header_offset,
            "roi1": (roi1.x, roi1.y, roi1.w, roi1.h),
            "roi2": (roi2.x, roi2.y, roi2.w, roi2.h),
        }
        return mean1, mean2, meta

class DropTable(QtWidgets.QTableWidget):
    fileDropped = QtCore.pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(0, 5, parent)
        self.setHorizontalHeaderLabels(["File", "ROI1 Mean", "ROI2 Mean", "Config", "Status"])
        self.horizontalHeader().setStretchLastSection(True)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setAcceptDrops(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DropOnly)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent):
        urls = event.mimeData().urls()
        paths = []
        for url in urls:
            p = url.toLocalFile()
            if p and p.lower().endswith((".img", ".IMG")):
                paths.append(p)
        if paths:
            self.fileDropped.emit(paths)
        event.acceptProposedAction()

    def add_or_update_row(self, filepath: str, mean1: Optional[float], mean2: Optional[float], config_str: str, status: str):
        # find row by filepath
        row = -1
        for r in range(self.rowCount()):
            if self.item(r, 0) and self.item(r, 0).text() == filepath:
                row = r
                break
        if row == -1:
            row = self.rowCount()
            self.insertRow(row)
            self.setItem(row, 0, QtWidgets.QTableWidgetItem(filepath))

        self.setItem(row, 1, QtWidgets.QTableWidgetItem("" if mean1 is None else f"{mean1:.6f}"))
        self.setItem(row, 2, QtWidgets.QTableWidgetItem("" if mean2 is None else f"{mean2:.6f}"))
        self.setItem(row, 3, QtWidgets.QTableWidgetItem(config_str))
        self.setItem(row, 4, QtWidgets.QTableWidgetItem(status))

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMG ROI Averager")
        self.resize(1100, 650)
        self.thread_pool = QtCore.QThreadPool.globalInstance()

        # Loader config widgets
        self.widthSpin = QtWidgets.QSpinBox()
        self.widthSpin.setRange(1, 50000)
        self.widthSpin.setValue(4096)

        self.heightSpin = QtWidgets.QSpinBox()
        self.heightSpin.setRange(1, 50000)
        self.heightSpin.setValue(2880)

        self.dtypeCombo = QtWidgets.QComboBox()
        self.dtypeCombo.addItems(list(DTYPE_MAP.keys()))
        self.dtypeCombo.setCurrentText("uint16")

        self.endianCombo = QtWidgets.QComboBox()
        self.endianCombo.addItems(list(ENDIAN_MAP.keys()))
        self.endianCombo.setCurrentText("little")

        self.headerSpin = QtWidgets.QSpinBox()
        self.headerSpin.setRange(0, 1_000_000_000)
        self.headerSpin.setValue(0)

        loaderForm = QtWidgets.QFormLayout()
        loaderForm.addRow("Width (X):", self.widthSpin)
        loaderForm.addRow("Height (Y):", self.heightSpin)
        loaderForm.addRow("DType:", self.dtypeCombo)
        loaderForm.addRow("Endianness:", self.endianCombo)
        loaderForm.addRow("Header offset (bytes):", self.headerSpin)

        # ROI1 config widgets
        self.x1Spin = QtWidgets.QSpinBox()
        self.x1Spin.setRange(0, 1_000_000)
        self.x1Spin.setValue(1420)

        self.y1Spin = QtWidgets.QSpinBox()
        self.y1Spin.setRange(0, 1_000_000)
        self.y1Spin.setValue(1536)

        self.w1Spin = QtWidgets.QSpinBox()
        self.w1Spin.setRange(1, 1_000_000)
        self.w1Spin.setValue(1000)

        self.h1Spin = QtWidgets.QSpinBox()
        self.h1Spin.setRange(1, 1_000_000)
        self.h1Spin.setValue(1000)

        roi1Form = QtWidgets.QFormLayout()
        roi1Form.addRow("ROI1 X:", self.x1Spin)
        roi1Form.addRow("ROI1 Y:", self.y1Spin)
        roi1Form.addRow("ROI1 Width:", self.w1Spin)
        roi1Form.addRow("ROI1 Height:", self.h1Spin)

        roi1Box = QtWidgets.QGroupBox("ROI 1")
        roi1Box.setLayout(roi1Form)

        # ROI2 config widgets
        self.x2Spin = QtWidgets.QSpinBox()
        self.x2Spin.setRange(0, 1_000_000)
        self.x2Spin.setValue(1420)

        self.y2Spin = QtWidgets.QSpinBox()
        self.y2Spin.setRange(0, 1_000_000)
        self.y2Spin.setValue(1536)

        self.w2Spin = QtWidgets.QSpinBox()
        self.w2Spin.setRange(1, 1_000_000)
        self.w2Spin.setValue(1000)

        self.h2Spin = QtWidgets.QSpinBox()
        self.h2Spin.setRange(1, 1_000_000)
        self.h2Spin.setValue(1000)

        roi2Form = QtWidgets.QFormLayout()
        roi2Form.addRow("ROI2 X:", self.x2Spin)
        roi2Form.addRow("ROI2 Y:", self.y2Spin)
        roi2Form.addRow("ROI2 Width:", self.w2Spin)
        roi2Form.addRow("ROI2 Height:", self.h2Spin)

        roi2Box = QtWidgets.QGroupBox("ROI 2")
        roi2Box.setLayout(roi2Form)

        roiLayout = QtWidgets.QVBoxLayout()
        roiLayout.addWidget(roi1Box)
        roiLayout.addWidget(roi2Box)

        # Buttons
        self.exportBtn = QtWidgets.QPushButton("Export CSV")
        self.exportBtn.clicked.connect(self.export_csv)

        # Results table with drag & drop
        self.table = DropTable()
        self.table.fileDropped.connect(self.handle_files)

        # Layout
        leftBox = QtWidgets.QGroupBox("Loader Configuration")
        leftBox.setLayout(loaderForm)
        rightBox = QtWidgets.QGroupBox("ROI Configuration")
        rightBox.setLayout(roiLayout)

        configLayout = QtWidgets.QHBoxLayout()
        configLayout.addWidget(leftBox, 1)
        configLayout.addWidget(rightBox, 1)

        mainLayout = QtWidgets.QVBoxLayout(self)
        mainLayout.addLayout(configLayout)
        mainLayout.addWidget(self.table, 1)
        mainLayout.addWidget(self.exportBtn, 0, alignment=QtCore.Qt.AlignRight)

        # Help hint
        self.table.add_or_update_row(
            "Drop *.img files here",
            None,
            None,
            "Adjust settings above then drop files",
            "Ready",
        )

    def current_loader(self) -> LoaderConfig:
        return LoaderConfig(
            width=self.widthSpin.value(),
            height=self.heightSpin.value(),
            dtype=self.dtypeCombo.currentText(),
            endian=self.endianCombo.currentText(),
            header_offset=self.headerSpin.value(),
        )

    def current_rois(self) -> Tuple[ROIConfig, ROIConfig]:
        roi1 = ROIConfig(
            x=self.x1Spin.value(),
            y=self.y1Spin.value(),
            w=self.w1Spin.value(),
            h=self.h1Spin.value(),
        )
        roi2 = ROIConfig(
            x=self.x2Spin.value(),
            y=self.y2Spin.value(),
            w=self.w2Spin.value(),
            h=self.h2Spin.value(),
        )
        return roi1, roi2

    @QtCore.pyqtSlot(list)
    def handle_files(self, paths: List[str]):
        loader = self.current_loader()
        roi1, roi2 = self.current_rois()
        cfg_str = (
            f"{loader.width}x{loader.height} {loader.dtype} {loader.endian} "
            f"off={loader.header_offset}; "
            f"ROI1=({roi1.x},{roi1.y},{roi1.w},{roi1.h}); "
            f"ROI2=({roi2.x},{roi2.y},{roi2.w},{roi2.h})"
        )
        for p in paths:
            self.table.add_or_update_row(p, None, None, cfg_str, "Processing…")
            task = ComputeTask(p, loader, roi1, roi2)
            task.signals.result.connect(self.on_result)
            task.signals.error.connect(self.on_error)
            self.thread_pool.start(task)

    @QtCore.pyqtSlot(str, float, float, dict)
    def on_result(self, filepath: str, mean1: float, mean2: float, meta: dict):
        cfg_str = (
            f"{meta['width']}x{meta['height']} {meta['dtype']} {meta['endian']} "
            f"off={meta['header_offset']}; "
            f"ROI1={meta['roi1']}; ROI2={meta['roi2']}"
        )
        self.table.add_or_update_row(filepath, mean1, mean2, cfg_str, "Done")

    @QtCore.pyqtSlot(str, str)
    def on_error(self, filepath: str, message: str):
        self.table.add_or_update_row(filepath, None, None, "", f"ERROR: {message.splitlines()[0]}")

    def export_csv(self):
        # Build DataFrame from the table
        rows = []
        for r in range(self.table.rowCount()):
            file_item = self.table.item(r, 0)
            if not file_item:
                continue
            fp = file_item.text()
            mean1_item = self.table.item(r, 1)
            mean2_item = self.table.item(r, 2)
            cfg_item = self.table.item(r, 3)
            status_item = self.table.item(r, 4)
            rows.append({
                "file": fp,
                "roi1_mean": None if mean1_item is None or not mean1_item.text() else float(mean1_item.text()),
                "roi2_mean": None if mean2_item is None or not mean2_item.text() else float(mean2_item.text()),
                "config": None if cfg_item is None else cfg_item.text(),
                "status": None if status_item is None else status_item.text(),
            })
        df = pd.DataFrame(rows)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save CSV", "roi_results.csv", "CSV Files (*.csv)")
        if path:
            df.to_csv(path, index=False)

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
