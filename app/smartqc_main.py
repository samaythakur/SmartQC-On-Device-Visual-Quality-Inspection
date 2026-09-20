"""
SmartQC — On-Device Visual Quality Inspection (Windows / Snapdragon)
 
Skeleton PyQt5 app. Wire your model into `InferenceEngine.predict()`.
Designed to run two backends side by side for the CPU-vs-NPU demo:
  - "cpu"  -> onnxruntime default CPUExecutionProvider
  - "npu"  -> onnxruntime QNNExecutionProvider (Snapdragon NPU)
 
Run:
    pip install PyQt5 opencv-python onnxruntime matplotlib numpy pillow
    python main.py
"""
 
import sys
import time
import sqlite3
from dataclasses import dataclass
from pathlib import Path
 
# Import onnxruntime BEFORE cv2 — opencv's bundled native DLLs can
# otherwise conflict with onnxruntime's DLL loading on Windows.
import onnxruntime  # noqa: F401
 
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QComboBox, QListWidget,
    QListWidgetItem, QGroupBox, QSplitter, QTabWidget, QSlider,
    QTableWidget, QTableWidgetItem
)
 
try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False
 
 
DB_PATH = Path(__file__).parent / "smartqc_history.db"
MODEL_PATH = Path(__file__).parent / "model" / "smartqc_model.onnx"  # your exported/optimized model
CLASS_NAMES = ["good", "defective"]  # edit to match your model's output classes
INPUT_SIZE = (224, 224)  # match your model's expected input
 
 
# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
 
class HistoryStore:
    def __init__(self, db_path: Path = DB_PATH):
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                latency_ms REAL NOT NULL,
                backend TEXT NOT NULL
            )
        """)
        self.conn.commit()
 
    def add(self, label: str, confidence: float, latency_ms: float, backend: str):
        self.conn.execute(
            "INSERT INTO inspections (ts, label, confidence, latency_ms, backend) VALUES (datetime('now'), ?, ?, ?, ?)",
            (label, confidence, latency_ms, backend),
        )
        self.conn.commit()
 
    def recent(self, limit=50):
        cur = self.conn.execute(
            "SELECT ts, label, confidence, latency_ms, backend FROM inspections ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return cur.fetchall()
 
    def defect_rate_series(self, limit=100):
        """Returns list of (index, is_defective) for the trend chart, oldest first."""
        cur = self.conn.execute(
            "SELECT label FROM inspections ORDER BY id DESC LIMIT ?", (limit,)
        )
        rows = [r[0] for r in cur.fetchall()][::-1]
        return [1 if lbl == "defective" else 0 for lbl in rows]
 
 
# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
 
@dataclass
class InferenceResult:
    label: str
    confidence: float
    latency_ms: float
    backend: str
 
 
class InferenceEngine:
    """
    Wraps an ONNX Runtime session. Swap `backend` between "cpu" and "npu"
    to power the benchmark toggle.
 
    NPU path expects the QNN Execution Provider to be available in your
    onnxruntime build (onnxruntime-qnn) and the model to have been
    optimized/quantized via Qualcomm AI Hub for the Snapdragon target.
    """
 
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self._sessions = {}  # backend -> ort.InferenceSession
        self._mock = not model_path.exists()  # falls back to mock until you drop in a real model
 
    def _get_session(self, backend: str):
        if backend in self._sessions:
            return self._sessions[backend]
 
        import onnxruntime as ort
 
        if backend == "npu":
            providers = ["QNNExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
 
        session = ort.InferenceSession(str(self.model_path), providers=providers)
        self._sessions[backend] = session
        return session
 
    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame_bgr, INPUT_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1))[None, ...].astype(np.float32)
        return img
 
    def predict(self, frame_bgr: np.ndarray, backend: str = "npu") -> InferenceResult:
        start = time.perf_counter()
 
        if self._mock:
            # Placeholder so the UI is demoable before the real model is wired in.
            # Replace this branch's absence (i.e. drop a real model at MODEL_PATH) to go live.
            probs = np.random.dirichlet(np.ones(len(CLASS_NAMES)))
        else:
            session = self._get_session(backend)
            input_name = session.get_inputs()[0].name
            x = self.preprocess(frame_bgr)
            outputs = session.run(None, {input_name: x})
            logits = outputs[0][0]
            exp = np.exp(logits - np.max(logits))
            probs = exp / exp.sum()
 
        latency_ms = (time.perf_counter() - start) * 1000.0
        idx = int(np.argmax(probs))
        return InferenceResult(
            label=CLASS_NAMES[idx],
            confidence=float(probs[idx]),
            latency_ms=latency_ms,
            backend=backend,
        )
 
 
# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
 
class TrendChart(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        if HAVE_MPL:
            self.figure = Figure(figsize=(4, 2.5))
            self.canvas = FigureCanvas(self.figure)
            layout.addWidget(self.canvas)
        else:
            layout.addWidget(QLabel("Install matplotlib to see the defect-rate trend chart."))
 
    def update_chart(self, series):
        if not HAVE_MPL:
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if series:
            # rolling defect rate over a small window
            window = 5
            rolling = [
                sum(series[max(0, i - window + 1): i + 1]) / min(window, i + 1)
                for i in range(len(series))
            ]
            ax.plot(rolling, color="#d64545")
            ax.set_ylim(0, 1)
            ax.set_title("Defect rate (rolling)", fontsize=9)
            ax.set_xlabel("Inspection #", fontsize=8)
        else:
            ax.text(0.5, 0.5, "No data yet", ha="center", va="center")
        self.canvas.draw()
 
 
class SmartQCWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SmartQC — On-Device Visual Quality Inspection")
        self.resize(1100, 650)
 
        self.engine = InferenceEngine()
        self.store = HistoryStore()
        self.cap = None
        self.current_frame = None
 
        self._build_ui()
 
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)
 
    # -- UI construction -----------------------------------------------
 
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
 
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)
 
        # Left: camera / preview
        left = QWidget()
        left_layout = QVBoxLayout(left)
 
        self.video_label = QLabel("Camera preview")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(480, 360)
        self.video_label.setStyleSheet("background:#111; color:#888; border-radius:6px;")
        left_layout.addWidget(self.video_label)
 
        controls = QHBoxLayout()
        self.start_btn = QPushButton("Start Webcam")
        self.start_btn.clicked.connect(self._toggle_webcam)
        controls.addWidget(self.start_btn)
 
        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self._upload_image)
        controls.addWidget(self.upload_btn)
 
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["cpu (local)", "npu (AI Hub optimized, pending EP deploy)"])
        controls.addWidget(QLabel("Backend:"))
        controls.addWidget(self.backend_combo)
 
        self.inspect_btn = QPushButton("Run Inspection")
        self.inspect_btn.clicked.connect(self._run_inspection)
        controls.addWidget(self.inspect_btn)
 
        left_layout.addLayout(controls)
 
        # -- Confidence threshold control --------------------------------
        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Confidence threshold:"))
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(50)
        self.threshold_slider.setMaximum(99)
        self.threshold_slider.setValue(70)
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.setTickInterval(5)
        self.threshold_slider.valueChanged.connect(self._update_threshold_label)
        threshold_row.addWidget(self.threshold_slider)
 
        self.threshold_value_label = QLabel("70%")
        self.threshold_value_label.setMinimumWidth(40)
        threshold_row.addWidget(self.threshold_value_label)
        left_layout.addLayout(threshold_row)
 
        self.threshold_hint = QLabel(
            "Predictions below this confidence are flagged UNCERTAIN instead of forced good/bad."
        )
        self.threshold_hint.setStyleSheet("color:#888; font-size:11px;")
        self.threshold_hint.setWordWrap(True)
        left_layout.addWidget(self.threshold_hint)
 
        self.result_label = QLabel("Result: —")
        self.result_label.setStyleSheet("font-size:16px; font-weight:bold; padding:8px;")
        left_layout.addWidget(self.result_label)
 
        self.latency_label = QLabel("Latency: —")
        left_layout.addWidget(self.latency_label)
 
        self.benchmark_btn = QPushButton("Run Timing Comparison (local CPU)")
        self.benchmark_btn.clicked.connect(self._run_benchmark)
        left_layout.addWidget(self.benchmark_btn)
 
        self.benchmark_note = QLabel(
            "Note: this compares repeated local inference calls, not a validated NPU benchmark.\n"
            "The model is compiled/optimized for the Snapdragon NPU via Qualcomm AI Hub\n"
            "(see the AI Hub job link in the README) — NPU execution provider deployment\n"
            "is the final integration step."
        )
        self.benchmark_note.setStyleSheet("color:#888; font-size:11px;")
        left_layout.addWidget(self.benchmark_note)
 
        self.benchmark_label = QLabel("")
        left_layout.addWidget(self.benchmark_label)
 
        splitter.addWidget(left)
 
        # Right: tabs for history + trend
        right = QTabWidget()
 
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        self.history_list = QListWidget()
        history_layout.addWidget(self.history_list)
        right.addTab(history_tab, "History")
 
        trend_tab = QWidget()
        trend_layout = QVBoxLayout(trend_tab)
        self.trend_chart = TrendChart()
        trend_layout.addWidget(self.trend_chart)
        right.addTab(trend_tab, "Trend")
 
        explain_tab = QWidget()
        explain_layout = QVBoxLayout(explain_tab)
        self.explain_label = QLabel(
            "Run 'Show Explainability Heatmap' below to see which region\n"
            "of the image most influenced the model's decision."
        )
        self.explain_label.setAlignment(Qt.AlignCenter)
        self.explain_label.setWordWrap(True)
        self.explain_label.setMinimumSize(400, 320)
        self.explain_label.setStyleSheet("background:#111; color:#888; border-radius:6px;")
        explain_layout.addWidget(self.explain_label)
 
        self.explain_btn = QPushButton("Show Explainability Heatmap (Grad-CAM)")
        self.explain_btn.clicked.connect(self._run_gradcam)
        explain_layout.addWidget(self.explain_btn)
 
        self.explain_status = QLabel("")
        self.explain_status.setWordWrap(True)
        self.explain_status.setStyleSheet("color:#888; font-size:11px;")
        explain_layout.addWidget(self.explain_status)
 
        right.addTab(explain_tab, "Explainability")
 
        batch_tab = QWidget()
        batch_layout = QVBoxLayout(batch_tab)
 
        batch_controls = QHBoxLayout()
        self.batch_folder_btn = QPushButton("Select Folder to Batch Inspect")
        self.batch_folder_btn.clicked.connect(self._run_batch_inspection)
        batch_controls.addWidget(self.batch_folder_btn)
        batch_layout.addLayout(batch_controls)
 
        self.batch_summary_label = QLabel("No batch run yet.")
        self.batch_summary_label.setStyleSheet("font-weight:bold; padding:4px;")
        batch_layout.addWidget(self.batch_summary_label)
 
        self.batch_table = QTableWidget()
        self.batch_table.setColumnCount(4)
        self.batch_table.setHorizontalHeaderLabels(["File", "Result", "Confidence", "Latency (ms)"])
        self.batch_table.horizontalHeader().setStretchLastSection(True)
        batch_layout.addWidget(self.batch_table)
 
        self.batch_export_btn = QPushButton("Export Results to CSV")
        self.batch_export_btn.clicked.connect(self._export_batch_csv)
        self.batch_export_btn.setEnabled(False)
        batch_layout.addWidget(self.batch_export_btn)
 
        right.addTab(batch_tab, "Batch Inspect")
 
        splitter.addWidget(right)
        splitter.setSizes([650, 450])
 
        self._refresh_history()
        self._gradcam = None  # lazy-loaded on first use
        self._batch_results = []  # populated by _run_batch_inspection
 
    # -- Webcam -----------------------------------------------------------
 
    def _toggle_webcam(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.timer.start(30)
            self.start_btn.setText("Stop Webcam")
        else:
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.start_btn.setText("Start Webcam")
 
    def _update_frame(self):
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            return
        self.current_frame = frame
        self._display_frame(frame)
 
    def _display_frame(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio
        )
        self.video_label.setPixmap(pix)
 
    def _upload_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select image", "", "Images (*.png *.jpg *.jpeg)")
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            return
        self.current_frame = frame
        self._display_frame(frame)
 
    # -- Inspection ---------------------------------------------------------
 
    def _update_threshold_label(self, value):
        self.threshold_value_label.setText(f"{value}%")
 
    def _run_inspection(self):
        if self.current_frame is None:
            self.result_label.setText("Result: no frame captured yet")
            return
        # both dropdown options currently execute on local CPU — the AI Hub
        # optimized model is compiled and validated for the Snapdragon NPU,
        # but the QNN execution provider is not yet wired up in this build.
        result = self.engine.predict(self.current_frame, backend="cpu")
        self._apply_result(result)
        self.store.add(result.label, result.confidence, result.latency_ms, result.backend)
        self._refresh_history()
 
    def _apply_result(self, result: InferenceResult):
        threshold = self.threshold_slider.value() / 100.0
        if result.confidence < threshold:
            color = "#e0a325"
            display_text = f"Result: UNCERTAIN  (best guess: {result.label.upper()}, {result.confidence*100:.1f}%)"
        else:
            color = "#2ecc71" if result.label == "good" else "#e74c3c"
            display_text = f"Result: {result.label.upper()}  ({result.confidence*100:.1f}%)"
        self.result_label.setText(display_text)
        self.result_label.setStyleSheet(f"font-size:16px; font-weight:bold; padding:8px; color:{color};")
        self.latency_label.setText(f"Latency: {result.latency_ms:.1f} ms  [{result.backend.upper()}]")
 
 
    def _run_benchmark(self):
        if self.current_frame is None:
            self.benchmark_label.setText("Capture or upload a frame first.")
            return
        run1 = self.engine.predict(self.current_frame, backend="cpu")
        run2 = self.engine.predict(self.current_frame, backend="cpu")
        self.benchmark_label.setText(
            f"Run 1: {run1.latency_ms:.1f} ms   |   Run 2: {run2.latency_ms:.1f} ms   "
            f"(both local CPU — see note above)"
        )
 
    def _run_gradcam(self):
        if self.current_frame is None:
            self.explain_status.setText("Capture or upload a frame first.")
            return
 
        self.explain_status.setText("Computing heatmap...")
        QApplication.processEvents()
 
        try:
            if self._gradcam is None:
                from gradcam_engine import GradCAM
                self._gradcam = GradCAM()
 
            overlay, label, confidence = self._gradcam.generate(self.current_frame)
 
            rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(
                self.explain_label.width(), self.explain_label.height(), Qt.KeepAspectRatio
            )
            self.explain_label.setPixmap(pix)
            clean_label = "good" if "good" in label.lower() or "fresh" in label.lower() else "defective"
            self.explain_status.setText(
                f"Heatmap for prediction: {clean_label.upper()} ({confidence*100:.1f}%). "
                f"Warmer colors (red/yellow) show the regions that most influenced this result."
            )
        except FileNotFoundError:
            self.explain_status.setText(
                "Model weights not found at model/smartqc_weights.pt — run train.py first."
            )
        except Exception as e:
            self.explain_status.setText(f"Could not generate heatmap: {e}")
 
    # -- Batch inspection -----------------------------------------------
 
    def _run_batch_inspection(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder of images to inspect")
        if not folder:
            return
 
        folder_path = Path(folder)
        image_exts = {".png", ".jpg", ".jpeg", ".bmp"}
        image_paths = sorted(
            p for p in folder_path.iterdir() if p.suffix.lower() in image_exts
        )
 
        if not image_paths:
            self.batch_summary_label.setText("No images found in that folder.")
            return
 
        threshold = self.threshold_slider.value() / 100.0
        results = []
        good_count = defective_count = uncertain_count = 0
 
        for path in image_paths:
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            result = self.engine.predict(frame, backend="cpu")
            if result.confidence < threshold:
                display_label = "UNCERTAIN"
                uncertain_count += 1
            elif result.label == "good":
                display_label = "GOOD"
                good_count += 1
            else:
                display_label = "DEFECTIVE"
                defective_count += 1
            results.append((path.name, display_label, result.confidence, result.latency_ms))
 
        self._batch_results = results
        self.batch_summary_label.setText(
            f"{len(results)} images inspected — {good_count} good, "
            f"{defective_count} defective, {uncertain_count} uncertain."
        )
 
        self.batch_table.setRowCount(len(results))
        for row, (name, label, conf, latency) in enumerate(results):
            self.batch_table.setItem(row, 0, QTableWidgetItem(name))
            self.batch_table.setItem(row, 1, QTableWidgetItem(label))
            self.batch_table.setItem(row, 2, QTableWidgetItem(f"{conf*100:.1f}%"))
            self.batch_table.setItem(row, 3, QTableWidgetItem(f"{latency:.1f}"))
 
        self.batch_export_btn.setEnabled(bool(results))
 
    def _export_batch_csv(self):
        if not self._batch_results:
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save batch results as CSV", "smartqc_batch_results.csv", "CSV Files (*.csv)"
        )
        if not save_path:
            return
        import csv
        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "result", "confidence", "latency_ms"])
            for name, label, conf, latency in self._batch_results:
                writer.writerow([name, label, f"{conf*100:.1f}%", f"{latency:.1f}"])
        self.batch_summary_label.setText(
            self.batch_summary_label.text() + f"  (saved to {Path(save_path).name})"
        )
 
    # -- History / trend -----------------------------------------------
 
    def _refresh_history(self):
        self.history_list.clear()
        for ts, label, conf, latency, backend in self.store.recent():
            item = QListWidgetItem(f"{ts}  |  {label}  ({conf*100:.1f}%)  |  {latency:.1f} ms  |  {backend}")
            self.history_list.addItem(item)
        self.trend_chart.update_chart(self.store.defect_rate_series())
 
 
def main():
    app = QApplication(sys.argv)
    window = SmartQCWindow()
    window.show()
    sys.exit(app.exec_())
 
 
if __name__ == "__main__":
    main()
 