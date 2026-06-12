import sys
import pickle
from spam_detector import SpamDetector
import re
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QTextEdit,
    QPushButton,
    QVBoxLayout,
    QProgressBar,
    QMessageBox
)

from PyQt6.QtCore import Qt


class SpamDetectorGUI(QWidget):

    def __init__(self):
        super().__init__()

        try:
            with open(r"S:\\Spam-Detection\\model\\spam_detector.pkl", "rb") as file:
                self.detector = pickle.load(file)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Could not load model:\n{e}"
            )
            sys.exit()

        self.init_ui()

    def init_ui(self):

        self.setWindowTitle("Email Spam Detector")
        self.resize(700, 500)

        title = QLabel("Email Spam Detector")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            padding: 10px;
        """)

        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText(
            "Paste or type an email message..."
        )

        self.detect_btn = QPushButton("Detect Spam")
        self.detect_btn.clicked.connect(self.detect_spam)

        self.result_label = QLabel("Result: -")
        self.result_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
        """)

        self.prob_label = QLabel("Spam Probability: -")

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        layout = QVBoxLayout()

        layout.addWidget(title)
        layout.addWidget(self.text_input)
        layout.addWidget(self.detect_btn)
        layout.addWidget(self.result_label)
        layout.addWidget(self.prob_label)
        layout.addWidget(self.progress)

        self.setLayout(layout)

    def detect_spam(self):

        text = self.text_input.toPlainText().strip()

        if not text:
            QMessageBox.warning(
                self,
                "Input Required",
                "Please enter an email message."
            )
            return

        result = self.detector._predict(text)

        prediction = result["Prediction"]
        probability = result["Probability"]

        self.result_label.setText(
            f"Result: {prediction}"
        )

        self.prob_label.setText(
            f"Spam Probability: {probability:.2f}%"
        )

        self.progress.setValue(int(probability))

        if prediction == "Spam":
            self.result_label.setStyleSheet("""
                color:red;
                font-size:16px;
                font-weight:bold;
            """)
        else:
            self.result_label.setStyleSheet("""
                color:green;
                font-size:16px;
                font-weight:bold;
            """)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = SpamDetectorGUI()
    window.show()

    sys.exit(app.exec())