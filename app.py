import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import pyautogui
from flask import Flask, Response, jsonify, render_template
from tensorflow.keras.models import load_model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "helmet_classifier.h5")
CASCADE_PATH = os.path.join(BASE_DIR, "models", "haarcascade_frontalface_default.xml")
ALERT_DIR = os.path.join(BASE_DIR, "alerts")

os.makedirs(ALERT_DIR, exist_ok=True)


class HelmetSurveillance:
    """Stream processor that detects workers and checks helmet usage."""

    def __init__(self) -> None:
        self.capture = cv2.VideoCapture(0)
        self.model = load_model(MODEL_PATH)
        self.face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        self.lock = threading.Lock()
        self.latest_frame = None
        self.stats = {
            "total_people": 0,
            "without_helmet": 0,
            "alerts_triggered": 0,
        }
        self.last_alert_time = 0.0
        self.alert_cooldown_seconds = 5

    def _predict_helmet(self, roi: np.ndarray) -> float:
        image = cv2.resize(roi, (128, 128))
        image = image.astype("float32") / 255.0
        image = np.expand_dims(image, axis=0)
        prediction = float(self.model.predict(image, verbose=0)[0][0])
        return prediction

    def _save_alert_evidence(self, frame: np.ndarray) -> None:
        now = time.time()
        if now - self.last_alert_time < self.alert_cooldown_seconds:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frame_path = os.path.join(ALERT_DIR, f"no_helmet_{timestamp}.jpg")
        screen_path = os.path.join(ALERT_DIR, f"desktop_{timestamp}.jpg")

        cv2.imwrite(frame_path, frame)

        # pyautogui captures the desktop as additional evidence.
        screenshot = pyautogui.screenshot()
        screenshot.save(screen_path)

        self.stats["alerts_triggered"] += 1
        self.last_alert_time = now

    def process(self) -> None:
        while True:
            ok, frame = self.capture.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))

            total_people = len(faces)
            without_helmet = 0

            for (x, y, w, h) in faces:
                head_roi = frame[max(y - 20, 0): y + h, max(x - 20, 0): x + w + 20]
                if head_roi.size == 0:
                    continue

                helmet_prob = self._predict_helmet(head_roi)
                is_wearing_helmet = helmet_prob >= 0.5

                color = (0, 200, 0) if is_wearing_helmet else (0, 0, 255)
                label = f"Helmet {helmet_prob:.2f}" if is_wearing_helmet else f"NO HELMET {helmet_prob:.2f}"

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                if not is_wearing_helmet:
                    without_helmet += 1

            if without_helmet > 0:
                self._save_alert_evidence(frame)

            self.stats["total_people"] = total_people
            self.stats["without_helmet"] = without_helmet

            with self.lock:
                self.latest_frame = frame

    def frame_generator(self):
        while True:
            with self.lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()

            if frame is None:
                time.sleep(0.05)
                continue

            ret, encoded = cv2.imencode(".jpg", frame)
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
            )


app = Flask(__name__)
surveillance = HelmetSurveillance()
threading.Thread(target=surveillance.process, daemon=True).start()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(surveillance.frame_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats")
def stats():
    return jsonify(surveillance.stats)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
