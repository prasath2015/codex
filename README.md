# Helmet Surveillance AI (Flask + Python)

This project provides a Flask web app that uses a laptop camera for surveillance and detects workers not wearing helmets.

## Features
- Live stream from laptop webcam.
- AI helmet classifier (`helmet` vs `no_helmet`).
- Real-time dashboard with worker count, violations, and alert captures.
- Automatic evidence capture:
  - Camera frame saved in `alerts/`
  - Desktop screenshot saved with **pyautogui** in `alerts/`

## 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Add a training dataset

Create a dataset structure:

```text
dataset/
  helmet/
  no_helmet/
```

Add many labeled worker head images in each folder.

## 3) Train the AI model

```bash
python train_model.py
```

This generates `models/helmet_classifier.h5`.

## 4) Download face cascade file

Download OpenCV Haar cascade and save it as:

`models/haarcascade_frontalface_default.xml`

(From OpenCV official GitHub repo)

## 5) Run the surveillance server

```bash
python app.py
```

Open http://localhost:5000

## Notes
- Webcam index `0` is used in `app.py`; change if needed.
- `pyautogui` screenshot may require GUI access (won't work in headless servers).
- Improve accuracy by training on domain-specific construction-site images.
