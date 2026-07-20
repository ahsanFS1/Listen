FROM python:3.11-slim

# libgl1/libglib2.0-0/etc are the usual runtime deps opencv-python-headless
# and mediapipe dlopen even in "headless" builds; libgomp1 is OpenMP for TF.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY server/requirements.txt ./server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt

COPY server/ ./server/
# Only the two model dirs the server actually loads (sign_session.py /
# alphabet_session.py) — models/train_models is an unused leftover artifact.
COPY models/psl_words ./models/psl_words
COPY models/psl ./models/psl

WORKDIR /app/server
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
