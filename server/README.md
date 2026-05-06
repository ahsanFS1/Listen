# PSL recognition server

WebSocket server that runs the same MediaPipe + TFLite pipeline as
`src/inference/psl_words_v2.py` but accepts JPEG frames over a
WebSocket so the Flutter mobile app can offload inference.

## Run

```bash
cd server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
# or: uvicorn app:app --host 0.0.0.0 --port 8000
```

The model and label encoder are read from `../models/psl_words/`.

## Protocol

- `GET /healthz` → `{"ok": true}`
- `WS /ws/translate`
  - **Client → server (binary):** raw JPEG bytes for one camera frame.
  - **Server → client (text JSON):**
    ```json
    {
      "state": "SIGNING",
      "label": "hello",
      "english": "hello",
      "urdu": "ہیلو",
      "confidence": 0.92,
      "committed": false,
      "hasHands": true,
      "bufferFill": 60,
      "bufferCapacity": 60,
      "error": null
    }
    ```

`committed` is `true` for exactly one frame when a word has just been
recognized above the confidence threshold.
