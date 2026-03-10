FROM python:3.11-slim

WORKDIR /app

# --- system deps (opencv / yolo ต้องใช้) ---
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# --- python deps ---
COPY requirements.txt .

RUN pip install --upgrade pip

# 🔒 Torch CPU (YOLO-compatible, stable)
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu \
    torchaudio==2.2.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# --- rest of deps ---
RUN pip install --no-cache-dir -r requirements.txt

# --- app ---
COPY . .

EXPOSE 10000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
