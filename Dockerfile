FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# system deps for opencv
RUN apt-get update && apt-get install -y --no-install-recommends 
# python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*




COPY src /app/src
COPY assets/ /app/assets
COPY readme.md .
COPY report.md .
