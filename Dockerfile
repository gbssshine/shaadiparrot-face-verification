FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Face matching models (face_match.py): official OpenCV model zoo, pinned commit, sha256-checked.
COPY tools/fetch_models.py tools/fetch_models.py
RUN python tools/fetch_models.py /app/models

COPY main.py face_live.py face_match.py fates_api.py fates_engine.py fates_ai.py fates_astro.py fates_astro_chart.py fates_tests_catalog.json places.sqlite3 ./

ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
