FROM python:3.11-slim
ENV PYTHONPATH=/app

WORKDIR /app

# Install deps first (better caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application package
COPY proteomics_app /app/proteomics_app

# Streamlit settings
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8525 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

CMD ["streamlit", "run", "proteomics_app/app.py", "--server.port=8525", "--server.address=0.0.0.0"]
