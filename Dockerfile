FROM python:3.10

# Avoids prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app
COPY . /app

# provide write permissions to the app directory
RUN chmod -R 777 /app
# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt
EXPOSE 8501
# Run Streamlit app
CMD ["streamlit", "run", "Predict.py", "--server.port=8501", "--server.enableCORS=false", "--server.enableXsrfProtection=false", "--server.address=0.0.0.0"]
