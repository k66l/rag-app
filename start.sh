#!/bin/bash

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Load environment variables
export $(grep -v '^#' .env | xargs)

# Run the FastAPI app using values from .env
uvicorn app.main:app --reload --host $HOST --port $PORT
