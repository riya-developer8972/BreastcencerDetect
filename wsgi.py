from waitress import serve
from app import app   # Import Flask app from app.py

if __name__ == "__main__":
    print("🚀 Server running at http://127.0.0.1:8000")
    serve(app, host="0.0.0.0", port=8000)
