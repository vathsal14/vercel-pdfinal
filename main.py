from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.predict import app as predict_router
import os

app = FastAPI()

# Configure CORS - allow all origins for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router, prefix="/api")

# Add a health check endpoint
@app.get("/")
async def root():
    return {"status": "healthy", "message": "Parkinson's Risk Prediction API is running"}

# For local development
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
