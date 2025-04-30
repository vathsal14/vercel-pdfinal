from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from api.predict import app as predict_router
import os

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://parkinsons-prediction-b65mhm63p-vathsal14-gmailcoms-projects.vercel.app", "*"],  # Include your Vercel frontend URL
    allow_credentials=True,
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
