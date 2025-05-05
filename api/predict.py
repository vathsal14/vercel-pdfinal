import pickle
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image, datasets, masking
import dicom2nifti
import pydicom
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
from io import BytesIO
import os
import tempfile
import time
import threading
import logging
import warnings
import gzip
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# FastAPI router
app = APIRouter()

# Global variables for caching
atlas_data = None
atlas_labels = None
model_loaded = False
final_model = None
scaler_X = None
scaler_y = None
model_lock = threading.Lock()

# Load models in background thread to avoid blocking startup
def load_models_background():
    global final_model, scaler_X, scaler_y, model_loaded
    
    try:
        # Load the trained model and scalers
        model_path = "model/Parkinson_Model.pkl"
        scaler_X_path = "model/scaler.pkl"
        scaler_y_path = "model/scaler_y.pkl"
        
        # Check if model files exist
        for path in [model_path, scaler_X_path, scaler_y_path]:
            if not os.path.exists(path):
                logger.error(f"Error: Model file not found: {path}")
                return
        
        with model_lock:
            # Suppress warnings during model loading
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                final_model = pickle.load(open(model_path, 'rb'))
                scaler_X = pickle.load(open(scaler_X_path, 'rb'))
                scaler_y = pickle.load(open(scaler_y_path, 'rb'))
            model_loaded = True
            logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")

# Start loading models in background
threading.Thread(target=load_models_background, daemon=True).start()

# Preload atlas data
def preload_atlas():
    global atlas_data, atlas_labels
    try:
        atlas = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-1mm')
        atlas_data = atlas.filename
        atlas_labels = atlas.labels
        logger.info("Atlas data loaded successfully")
    except Exception as e:
        logger.error(f"Error preloading atlas: {str(e)}")

# Start preloading atlas in background
threading.Thread(target=preload_atlas, daemon=True).start()

# Optimized function to process the uploaded file
def process_file(file: UploadFile):
    global atlas_data, atlas_labels
    
    # Wait for atlas data to be loaded
    start_time = time.time()
    max_wait = 10  # Maximum seconds to wait
    while atlas_data is None and time.time() - start_time < max_wait:
        time.sleep(0.1)
    
    if atlas_data is None:
        # If atlas still not loaded, load it now
        preload_atlas()
        if atlas_data is None:
            raise HTTPException(status_code=500, detail="Failed to load brain atlas data")
    
    # Create a temporary directory that will be automatically cleaned up
    with tempfile.TemporaryDirectory() as temp_dir:
        # Get the original filename and extension
        original_filename = file.filename
        logger.info(f"Processing file: {original_filename}")
        
        # Save uploaded file to the temporary directory
        temp_file_path = os.path.join(temp_dir, original_filename)
        logger.info(f"Saving to temporary file: {temp_file_path}")
        
        try:
            # Read the file content
            file_content = file.file.read()
            
            # Write to temporary file
            with open(temp_file_path, "wb") as f:
                f.write(file_content)
            
            # Check if DICOM file
            is_dicom = False
            try:
                dcm = pydicom.dcmread(temp_file_path)
                is_dicom = True
                logger.info("File detected as DICOM")
            except Exception as e:
                logger.info(f"Not a DICOM file: {str(e)}")
            
            # Convert DICOM to NIfTI if needed
            if is_dicom:
                # Create dicom folder
                dicom_folder = os.path.join(temp_dir, "dicom")
                os.makedirs(dicom_folder, exist_ok=True)
                
                # Move DICOM file to the folder
                dicom_file_path = os.path.join(dicom_folder, "image.dcm")
                shutil.copy(temp_file_path, dicom_file_path)
                
                # Convert DICOM to NIfTI
                nifti_output = os.path.join(temp_dir, "converted.nii.gz")
                dicom2nifti.convert_directory(dicom_folder, temp_dir)
                
                # Find the converted NIfTI file
                for file_name in os.listdir(temp_dir):
                    if file_name.endswith('.nii.gz') and not file_name.startswith('temp_file'):
                        nifti_output = os.path.join(temp_dir, file_name)
                        break
                
                datscan_img = nib.load(nifti_output)
                logger.info(f"Successfully converted DICOM to NIfTI: {nifti_output}")
            else:
                # Load as NIfTI file
                try:
                    datscan_img = nib.load(temp_file_path)
                    logger.info("Successfully loaded as NIfTI")
                except Exception as nifti_error:
                    error_msg = f"Unable to process file. Not recognized as DICOM or NIfTI format. Error: {str(nifti_error)}"
                    logger.error(error_msg)
                    raise HTTPException(status_code=400, detail=error_msg)
            
            # Extract brain data
            datscan_data = datscan_img.get_fdata()
            
            # Compute brain mask
            try:
                brain_mask = masking.compute_brain_mask(datscan_img)
                brain_data = datscan_data[brain_mask.get_fdata() > 0]
                logger.info(f"Computed brain mask with {np.sum(brain_mask.get_fdata() > 0)} voxels")
            except Exception as mask_error:
                logger.warning(f"Error computing brain mask: {str(mask_error)}. Using fallback method.")
                # Fallback to simple thresholding if nilearn mask fails
                brain_mask = np.ones_like(datscan_data) > 0
                non_zero_mask = datscan_data > 0
                brain_mask = np.logical_and(brain_mask, non_zero_mask)
                brain_data = datscan_data[brain_mask]
            
            if len(brain_data) == 0:
                # Fallback if no non-zero voxels
                logger.warning("No non-zero voxels found, using all voxels")
                brain_mask = np.ones_like(datscan_data) > 0
                brain_data = datscan_data[brain_mask]
                
                if len(brain_data) == 0:
                    raise ValueError("No valid voxel data found in the image")
            
            # Calculate whole brain mean for normalization
            brain_mean = np.mean(brain_data) if brain_data.size > 0 else 1.0
            logger.info(f"Whole brain mean value: {brain_mean}")
            
            # Resample atlas to match input image
            atlas_resampled = image.resample_to_img(atlas_data, datscan_img, interpolation='nearest')
            atlas_data_array = atlas_resampled.get_fdata()
            
            # Extract ROI values - following the user's previous implementation
            roi_values = {}
            
            for i, label in enumerate(atlas_labels):
                if not label:
                    continue
                
                if "Putamen" in label or "Caudate" in label:
                    region_mask = (atlas_data_array == i)
                    
                    if not np.any(region_mask):
                        logger.warning(f"Region {label} not found in atlas")
                        roi_values[label] = 0.0
                        continue
                    
                    region_voxels = datscan_data[region_mask]
                    
                    if region_voxels.size == 0:
                        logger.warning(f"No voxels found for region {label}")
                        roi_values[label] = 0.0
                        continue
                    
                    # Calculate region mean
                    region_mean = np.mean(region_voxels)
                    
                    # Normalize region mean value by the brain mean
                    normalized_value = region_mean / brain_mean
                    
                    roi_values[label] = normalized_value
                    logger.info(f"Region {label}: {normalized_value}")
            
            # Extracted Putamen and Caudate values
            putamen_r = roi_values.get('Right Putamen', 0.0)
            putamen_l = roi_values.get('Left Putamen', 0.0)
            caudate_r = roi_values.get('Right Caudate', 0.0)
            caudate_l = roi_values.get('Left Caudate', 0.0)
            
            # Log final values
            logger.info(f"Final values - Right Putamen: {putamen_r}, Left Putamen: {putamen_l}, Right Caudate: {caudate_r}, Left Caudate: {caudate_l}")
            
            return putamen_r, putamen_l, caudate_r, caudate_l
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing brain scan: {str(e)}")

# Prediction endpoint
@app.post("/")
async def predict(
    file: UploadFile = File(...),
    NP3TOT: int = Form(0),
    UPSIT_PRCNTGE: float = Form(0.0),
    COGCHG: int = Form(0)
):
    start_time = time.time()
    
    # Wait for models to be loaded
    if not model_loaded:
        max_wait = 5  # Maximum seconds to wait for model loading
        wait_start = time.time()
        while not model_loaded and time.time() - wait_start < max_wait:
            time.sleep(0.1)
        
        if not model_loaded:
            return {"error": "Models are still loading. Please try again in a few seconds."}
    
    try:
        # Process the uploaded file to get brain region values
        putamen_r, putamen_l, caudate_r, caudate_l = process_file(file)
        
        # Prepare the input data
        prediction_input = np.array([[
            putamen_r, putamen_l, 
            caudate_r, caudate_l, 
            NP3TOT, UPSIT_PRCNTGE, COGCHG
        ]])

        # Convert to DataFrame to match model input format
        input_df = pd.DataFrame(prediction_input, columns=[
            'DATSCAN_PUTAMEN_R', 'DATSCAN_PUTAMEN_L', 
            'DATSCAN_CAUDATE_R', 'DATSCAN_CAUDATE_L', 
            'NP3TOT', 'UPSIT_PRCNTGE', 'COGCHG'
        ])

        # Acquire lock before using the model
        with model_lock:
            # Scale the input features
            input_scaled = scaler_X.transform(input_df)
            
            # Make prediction
            pred_scaled = final_model.predict(input_scaled)
            
            # Convert the prediction to original scale
            risk_percent = float(scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0])

        # Determine risk status
        risk_status = "No significant risk detected (Negative)" if risk_percent <= 20 else "Significant risk detected (Positive)"

        processing_time = time.time() - start_time
        
        return {
            "right_putamen": float(putamen_r),
            "left_putamen": float(putamen_l),
            "right_caudate": float(caudate_r),
            "left_caudate": float(caudate_l),
            "risk_percent": float(risk_percent),
            "risk_status": risk_status,
            "processing_time_seconds": processing_time
        }

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error during prediction: {str(e)}, took {processing_time} seconds")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Test endpoint that doesn't require file upload
@app.post("/test")
async def predict(
    file: UploadFile = File(...),
    NP3TOT: int = Form(0),
    UPSIT_PRCNTGE: float = Form(0.0),
    COGCHG: int = Form(0)
):
    start_time = time.time()
    logger.info(f"Test prediction request received with params: NP3TOT={NP3TOT}, UPSIT={UPSIT_PRCNTGE}, COGCHG={COGCHG}")
    
    try:
        # Use dummy values for brain regions
        putamen_r, putamen_l, caudate_r, caudate_l = 0.8, 0.7, 0.6, 0.5
        
        # Prepare the input data
        prediction_input = np.array([[
            putamen_r, putamen_l, 
            caudate_r, caudate_l, 
            NP3TOT, UPSIT_PRCNTGE, COGCHG
        ]])

        # Convert to DataFrame to match model input format
        input_df = pd.DataFrame(prediction_input, columns=[
            'DATSCAN_PUTAMEN_R', 'DATSCAN_PUTAMEN_L', 
            'DATSCAN_CAUDATE_R', 'DATSCAN_CAUDATE_L', 
            'NP3TOT', 'UPSIT_PRCNTGE', 'COGCHG'
        ])

        # Acquire lock before using the model
        with model_lock:
            # Scale the input features
            input_scaled = scaler_X.transform(input_df)
            
            # Make prediction
            pred_scaled = final_model.predict(input_scaled)
            
            # Convert the prediction to original scale
            risk_percent = float(scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0])

        # Determine risk status
        risk_status = "No significant risk detected (Negative)" if risk_percent <= 20 else "Significant risk detected (Positive)"

        processing_time = time.time() - start_time
        
        return {
            "right_putamen": float(putamen_r),
            "left_putamen": float(putamen_l),
            "right_caudate": float(caudate_r),
            "left_caudate": float(caudate_l),
            "risk_percent": float(risk_percent),
            "risk_status": risk_status,
            "processing_time_seconds": processing_time,
            "test_mode": True
        }

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error during test prediction: {str(e)}, took {processing_time} seconds")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Vercel serverless function handler
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/api/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # Process the request using FastAPI app
            from io import BytesIO
            import json
            from fastapi import UploadFile, File, Form
            
            # Parse multipart form data
            try:
                # This is a simplified handler - in production you'd need proper multipart parsing
                # For now, we'll use the test endpoint for Vercel deployment
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                # Return a test response
                response = {
                    "right_putamen": 1.25,
                    "left_putamen": 1.35,
                    "right_caudate": 1.45,
                    "left_caudate": 1.55,
                    "risk_percent": 35.5,
                    "risk_status": "Significant risk detected (Positive)",
                    "processing_time_seconds": 2.5,
                    "note": "This is a test response from Vercel serverless function"
                }
                
                self.wfile.write(json.dumps(response).encode())
                return
                import pickle
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image, datasets, masking
import dicom2nifti
import pydicom
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
from io import BytesIO
import os
import tempfile
import time
import threading
import logging
import warnings
import gzip
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# FastAPI router
app = APIRouter()

# Global variables for caching
atlas_data = None
atlas_labels = None
model_loaded = False
final_model = None
scaler_X = None
scaler_y = None
model_lock = threading.Lock()

# Load models in background thread to avoid blocking startup
def load_models_background():
    global final_model, scaler_X, scaler_y, model_loaded
    
    try:
        # Load the trained model and scalers
        model_path = "model/Parkinson_Model.pkl"
        scaler_X_path = "model/scaler.pkl"
        scaler_y_path = "model/scaler_y.pkl"
        
        # Check if model files exist
        for path in [model_path, scaler_X_path, scaler_y_path]:
            if not os.path.exists(path):
                logger.error(f"Error: Model file not found: {path}")
                return
        
        with model_lock:
            # Suppress warnings during model loading
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                final_model = pickle.load(open(model_path, 'rb'))
                scaler_X = pickle.load(open(scaler_X_path, 'rb'))
                scaler_y = pickle.load(open(scaler_y_path, 'rb'))
            model_loaded = True
            logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")

# Start loading models in background
threading.Thread(target=load_models_background, daemon=True).start()

# Preload atlas data
def preload_atlas():
    global atlas_data, atlas_labels
    try:
        atlas = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-1mm')
        atlas_data = atlas.filename
        atlas_labels = atlas.labels
        logger.info("Atlas data loaded successfully")
    except Exception as e:
        logger.error(f"Error preloading atlas: {str(e)}")

# Start preloading atlas in background
threading.Thread(target=preload_atlas, daemon=True).start()

# Optimized function to process the uploaded file
def process_file(file: UploadFile):
    global atlas_data, atlas_labels
    
    # Wait for atlas data to be loaded
    start_time = time.time()
    max_wait = 10  # Maximum seconds to wait
    while atlas_data is None and time.time() - start_time < max_wait:
        time.sleep(0.1)
    
    if atlas_data is None:
        # If atlas still not loaded, load it now
        preload_atlas()
        if atlas_data is None:
            raise HTTPException(status_code=500, detail="Failed to load brain atlas data")
    
    # Create a temporary directory that will be automatically cleaned up
    with tempfile.TemporaryDirectory() as temp_dir:
        # Get the original filename and extension
        original_filename = file.filename
        logger.info(f"Processing file: {original_filename}")
        
        # Save uploaded file to the temporary directory
        temp_file_path = os.path.join(temp_dir, original_filename)
        logger.info(f"Saving to temporary file: {temp_file_path}")
        
        try:
            # Read the file content
            file_content = file.file.read()
            
            # Write to temporary file
            with open(temp_file_path, "wb") as f:
                f.write(file_content)
            
            # Check if DICOM file
            is_dicom = False
            try:
                dcm = pydicom.dcmread(temp_file_path)
                is_dicom = True
                logger.info("File detected as DICOM")
            except Exception as e:
                logger.info(f"Not a DICOM file: {str(e)}")
            
            # Convert DICOM to NIfTI if needed
            if is_dicom:
                try:
                    # Create dicom folder
                    dicom_folder = os.path.join(temp_dir, "dicom")
                    os.makedirs(dicom_folder, exist_ok=True)
                    
                    # Move DICOM file to the folder
                    dicom_file_path = os.path.join(dicom_folder, "image.dcm")
                    shutil.copy(temp_file_path, dicom_file_path)
                    
                    # Convert DICOM to NIfTI
                    nifti_output = os.path.join(temp_dir, "converted.nii.gz")
                    logger.info(f"Attempting to convert DICOM directory: {dicom_folder} to {temp_dir}")
                    dicom2nifti.convert_directory(dicom_folder, temp_dir)
                    
                    # Find the converted NIfTI file
                    nifti_file_found = False
                    for file_name in os.listdir(temp_dir):
                        if file_name.endswith('.nii.gz') and not file_name.startswith('temp_file'):
                            nifti_output = os.path.join(temp_dir, file_name)
                            nifti_file_found = True
                            break
                    
                    if not nifti_file_found:
                        logger.error("DICOM conversion completed but no NIfTI file was found")
                        raise Exception("DICOM conversion failed: No NIfTI file produced")
                    
                    datscan_img = nib.load(nifti_output)
                    logger.info(f"Successfully converted DICOM to NIfTI: {nifti_output}")
                except Exception as dicom_error:
                    logger.error(f"Error converting DICOM to NIfTI: {str(dicom_error)}")
                    raise HTTPException(status_code=500, detail=f"Error processing DICOM file: {str(dicom_error)}")
            else:
                # Load as NIfTI file
                try:
                    datscan_img = nib.load(temp_file_path)
                    logger.info("Successfully loaded as NIfTI")
                except Exception as nifti_error:
                    error_msg = f"Unable to process file. Not recognized as DICOM or NIfTI format. Error: {str(nifti_error)}"
                    logger.error(error_msg)
                    raise HTTPException(status_code=400, detail=error_msg)
            
            # Extract brain data
            datscan_data = datscan_img.get_fdata()
            
            # Compute brain mask
            try:
                brain_mask = masking.compute_brain_mask(datscan_img)
                brain_data = datscan_data[brain_mask.get_fdata() > 0]
                logger.info(f"Computed brain mask with {np.sum(brain_mask.get_fdata() > 0)} voxels")
            except Exception as mask_error:
                logger.warning(f"Error computing brain mask: {str(mask_error)}. Using fallback method.")
                # Fallback to simple thresholding if nilearn mask fails
                brain_mask = np.ones_like(datscan_data) > 0
                non_zero_mask = datscan_data > 0
                brain_mask = np.logical_and(brain_mask, non_zero_mask)
                brain_data = datscan_data[brain_mask]
            
            if len(brain_data) == 0:
                # Fallback if no non-zero voxels
                logger.warning("No non-zero voxels found, using all voxels")
                brain_mask = np.ones_like(datscan_data) > 0
                brain_data = datscan_data[brain_mask]
                
                if len(brain_data) == 0:
                    raise ValueError("No valid voxel data found in the image")
            
            # Calculate whole brain mean for normalization
            brain_mean = np.mean(brain_data) if brain_data.size > 0 else 1.0
            logger.info(f"Whole brain mean value: {brain_mean}")
            
            # Resample atlas to match input image
            atlas_resampled = image.resample_to_img(atlas_data, datscan_img, interpolation='nearest')
            atlas_data_array = atlas_resampled.get_fdata()
            
            # Extract ROI values - following the user's previous implementation
            roi_values = {}
            
            for i, label in enumerate(atlas_labels):
                if not label:
                    continue
                
                if "Putamen" in label or "Caudate" in label:
                    region_mask = (atlas_data_array == i)
                    
                    if not np.any(region_mask):
                        logger.warning(f"Region {label} not found in atlas")
                        roi_values[label] = 0.0
                        continue
                    
                    region_voxels = datscan_data[region_mask]
                    
                    if region_voxels.size == 0:
                        logger.warning(f"No voxels found for region {label}")
                        roi_values[label] = 0.0
                        continue
                    
                    # Calculate region mean
                    region_mean = np.mean(region_voxels)
                    
                    # Normalize region mean value by the brain mean
                    normalized_value = region_mean / brain_mean
                    
                    roi_values[label] = normalized_value
                    logger.info(f"Region {label}: {normalized_value}")
            
            # Extracted Putamen and Caudate values
            putamen_r = roi_values.get('Right Putamen', 0.0)
            putamen_l = roi_values.get('Left Putamen', 0.0)
            caudate_r = roi_values.get('Right Caudate', 0.0)
            caudate_l = roi_values.get('Left Caudate', 0.0)
            
            # Log final values
            logger.info(f"Final values - Right Putamen: {putamen_r}, Left Putamen: {putamen_l}, Right Caudate: {caudate_r}, Left Caudate: {caudate_l}")
            
            return putamen_r, putamen_l, caudate_r, caudate_l
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing brain scan: {str(e)}")

# Prediction endpoint
@app.post("/")
async def predict(
    file: UploadFile = File(...),
    NP3TOT: int = Form(0),
    UPSIT_PRCNTGE: float = Form(0.0),
    COGCHG: int = Form(0)
):
    start_time = time.time()
    
    # Wait for models to be loaded
    if not model_loaded:
        max_wait = 5  # Maximum seconds to wait for model loading
        wait_start = time.time()
        while not model_loaded and time.time() - wait_start < max_wait:
            time.sleep(0.1)
        
        if not model_loaded:
            return {"error": "Models are still loading. Please try again in a few seconds."}
    
    try:
        # Process the uploaded file to get brain region values
        putamen_r, putamen_l, caudate_r, caudate_l = process_file(file)
        
        # Prepare the input data
        prediction_input = np.array([[
            putamen_r, putamen_l, 
            caudate_r, caudate_l, 
            NP3TOT, UPSIT_PRCNTGE, COGCHG
        ]])

        # Convert to DataFrame to match model input format
        input_df = pd.DataFrame(prediction_input, columns=[
            'DATSCAN_PUTAMEN_R', 'DATSCAN_PUTAMEN_L', 
            'DATSCAN_CAUDATE_R', 'DATSCAN_CAUDATE_L', 
            'NP3TOT', 'UPSIT_PRCNTGE', 'COGCHG'
        ])

        # Acquire lock before using the model
        with model_lock:
            # Scale the input features
            input_scaled = scaler_X.transform(input_df)
            
            # Make prediction
            pred_scaled = final_model.predict(input_scaled)
            
            # Convert the prediction to original scale
            risk_percent = float(scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0])

        # Determine risk status
        risk_status = "No significant risk detected (Negative)" if risk_percent <= 20 else "Significant risk detected (Positive)"

        processing_time = time.time() - start_time
        
        return {
            "right_putamen": float(putamen_r),
            "left_putamen": float(putamen_l),
            "right_caudate": float(caudate_r),
            "left_caudate": float(caudate_l),
            "risk_percent": float(risk_percent),
            "risk_status": risk_status,
            "processing_time_seconds": processing_time
        }

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error during prediction: {str(e)}, took {processing_time} seconds")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Test endpoint that doesn't require file upload
@app.post("/test")
async def predict(
    file: UploadFile = File(...),
    NP3TOT: int = Form(0),
    UPSIT_PRCNTGE: float = Form(0.0),
    COGCHG: int = Form(0)
):
    start_time = time.time()
    logger.info(f"Test prediction request received with params: NP3TOT={NP3TOT}, UPSIT={UPSIT_PRCNTGE}, COGCHG={COGCHG}")
    
    try:
        # Use dummy values for brain regions
        putamen_r, putamen_l, caudate_r, caudate_l = 0.8, 0.7, 0.6, 0.5
        
        # Prepare the input data
        prediction_input = np.array([[
            putamen_r, putamen_l, 
            caudate_r, caudate_l, 
            NP3TOT, UPSIT_PRCNTGE, COGCHG
        ]])

        # Convert to DataFrame to match model input format
        input_df = pd.DataFrame(prediction_input, columns=[
            'DATSCAN_PUTAMEN_R', 'DATSCAN_PUTAMEN_L', 
            'DATSCAN_CAUDATE_R', 'DATSCAN_CAUDATE_L', 
            'NP3TOT', 'UPSIT_PRCNTGE', 'COGCHG'
        ])

        # Acquire lock before using the model
        with model_lock:
            # Scale the input features
            input_scaled = scaler_X.transform(input_df)
            
            # Make prediction
            pred_scaled = final_model.predict(input_scaled)
            
            # Convert the prediction to original scale
            risk_percent = float(scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0])

        # Determine risk status
        risk_status = "No significant risk detected (Negative)" if risk_percent <= 20 else "Significant risk detected (Positive)"

        processing_time = time.time() - start_time
        
        return {
            "right_putamen": float(putamen_r),
            "left_putamen": float(putamen_l),
            "right_caudate": float(caudate_r),
            "left_caudate": float(caudate_l),
            "risk_percent": float(risk_percent),
            "risk_status": risk_status,
            "processing_time_seconds": processing_time,
            "test_mode": True
        }

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error during test prediction: {str(e)}, took {processing_time} seconds")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Vercel serverless function handler
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/api/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # Process the request using FastAPI app
            from io import BytesIO
            import json
            from fastapi import UploadFile, File, Form
            
            # Parse multipart form data
            try:
                # This is a simplified handler - in production you'd need proper multipart parsing
                # For now, we'll use the test endpoint for Vercel deployment
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                # Return a test response
                response = {
                    "right_putamen": 1.25,
                    "left_putamen": 1.35,
                    "right_caudate": 1.45,
                    "left_caudate": 1.55,
                    "risk_percent": 35.5,
                    "risk_status": "Significant risk detected (Positive)",
                    "processing_time_seconds": 2.5,
                    "note": "This is a test response from Vercel serverless function"
                }
                
                self.wfile.write(json.dumps(response).encode())
                return
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
                return
        else:
            self.send_response(404)
            self.end_headers()
            return

            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
                return
        else:
            self.send_response(404)
            self.end_headers()
            return
