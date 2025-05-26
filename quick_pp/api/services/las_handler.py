from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List
from pathlib import Path
import os
import shutil
from hashlib import sha256

from quick_pp.las_handler import read_las_file_welly

router = APIRouter(prefix="/las_handler", tags=["File Handler"])

# Allowed file extensions
ALLOWED_EXTENSIONS = {".LAS", ".las"}


def validate_file_extension(filename: str):
    _, ext = os.path.splitext(filename)
    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}"
        )


def save_upload_file(upload_file: UploadFile, destination: Path) -> str:
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()
    return destination.name


def unique_id(upload_file: UploadFile) -> str:
    # Read the file content and hash it directly
    file_bytes = upload_file.file.read()
    upload_file.file.seek(0)  # Reset file pointer after reading
    return sha256(file_bytes).hexdigest()[:8]


def read_las_file(input_path: Path, destination: Path):
    try:
        with open(input_path, "rb") as f:
            df, _ = read_las_file_welly(f)
            df.to_parquet(destination, index=False)
        return {'message': f"File {destination.name} processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading LAS file: {str(e)}")


@router.post(
    "",
    summary="Process LAS Files",
    description=(
        """
        Processes uploaded LAS files by saving them and converting to Parquet format.

        Input:
        - files: List of LAS files uploaded as multipart/form-data. Each file must have a .las or .LAS extension.

        Request:
        - Content-Type: multipart/form-data
        - Field: files (required) — one or more LAS files to be processed.

        Example (using curl):
        curl -X POST "<host>/las_handler" -F "files=@/path/to/file1.las" -F "files=@/path/to/file2.las"

        Workflow:
        - For each file:
            1. Validate file extension (.las or .LAS).
            2. Generate a unique file name using a hash of the file content and the original filename.
            3. Save the raw LAS file to 'uploads/las/'.
            4. Convert the LAS file to Parquet and save to 'uploads/parquet/'.
            5. Raise HTTP 500 error if something goes wrong.
            6. Ensure the uploaded file is closed after processing.

        Returns:
        - message: Success message listing the original filenames processed.
        - file_paths: List of output Parquet file paths (relative to the server).

        Raises:
        - HTTPException 400: If file extension is invalid.
        - HTTPException 500: On any processing error.
        """
    ),
    operation_id="process_las_file_to_parquet",
)
async def process_las_file(files: List[UploadFile] = File(...)):
    """
    Asynchronously processes a list of uploaded LAS files by saving the raw files and converting them to Parquet.
    Args:
        files (List[UploadFile]): Uploaded LAS files, provided by FastAPI using File(...).
    Workflow:
        - For each file:
            1. Generate a unique file name using `unique_id(file)` and the original filename.
            2. Save the raw LAS file to 'uploads/las/'.
            3. Convert the LAS file to Parquet and save to 'uploads/parquet/'.
            4. Raise HTTP 500 error if something goes wrong.
            5. Ensure the uploaded file is closed after processing.
    Returns:
        dict: Success message and list of uploaded filenames.
    Raises:
        HTTPException: On error, raises HTTP 500 with a generic message.
    Notes:
        - Uses helper functions: `unique_id(file)`, `save_upload_file(file, path)`, and
          `read_las_file(input_path, output_path)`.
        - Designed as a FastAPI endpoint handler.
    """
    processed_file_paths = []
    for file in files:
        try:
            # Validate file extension
            validate_file_extension(str(file.filename))

            # Save the raw LAS file
            file_name = f"{unique_id(file)}-{file.filename}"
            raw_file_path = Path(f"uploads/las/{file_name}")
            raw_file_path.parent.mkdir(parents=True, exist_ok=True)
            save_upload_file(file, raw_file_path)

            # Process and save the LAS file
            file_name = file_name.split(".")[0]
            processed_file_path = Path(f"uploads/parquet/{file_name}.parquet")
            processed_file_path.parent.mkdir(parents=True, exist_ok=True)
            read_las_file(raw_file_path, processed_file_path)
            processed_file_paths.append(processed_file_path)

        except Exception:
            raise HTTPException(status_code=500, detail='Something went wrong')
        finally:
            file.file.close()

    return {
        "message": f"Successfuly converted the LAS files to parquet: {[file.filename for file in files]}",
        "file_paths": processed_file_paths,
    }
