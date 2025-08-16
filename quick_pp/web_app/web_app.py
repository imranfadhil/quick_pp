"""
High-performance web application for quick_pp using FastAPI + HTMX
"""
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import io
import base64
import numpy as np
import matplotlib
import logging
import tempfile
import os

# Import quick_pp modules
from ..las_handler import read_las_file_welly
from ..lithology import sand_shale, sand_silt_clay, carbonate
from ..porosity import density_porosity, sonic_porosity_wyllie

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Quick PP Web Interface",
    description="High-performance web interface for quick_pp petrophysical analysis",
    version="1.0.0"
)

# Setup templates and static files
templates = Jinja2Templates(directory="quick_pp/web_app/templates")
app.mount("/static", StaticFiles(directory="quick_pp/web_app/static"), name="static")

# Global storage for uploaded data (in production, use Redis or database)
data_store = {}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """File upload page"""
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/upload-las")
async def upload_las_file(
    request: Request,
    file: UploadFile = File(...),
    well_name: str = Form("Unknown Well")
):
    """Handle LAS file upload and processing"""
    try:
        # Read LAS file
        content = await file.read()
        
        # Create a temporary file-like object for welly to read
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.las', delete=False) as temp_file:
            temp_file.write(content)
            temp_file.flush()
            
            # Use the las_handler function to read the file
            well_data, well_header = read_las_file_welly(temp_file)
            
            # Clean up temp file
            os.unlink(temp_file.name)
        
        # Store data with unique ID
        data_id = f"well_{len(data_store) + 1}"
        data_store[data_id] = {
            "well_name": well_name,
            "data": well_data,
            "curves": list(well_data.columns),
            "depth_range": [well_data.index.min(), well_data.index.max()]
        }
        
        # Return success response
        return HTMLResponse(f"""
        <div class="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded mb-4">
            <strong>Success!</strong> {well_name} loaded with {len(well_data)} depth points
        </div>
        <div class="bg-blue-100 border border-blue-400 text-blue-700 px-4 py-3 rounded mb-4">
            <strong>Data ID:</strong> {data_id}<br>
            <strong>Available Curves:</strong> {', '.join(data_store[data_id]['curves'])}
        </div>
        """)
        
    except Exception as e:
        logger.error(f"Error processing LAS file: {e}")
        return HTMLResponse(f"""
        <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
            <strong>Error:</strong> {str(e)}
        </div>
        """)

@app.get("/wells", response_class=HTMLResponse)
async def list_wells(request: Request):
    """List all loaded wells"""
    if not data_store:
        return HTMLResponse("<p class='text-gray-500'>No wells loaded yet.</p>")
    
    wells_html = ""
    for data_id, well_info in data_store.items():
        wells_html += f"""
        <div class="bg-white p-4 rounded-lg shadow border mb-4">
            <h3 class="text-lg font-semibold text-gray-800">{well_info['well_name']}</h3>
            <p class="text-sm text-gray-600">ID: {data_id}</p>
            <p class="text-sm text-gray-600">Depth: {well_info['depth_range'][0]:.1f} - {well_info['depth_range'][1]:.1f} ft</p>
            <p class="text-sm text-gray-600">Curves: {len(well_info['curves'])}</p>
            <div class="mt-3 space-x-2">
                <button class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
                        onclick="loadWell('{data_id}')">
                    Load
                </button>
                <button class="bg-red-500 hover:bg-red-700 text-white font-bold py-2 px-4 rounded"
                        onclick="deleteWell('{data_id}')">
                    Delete
                </button>
            </div>
        </div>
        """
    
    return HTMLResponse(wells_html)

@app.get("/well/{data_id}/analysis", response_class=HTMLResponse)
async def well_analysis_page(request: Request, data_id: str):
    """Well analysis page for a specific well"""
    if data_id not in data_store:
        return HTMLResponse("<p class='text-red-500'>Well not found.</p>")
    
    well_info = data_store[data_id]
    return templates.TemplateResponse("analysis.html", {
        "request": request,
        "data_id": data_id,
        "well_name": well_info["well_name"],
        "curves": well_info["curves"]
    })

@app.post("/analyze-lithology")
async def analyze_lithology(
    request: Request,
    data_id: str = Form(...),
    method: str = Form(...),
    gr_curve: str = Form(...),
    nphi_curve: str = Form(...),
    rhob_curve: str = Form(...)
):
    """Perform lithology analysis"""
    try:
        well_data = data_store[data_id]["data"]
        
        # Prepare input data
        gr = well_data[gr_curve].values
        nphi = well_data[nphi_curve].values
        rhob = well_data[rhob_curve].values
        
        # Perform lithology analysis based on method
        if method == "sand_shale":
            vsh, lithology = sand_shale.vsh_gr(gr)
        elif method == "sand_silt_clay":
            vsh, lithology = sand_silt_clay.vsh_gr(gr)
        elif method == "carbonate":
            vsh, lithology = carbonate.lithology(nphi, rhob)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        
        # Track 1: GR and VSH
        ax1.plot(gr, well_data.index, 'g-', label='GR', linewidth=0.8)
        ax1.plot(vsh * 100, well_data.index, 'r-', label='VSH (%)', linewidth=0.8)
        ax1.set_xlabel('GR / VSH (%)')
        ax1.set_ylabel('Depth (ft)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()
        
        # Track 2: Lithology
        colors = ['yellow', 'brown', 'gray', 'blue']
        for i, lith in enumerate(np.unique(lithology)):
            mask = lithology == lith
            ax2.fill_betweenx(well_data.index[mask], 0, 1, 
                             color=colors[i % len(colors)], alpha=0.7, label=lith)
        ax2.set_xlim(0, 1)
        ax2.set_xlabel('Lithology')
        ax2.set_ylabel('Depth (ft)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()
        
        plt.tight_layout()
        
        # Convert plot to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return HTMLResponse(f"""
        <div class="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded mb-4">
            <strong>Analysis Complete!</strong> {method.replace('_', ' ').title()} lithology analysis
        </div>
        <div class="text-center">
            <img src="data:image/png;base64,{img_base64}" class="max-w-full h-auto border rounded-lg shadow-lg" />
        </div>
        """)
        
    except Exception as e:
        logger.error(f"Error in lithology analysis: {e}")
        return HTMLResponse(f"""
        <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
            <strong>Error:</strong> {str(e)}
        </div>
        """)

@app.post("/analyze-porosity")
async def analyze_porosity(
    request: Request,
    data_id: str = Form(...),
    method: str = Form(...),
    nphi_curve: str = Form(...),
    rhob_curve: str = Form(...),
    dt_curve: str = Form(None)
):
    """Perform porosity analysis"""
    try:
        well_data = data_store[data_id]["data"]
        
        # Prepare input data
        nphi = well_data[nphi_curve].values
        rhob = well_data[rhob_curve].values
        dt = well_data[dt_curve].values if dt_curve else None
        
        # Perform porosity analysis
        if method == "density":
            phi = density_porosity(rhob, rho_matrix=2.65)  # Default matrix density
        elif method == "neutron":
            # For neutron porosity, we'll use a simple conversion if needed
            phi = nphi  # Assuming nphi is already in porosity units
        elif method == "density_neutron":
            # Use density porosity as fallback for combined method
            phi = density_porosity(rhob, rho_matrix=2.65)
        elif method == "sonic" and dt is not None:
            phi = sonic_porosity_wyllie(dt, dt_matrix=55.5, dt_fluid=189)  # Default values
        else:
            raise ValueError(f"Invalid method or missing data: {method}")
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        
        # Track 1: Input curves
        ax1.plot(nphi, well_data.index, 'b-', label='Neutron Porosity', linewidth=0.8)
        ax1.plot(rhob, well_data.index, 'g-', label='Bulk Density', linewidth=0.8)
        if dt is not None:
            ax1.plot(dt, well_data.index, 'r-', label='Sonic Transit Time', linewidth=0.8)
        ax1.set_xlabel('Input Values')
        ax1.set_ylabel('Depth (ft)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()
        
        # Track 2: Calculated porosity
        ax2.plot(phi * 100, well_data.index, 'purple', label='Porosity (%)', linewidth=1.2)
        ax2.set_xlabel('Porosity (%)')
        ax2.set_ylabel('Depth (ft)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()
        
        plt.tight_layout()
        
        # Convert plot to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return HTMLResponse(f"""
        <div class="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded mb-4">
            <strong>Analysis Complete!</strong> {method.replace('_', ' ').title()} porosity analysis
        </div>
        <div class="text-center">
            <img src="data:image/png;base64,{img_base64}" class="max-w-full h-auto border rounded-lg shadow-lg" />
        </div>
        """)
        
    except Exception as e:
        logger.error(f"Error in porosity analysis: {e}")
        return HTMLResponse(f"""
        <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
            <strong>Error:</strong> {str(e)}
        </div>
        """)

@app.get("/api/wells")
async def api_wells():
    """API endpoint to get list of wells"""
    return {"wells": list(data_store.keys()), "data": data_store}

@app.delete("/api/well/{data_id}")
async def delete_well(data_id: str):
    """Delete a well from storage"""
    if data_id in data_store:
        del data_store[data_id]
        return {"message": f"Well {data_id} deleted successfully"}
    return {"error": "Well not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
