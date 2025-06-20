from fastapi import APIRouter
import requests
import os

router = APIRouter(prefix="/langflow", tags=["Langflow"])

# Configuration
local_url = os.getenv("LANGFLOW_HOST", "http://localhost:7860")


@router.get("/get_projects")
async def get_projects():

    url = f"{local_url}/api/v1/projects/"

    payload = {}
    headers = {
        "Accept": "application/json"
    }

    response = requests.request("GET", url, headers=headers, data=payload)

    return response.json()


@router.get("/get_projects_id")
async def get_projects_id():
    projects = await get_projects()
    return [{"id": project["id"], "name": project["name"]} for project in projects]


@router.get("/get_flows")
async def get_flows():

    url = f"{local_url}/api/v1/flows/?remove_example_flows=true&get_all=true&header_flows=true"

    payload = {}
    headers = {
        "Accept": "application/json"
    }

    response = requests.request("GET", url, headers=headers, data=payload)

    return response.json()


@router.get("/get_flows_id")
async def get_flows_id():
    flows = await get_flows()
    return [{"id": flow["id"], "name": flow["name"]} for flow in flows]


@router.post("/chat")
async def chat(flow_id: str, question: str):
    import requests
    url = f"{local_url}/api/v1/run/{flow_id}"  # The complete API endpoint URL for this flow

    # Request payload configuration
    payload = {
        "input_value": question,  # The input value to be processed by the flow
        "output_type": "chat",  # Specifies the expected output format
        "input_type": "chat"  # Specifies the input format
    }

    # Request headers
    headers = {
        "Content-Type": "application/json"
    }

    try:
        # Send API request
        response = requests.request("POST", url, json=payload, headers=headers)
        response.raise_for_status()  # Raise exception for bad status codes

        # Extract the text message from the nested response structure
        response_data = response.json()

        # Navigate through the nested structure to get the text message
        if response_data and "outputs" in response_data:
            outputs = response_data["outputs"]
            if outputs and len(outputs) > 0:
                response_data = outputs[0]['outputs'][0]['results']

        # Fallback: return the full response if structure is different
        return response_data

    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        return {"error": f"Request error: {str(e)}"}
    except ValueError as e:
        print(f"Error parsing response: {e}")
        return {"error": f"Parsing error: {str(e)}"}
