from fastapi import APIRouter
import requests

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.get("/get_flows")
async def get_flows():

    url = "https://docs.langflow.org/api/v1/flows/?get_all=true"

    payload = {}
    headers = {
        "Accept": "application/json"
    }

    response = requests.request("GET", url, headers=headers, data=payload)

    return response.text
