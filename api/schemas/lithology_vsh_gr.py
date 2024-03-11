from pydantic import BaseModel
from typing import List


class data(BaseModel):
    gr: float


class inputData(BaseModel):
    data: List[data]


EXAMPLE = {'data': [
    {'gr': 121.54},
    {'gr': 120.133},
    {'gr': 117.059},
    {'gr': 124.022},
    {'gr': 125.076},
    {'gr': 86.06},
    {'gr': 130.02},
    {'gr': 131.419},
    {'gr': 103.284},
    {'gr': 102.317},
]}
