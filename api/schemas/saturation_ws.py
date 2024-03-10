from pydantic import BaseModel
from typing import List


class data(BaseModel):
    rt: float
    rw: float
    phit: float
    qv: float
    b: float
    m: float


class swt_ws_inputData(BaseModel):
    data: List[data]


SWT_WS_EXAMPLE = {
    'data': [
        {'rt': 1.834, 'rw': 0.342, 'phit': 0.148, 'qv': 1.197, 'b': 5.479, 'm': 2},
        {'rt': 2.123, 'rw': 0.338, 'phit': 0.122, 'qv': 1.545, 'b': 5.541, 'm': 2},
        {'rt': 12.366, 'rw': 0.334, 'phit': 0.459, 'qv': 0.063, 'b': 5.628, 'm': 2},
        {'rt': 2.414, 'rw': 0.336, 'phit': 0.107, 'qv': 1.573, 'b': 5.596, 'm': 2},
        {'rt': 3.213, 'rw': 0.336, 'phit': 0.12, 'qv': 1.353, 'b': 5.579, 'm': 2},
        {'rt': 3.248, 'rw': 0.342, 'phit': 0.229, 'qv': 0.34, 'b': 5.477, 'm': 2},
        {'rt': 1.687, 'rw': 0.339, 'phit': 0.133, 'qv': 1.244, 'b': 5.53, 'm': 2},
        {'rt': 2.172, 'rw': 0.344, 'phit': 0.125, 'qv': 1.518, 'b': 5.439, 'm': 2},
        {'rt': 1.924, 'rw': 0.333, 'phit': 0.39, 'qv': 0.071, 'b': 5.643, 'm': 2},
        {'rt': 1.82, 'rw': 0.341, 'phit': 0.152, 'qv': 1.175, 'b': 5.494, 'm': 2}
    ]
}
