from pydantic import BaseModel
from typing import List


class data(BaseModel):
    depth: float
    vclw: float
    phit: float
    swt: float
    perm: float
    zones: str


class inputData(BaseModel):
    cut_offs: tuple
    data: List[data]


EXAMPLE = {
    'cut_offs': [0.3, 0.1, 0.75],
    'data': [
        {
            'depth': 3898.24,
            'vclw': 0.702,
            'phit': 0.095,
            'swt': 1.0,
            'perm': 0,
            'zones': 'A'},
        {
            'depth': 3954.932,
            'vclw': 0.671,
            'phit': 0.108,
            'swt': 1.0,
            'perm': 0,
            'zones': 'A'},
        {
            'depth': 3804.056,
            'vclw': 0.698,
            'phit': 0.121,
            'swt': 1.0,
            'perm': 0,
            'zones': 'A'},
        {
            'depth': 3742.639,
            'vclw': 0.663,
            'phit': 0.149,
            'swt': 1.0,
            'perm': 0,
            'zones': 'B'},
        {
            'depth': 3641.75,
            'vclw': 0.635,
            'phit': 0.121,
            'swt': 1.0,
            'perm': 0,
            'zones': 'B'},
        {
            'depth': 3919.423,
            'vclw': 0.036,
            'phit': 0.483,
            'swt': 0.281,
            'perm': 0,
            'zones': 'B'},
        {
            'depth': 4014.368,
            'vclw': 0.352,
            'phit': 0.178,
            'swt': 0.801,
            'perm': 0,
            'zones': 'B'},
        {
            'depth': 3919.88,
            'vclw': 0.036,
            'phit': 0.483,
            'swt': 0.248,
            'perm': 0,
            'zones': 'C'},
        {
            'depth': 3864.712,
            'vclw': 0.007,
            'phit': 0.255,
            'swt': 0.343,
            'perm': 0,
            'zones': 'C'},
        {
            'depth': 4062.527,
            'vclw': 0.638,
            'phit': 0.188,
            'swt': 1.0,
            'perm': 0,
            'zones': 'C'}
    ],
}
