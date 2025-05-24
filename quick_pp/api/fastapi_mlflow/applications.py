# -*- coding: utf-8 -*-
"""Build FastAPI applications for mlflow model predictions.

Copyright (C) 2022, Auto Trader UK

"""

import logging
from inspect import signature

from fastapi import FastAPI, Request
from fastapi.responses import ORJSONResponse
from mlflow.pyfunc import PyFuncModel  # type: ignore

from quick_pp.api.fastapi_mlflow.exceptions import DictSerialisableException
from quick_pp.api.fastapi_mlflow.predictors import build_predictor


def build_app(app, pyfunc_model: PyFuncModel, route: str) -> FastAPI:
    """Build and return a FastAPI app for the mlflow model."""
    logger = logging.getLogger(__name__)
    predictor = build_predictor(pyfunc_model)
    response_model = signature(predictor).return_annotation

    # Get input and output schemas from the model metadata
    input_schema = pyfunc_model.metadata.get_input_schema()
    input_features = [item['name'] for item in input_schema.to_dict()]
    output_schema = pyfunc_model.metadata.get_output_schema()
    target_features = [item['name'] for item in output_schema.to_dict()]

    model_name = route.removeprefix('/')
    app.add_api_route(
        f"{route}",
        predictor,
        response_model=response_model,
        response_class=ORJSONResponse,
        methods=["POST"],
        description=f"Predict using {model_name} model with input features: {', '.join(input_features)} and "
        f"output features: {', '.join(target_features)}",
        operation_id=f"predict_{model_name}",
    )

    @app.exception_handler(DictSerialisableException)
    def handle_serialisable_exception(
        req: Request, exc: DictSerialisableException
    ) -> ORJSONResponse:
        nonlocal logger
        req_id = req.headers.get("x-request-id")
        extra = {"x-request-id": req_id} if req_id is not None else {}
        logger.exception(exc.message, extra=extra)
        return ORJSONResponse(
            status_code=500,
            content=exc.to_dict(),
        )

    @app.exception_handler(Exception)
    def handle_exception(req: Request, exc: Exception) -> ORJSONResponse:
        return handle_serialisable_exception(
            req, DictSerialisableException.from_exception(exc)
        )

    return app
