from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import generate_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                generate_data,
                inputs=["params:samples", "params:noise", "params:seed"],
                outputs=["x_values", "y_values"],
                name="generate_data",
            )
        ]
    )
