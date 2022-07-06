from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import convert_noise_type, generate_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                convert_noise_type,
                inputs="params:noise_type",
                outputs="noise_type",
                name="convert_noise_type",
            ),
            node(
                generate_data,
                inputs=["params:samples", "params:noise", "params:seed", "noise_type"],
                outputs=["x_values", "y_values", "scale_x", "scale_y"],
                name="generate_data",
            ),
            node(
                generate_data,
                inputs=[
                    "params:samples",
                    "params:valnoise",
                    "params:valseed",
                    "noise_type",
                ],
                outputs=[
                    "x_validation",
                    "y_validation",
                    "scale_x_validation",
                    "scale_y_validation",
                ],
                name="generate_validation_data",
            ),
            node(
                generate_data,
                inputs="params:samples",
                outputs=["x_test", "y_test", "scale_x_test", "scale_y_test"],
                name="generate_test_data",
            ),
        ]
    )
