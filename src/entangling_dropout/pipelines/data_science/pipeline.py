"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import train, create_qnode, create_cost_fn, evaluate, generate_loss_curve


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                create_qnode,
                inputs=["params:wires", "params:layers", "params:shots"],
                outputs="qnode",
                name="create_qnode",
            ),
            node(
                create_cost_fn,
                inputs="qnode",
                outputs="cost_fn",
                name="create_cost_fn",
            ),
            node(
                train,
                inputs={
                    "x_values": "x_values",
                    "y_values": "y_values",
                    "x_validation": "x_validation",
                    "y_validation": "y_validation",
                    "wires": "params:wires",
                    "layers": "params:layers",
                    "stepsize": "params:stepsize",
                    "epochs": "params:epochs",
                    "batch_size": "params:batchsize",
                    "seed": "params:seed",
                    "qnode": "qnode",
                    "cost_fn": "cost_fn",
                },
                outputs=["trained_params", "history"],
                name="train",
            ),
            node(
                evaluate,
                inputs={
                    "params": "trained_params",
                    "x_values": "x_test",
                    "y_values": "y_test",
                    "cost_fn": "cost_fn",
                },
                outputs="loss_test",
                name="evaluate_test_data"
            ),
            node(
                generate_loss_curve,
                inputs=["history", "loss_test"],
                outputs="loss_curve",
                name="generate_loss_curve",
            ),
        ]
    )
