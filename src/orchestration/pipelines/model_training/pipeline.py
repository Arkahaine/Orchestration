from kedro.pipeline import Pipeline, node
from .nodes import train_yolo, evaluate_model, test_model

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=train_yolo,
                inputs=["params:data_yaml_path", "params:img_size", "params:epochs", "params:batch_size"],
                outputs="trained_model_path",
                name="train_yolo_node",
            ),
            node(
                func=evaluate_model,
                inputs="trained_model_path",
                outputs="model_metrics",
                name="evaluate_model_node",
            ),
            node(
                func=test_model,
                inputs=["trained_model_path", "params:test_data_path"],
                outputs="test_results",
                name="test_model_node",
            ),
        ]
    )
