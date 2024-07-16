from kedro.pipeline import Pipeline, node
from .nodes import train_yolo

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=train_yolo,
                inputs=["params:data_yaml_path", "params:img_size", "params:epochs", "params:batch_size"],
                outputs="trained_model_path",
                name="train_yolo_node",
            ),
        ]
    )
