from kedro.pipeline import Pipeline, node
from .nodes import evaluate_model, test_model

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=evaluate_model,
                inputs="trained_model_path",
                outputs="model_metrics",
                name="evaluate_model_node",
            ),
            node(
                func=test_model,
                inputs=["trained_model_path", "params:test_data_path"],
                outputs="test_metrics",
                name="test_model_node",
            ),
        ]
    )
