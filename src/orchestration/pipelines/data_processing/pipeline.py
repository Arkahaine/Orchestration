from kedro.pipeline import Pipeline, node
from .nodes import preprocess_data

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=preprocess_data,
                inputs="params:raw_data_path",
                outputs="preprocessed_data",
                name="preprocess_data_node",
            )
        ]
    )
