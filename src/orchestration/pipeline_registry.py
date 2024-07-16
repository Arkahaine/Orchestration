from orchestration.pipelines.data_processing import pipeline as dp
from orchestration.pipelines.model_training import pipeline as mt
from orchestration.pipelines.model_evaluation import pipeline as me

def register_pipelines():
    return {
        "dp": dp.create_pipeline(),
        "mt": mt.create_pipeline(),
        "me": me.create_pipeline(),
        "__default__": dp.create_pipeline() + mt.create_pipeline() + me.create_pipeline(),
    }