from .bicubic import BicubicDatasetTrain, BicubicDatasetEval
from .div2k import DIV2KDatasetTrain, DIV2KDatasetEval
from .single_batch import SingleBatchDatasetTrain, SingleBatchDatasetEval

__all__ = [
    BicubicDatasetTrain.__name__,
    BicubicDatasetEval.__name__,
    DIV2KDatasetTrain.__name__,
    DIV2KDatasetEval.__name__,
    SingleBatchDatasetTrain.__name__,
    SingleBatchDatasetEval.__name__,
]
