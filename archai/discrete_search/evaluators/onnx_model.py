from typing import Tuple, Union, List, Dict, Optional
import io

import torch
from overrides import overrides
import onnxruntime as rt

from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.dataset_provider import DatasetProvider
from archai.discrete_search.api.model_evaluator import ModelEvaluator
from archai.common.timing import MeasureBlockTime


class AvgOnnxLatency(ModelEvaluator):
    def __init__(self, input_shape: Union[Tuple[int, ...], List[Tuple[int, ...]]], num_trials: int = 1,
                 input_dtype: str = 'torch.FloatTensor', rand_range: Tuple[float, float] = (0.0, 1.0),
                 export_kwargs: Optional[Dict] = None, inf_session_kwargs: Optional[Dict] = None):
        """Evaluates the average ONNX Latency (in seconds) of an architecture. The latency is measured
        by running the model on random inputs and averaging the latency over `num_trials` trials.

        Args:
            input_shape (Union[Tuple, List[Tuple]]): input shape(s) of the model. If a list of shapes
                is provided, the model is assumed to have multiple inputs.
            
            num_trials (int, optional): Number of trials to run. Defaults to 1.
            input_dtype (str, optional): Data type of the input. Defaults to 'torch.FloatTensor'.
            rand_range (Tuple[float, float], optional): Range of random values to use for the input.
            export_kwargs (Optional[Dict], optional): Keyword arguments to pass to `torch.onnx.export`.
                Defaults to None.
            
            inf_session_kwargs (Optional[Dict], optional): Keyword arguments to pass to `onnxruntime.InferenceSession`.
                Defaults to None.
        """
        input_shapes = [input_shape] if isinstance(input_shape, tuple) else input_shape            
        
        rand_min, rand_max = rand_range
        self.sample_input = tuple([
            ((rand_max - rand_min) * torch.rand(*input_shape) + rand_min).type(input_dtype)
            for input_shape in input_shapes
        ])

        self.input_dtype = input_dtype
        self.rand_range = rand_range
        self.num_trials = num_trials
        self.export_kwargs = export_kwargs or dict()
        self.inf_session_kwargs = inf_session_kwargs or dict()

    @overrides
    def evaluate(self, model: ArchaiModel, dataset_provider: DatasetProvider,
                budget: Optional[float] = None) -> float:
        model.arch.to('cpu')

        # Exports model to ONNX
        exported_model_buffer = io.BytesIO()
        torch.onnx.export(
            model.arch, self.sample_input, exported_model_buffer,
            input_names=[f'input_{i}' for i in range(len(self.sample_input))],
            **self.export_kwargs
        )
        
        exported_model_buffer.seek(0)

        # Benchmarks ONNX model
        onnx_session = rt.InferenceSession(exported_model_buffer.read(), **self.inf_session_kwargs)
        sample_input = {f'input_{i}': inp.numpy() for i, inp in enumerate(self.sample_input)}
        inf_times = []

        for _ in range(self.num_trials):
            with MeasureBlockTime('onnx_inference') as t:
                onnx_session.run(None, input_feed=sample_input)
            inf_times.append(t.elapsed)

        return sum(inf_times) / self.num_trials
