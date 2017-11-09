"""
PyTorch Frontend
"""

from webdnn.frontend.converter import Converter
from webdnn.frontend.onnx import ONNXConverter
from webdnn.graph.graph import Graph
from webdnn.util import console

FLAG_PYTORCH_INSTALLED = False
try:
    import torch
    import torch.onnx

    FLAG_PYTORCH_INSTALLED = True

except ImportError as e:
    console.debug("PyTorch is not completely installed.")
    pass

FLAG_ONNX_INSTALLED = False
try:
    import onnx

    FLAG_ONNX_INSTALLED = True

except ImportError as e:
    console.debug("ONNX is not completely installed.")
    pass


class PyTorchConverter(Converter["torch.nn.Module"]):
    """PyTorchConverter()"""

    def __init__(self):
        if not FLAG_PYTORCH_INSTALLED:
            raise ImportError("""
Module "pytorch" cannot be imported. Please check that follow command works correctly.
    
    python -c "import torch"

""")

        if not FLAG_ONNX_INSTALLED:
            raise ImportError("""
Module "onnx" cannot be imported. Please check that follow command works correctly.

    python -c "import onnx"

""")

    def convert(self, model: "torch.nn.Module", dummy_inputs) -> Graph:
        """convert(model)

        Convert PyTorch computational graph into WebDNN IR.

        Returns:
            (:class:`~webdnn.Graph`): WebDNN Graph
        """
        proto_path = "/tmp/model.proto"
        torch.onnx.export(model, dummy_inputs, proto_path, verbose=False)
        return ONNXConverter().convert(onnx.load(proto_path))
