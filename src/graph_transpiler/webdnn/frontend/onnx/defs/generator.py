"""
https://github.com/onnx/onnx/blob/09ada0f107f1cc1877f9194475c98d2d8512e188/onnx/defs/generator/defs.cc
"""

from webdnn.frontend.onnx.converter import ONNXConverter
from webdnn.frontend.onnx.type_hint import INodeProto


@ONNXConverter.register_handler("Constant")
def _convert_constant(converter: ONNXConverter, onnx_op: INodeProto):
    # FIXME: It's possible to support in current version of webdnn
    raise NotImplementedError("[ONNXConverter] Operator \"Concat\" is not supported yet.")


@ONNXConverter.register_handler("RandomUniform")
def _convert_random_uniform(converter: ONNXConverter, onnx_op: INodeProto):
    raise NotImplementedError("[ONNXConverter] Operator \"RandomUniform\" is not supported yet.")


@ONNXConverter.register_handler("RandomNormal")
def _convert_random_normal(converter: ONNXConverter, onnx_op: INodeProto):
    raise NotImplementedError("[ONNXConverter] Operator \"RandomNormal\" is not supported yet.")


@ONNXConverter.register_handler("RandomUniformLike")
def _convert_random_uniform_like(converter: ONNXConverter, onnx_op: INodeProto):
    raise NotImplementedError("[ONNXConverter] Operator \"RandomUniformLike\" is not supported yet.")


@ONNXConverter.register_handler("RandomNormalLike")
def _convert_random_normal_like(converter: ONNXConverter, onnx_op: INodeProto):
    raise NotImplementedError("[ONNXConverter] Operator \"RandomNormalLike\" is not supported yet.")
