"""
https://github.com/onnx/onnx/blob/09ada0f107f1cc1877f9194475c98d2d8512e188/onnx/defs/tensor/defs.cc
"""

from webdnn.frontend.constraints import AxisVar
from webdnn.frontend.onnx.converter import ONNXConverter, attribute_dict
from webdnn.frontend.onnx.type_hint import INodeProto
from webdnn.graph.operators.reshape import Reshape
from webdnn.graph.operators.transpose import Transpose
from webdnn.graph.order import Order
from webdnn.util import console


@ONNXConverter.register_handler("Cast")
def _convert_cast(converter: ONNXConverter, onnx_op: INodeProto):
    console.warning("[ONNXConverter] Operator \"Cast\" is ignored")
    x = converter.get_variable(onnx_op.input[0])
    converter.set_variable(onnx_op.output[0], x)


@ONNXConverter.register_handler("Reshape")
def _convert_reshape(converter: ONNXConverter, onnx_op: INodeProto):
    x = converter.get_variable(onnx_op.input[0])
    attrs = attribute_dict(onnx_op)
    out_shape = list(attrs["shape"].ints)
    # noinspection PyTypeChecker
    out_order = Order([AxisVar() for _ in out_shape])

    y, = Reshape(None, in_order=x.order, out_order=out_order, out_shape=out_shape)(x)
    converter.set_variable(onnx_op.output[0], y)


@ONNXConverter.register_handler("Concat")
def _convert_concat(converter: ONNXConverter, onnx_op: INodeProto):
    # FIXME: It's possible to support in current version of webdnn
    raise NotImplementedError("[ONNXConverter] Operator \"Concat\" is not supported yet.")


@ONNXConverter.register_handler("Split")
def _convert_split(converter: ONNXConverter, onnx_op: INodeProto):
    # FIXME: It's possible to support in current version of webdnn
    raise NotImplementedError("[ONNXConverter] Operator \"Split\" is not supported yet.")


@ONNXConverter.register_handler("Slice")
def _convert_slice(converter: ONNXConverter, onnx_op: INodeProto):
    raise NotImplementedError("[ONNXConverter] Operator \"Slice\" is not supported yet.")


@ONNXConverter.register_handler("Transpose")
def _convert_transpose(converter: ONNXConverter, onnx_op: INodeProto):
    x = converter.get_variable(onnx_op.input[0])
    attrs = attribute_dict(onnx_op)

    y, = Transpose(None)(x)
    y.change_order(Order([x.order.axes[i] for i in list(attrs["perm"].ints)]))

    converter.set_variable(onnx_op.output[0], y)


@ONNXConverter.register_handler("Gather")
def _convert_gather(converter: ONNXConverter, onnx_op: INodeProto):
    raise NotImplementedError("[ONNXConverter] Operator \"Gather\" is not supported yet.")


@ONNXConverter.register_handler("Squeeze")
def _convert_squeeze(converter: ONNXConverter, onnx_op: INodeProto):
    # FIXME: It's possible to support in current version of webdnn
    raise NotImplementedError("[ONNXConverter] Operator \"Squeeze\" is not supported yet.")


@ONNXConverter.register_handler("Pad")
def _convert_pad(converter: ONNXConverter, onnx_op: INodeProto):
    # FIXME: It's possible to support in current version of webdnn
    raise NotImplementedError("[ONNXConverter] Operator \"Pad\" is not supported yet.")
