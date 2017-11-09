from webdnn.frontend.constraints import AxisVar
from webdnn.frontend.onnx.converter import ONNXConverter, attribute_dict
from webdnn.frontend.onnx.type_hint import INodeProto
from webdnn.graph.operators.reshape import Reshape
from webdnn.graph.operators.transpose import Transpose
from webdnn.graph.order import Order


@ONNXConverter.register_handler("Transpose")
def _convert_transpose(converter: ONNXConverter, onnx_op: INodeProto):
    x = converter.get_variable(onnx_op.input[0])
    attrs = attribute_dict(onnx_op)

    y, = Transpose(None)(x)
    y.change_order(Order([x.order.axes[i] for i in list(attrs["perm"].ints)]))

    converter.set_variable(onnx_op.output[0], y)


@ONNXConverter.register_handler("Reshape")
def _convert_reshape(converter: ONNXConverter, onnx_op: INodeProto):
    x = converter.get_variable(onnx_op.input[0])
    attrs = attribute_dict(onnx_op)
    out_shape = list(attrs["shape"].ints)
    # noinspection PyTypeChecker
    out_order = Order([AxisVar() for _ in out_shape])

    y, = Reshape(None, in_order=x.order, out_order=out_order, out_shape=out_shape)(x)
    converter.set_variable(onnx_op.output[0], y)
