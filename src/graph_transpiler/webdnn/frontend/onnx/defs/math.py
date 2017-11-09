from webdnn.frontend.constraints import unify_order
from webdnn.frontend.onnx.converter import ONNXConverter, attribute_dict
from webdnn.frontend.onnx.defs.util import check_broadcast_constraints
from webdnn.frontend.onnx.type_hint import INodeProto
from webdnn.graph.operators.linear import Linear
from webdnn.graph.operators.relu import Relu
from webdnn.graph.order import OrderNC, OrderCN


@ONNXConverter.register_handler("Add")
def _convert_add(converter: ONNXConverter, onnx_op: INodeProto):
    x0 = converter.get_variable(onnx_op.input[0])
    x1 = converter.get_variable(onnx_op.input[1])

    attrs = attribute_dict(onnx_op)

    if "broadcast" in attrs:
        broadcast = attrs["broadcast"].i
        if broadcast:
            check_broadcast_constraints(x0, x1, axis=attrs["axis"].i if "axis" in attrs else None)

        else:
            unify_order(x0.order, x1.order)
    else:
        unify_order(x0.order, x1.order)

    y = x0 + x1
    converter.set_variable(onnx_op.output[0], y)


@ONNXConverter.register_handler("Gemm")
def _convert_gemm(converter: ONNXConverter, onnx_op: INodeProto):
    A = converter.get_variable(onnx_op.input[0])
    B = converter.get_variable(onnx_op.input[1])
    C = converter.get_variable(onnx_op.input[2])

    attrs = attribute_dict(onnx_op)
    alpha = attrs["alpha"].f
    beta = attrs["beta"].f
    broadcast = attrs.get("broadcast", 0)

    unify_order(A.order, OrderNC)
    unify_order(B.order, OrderCN)
    y, = Linear(None)(A, B)

    if broadcast:
        check_broadcast_constraints(y, C)
    y = alpha * y + beta * C

    converter.set_variable(onnx_op.output[0], y)


@ONNXConverter.register_handler("Relu")
def _convert_relu(converter: ONNXConverter, onnx_op: INodeProto):
    x0 = converter.get_variable(onnx_op.input[0])

    y, = Relu(None)(x0)
    converter.set_variable(onnx_op.output[0], y)
