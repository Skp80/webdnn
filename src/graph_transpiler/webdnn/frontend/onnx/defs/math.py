"""
https://github.com/onnx/onnx/blob/09ada0f107f1cc1877f9194475c98d2d8512e188/onnx/defs/math/defs.cc
"""

from webdnn.frontend.constraints import unify_order
from webdnn.frontend.onnx.converter import ONNXConverter, attribute_dict
from webdnn.frontend.onnx.defs.util import check_broadcast_constraints
from webdnn.frontend.onnx.type_hint import INodeProto
from webdnn.graph.operators.elu import Elu
from webdnn.graph.operators.exp import Exp
from webdnn.graph.operators.leaky_relu import LeakyRelu
from webdnn.graph.operators.linear import Linear
from webdnn.graph.operators.relu import Relu
from webdnn.graph.operators.sigmoid import Sigmoid
from webdnn.graph.operators.softmax import Softmax
from webdnn.graph.operators.tanh import Tanh
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


@ONNXConverter.register_handler("Sub")
def _convert_sub(converter: ONNXConverter, onnx_op: INodeProto):
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

    y = x0 - x1
    converter.set_variable(onnx_op.output[0], y)


@ONNXConverter.register_handler("Mul")
def _convert_mul(converter: ONNXConverter, onnx_op: INodeProto):
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

    y = x0 * x1
    converter.set_variable(onnx_op.output[0], y)


@ONNXConverter.register_handler("Div")
def _convert_div(converter: ONNXConverter, onnx_op: INodeProto):
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

    y = x0 / x1
    converter.set_variable(onnx_op.output[0], y)


@ONNXConverter.register_handler("Neg")
def _convert_neg(converter: ONNXConverter, onnx_op: INodeProto):
    x0 = converter.get_variable(onnx_op.input[0])
    y = -x0
    converter.set_variable(onnx_op.output[0], y)


@ONNXConverter.register_handler("Abs")
def _convert_abs(converter: ONNXConverter, onnx_op: INodeProto):
    x0 = converter.get_variable(onnx_op.input[0])
    y = abs(x0)
    converter.set_variable(onnx_op.output[0], y)


@ONNXConverter.register_handler("Reciprocal")
def _convert_reciprocal(converter: ONNXConverter, onnx_op: INodeProto):
    x0 = converter.get_variable(onnx_op.input[0])
    y = 1 / x0
    converter.set_variable(onnx_op.output[0], y)


@ONNXConverter.register_handler("Floor")
def _convert_floor(converter: ONNXConverter, onnx_op: INodeProto):
    # FIXME: It's easy to support in current version of webdnn
    raise NotImplementedError("[ONNXConverter] \"Floor\" is not supported yet.")


@ONNXConverter.register_handler("Ceil")
def _convert_ceil(converter: ONNXConverter, onnx_op: INodeProto):
    # FIXME: It's easy to support in current version of webdnn
    raise NotImplementedError("[ONNXConverter] Operator \"Ceil\" is not supported yet.")


@ONNXConverter.register_handler("Sqrt")
def _convert_sqrt(converter: ONNXConverter, onnx_op: INodeProto):
    # FIXME: It's easy to support in current version of webdnn
    raise NotImplementedError("[ONNXConverter] Operator \"Sqrt\" is not supported yet.")


@ONNXConverter.register_handler("Relu")
def _convert_relu(converter: ONNXConverter, onnx_op: INodeProto):
    x0 = converter.get_variable(onnx_op.input[0])

    y, = Relu(None)(x0)
    converter.set_variable(onnx_op.output[0], y)


@ONNXConverter.register_handler("LeakyRelu")
def _convert_leaky_relu(converter: ONNXConverter, onnx_op: INodeProto):
    x0 = converter.get_variable(onnx_op.input[0])

    attrs = attribute_dict(onnx_op)
    alpha = attrs["alpha"].f

    y, = LeakyRelu(None, slope=alpha)(x0)
    converter.set_variable(onnx_op.output[0], y)


@ONNXConverter.register_handler("Selu")
def _convert_selu(converter: ONNXConverter, onnx_op: INodeProto):
    raise NotImplementedError("[ONNXConverter] Operator \"Selu\" is not supported yet.")


@ONNXConverter.register_handler("Elu")
def _convert_elu(converter: ONNXConverter, onnx_op: INodeProto):
    x0 = converter.get_variable(onnx_op.input[0])

    attrs = attribute_dict(onnx_op)
    alpha = attrs["alpha"].f
    if alpha != 1:
        raise NotImplementedError("[ONNXConverter] Operator \"Elu\" is supported only the case when parameter \"alpha\" is 1.")

    y, = Elu(None)(x0)
    converter.set_variable(onnx_op.output[0], y)


@ONNXConverter.register_handler("Exp")
def _convert_exp(converter: ONNXConverter, onnx_op: INodeProto):
    x0 = converter.get_variable(onnx_op.input[0])
    y, = Exp(None)(x0)
    converter.set_variable(onnx_op.output[0], y)


@ONNXConverter.register_handler("Log")
def _convert_log(converter: ONNXConverter, onnx_op: INodeProto):
    # FIXME: It's easy to support in current version of webdnn
    raise NotImplementedError("[ONNXConverter] Operator \"Log\" is not supported yet.")


@ONNXConverter.register_handler("Tanh")
def _convert_tanh(converter: ONNXConverter, onnx_op: INodeProto):
    x0 = converter.get_variable(onnx_op.input[0])
    y, = Tanh(None)(x0)
    converter.set_variable(onnx_op.output[0], y)


@ONNXConverter.register_handler("Pow")
def _convert_pow(converter: ONNXConverter, onnx_op: INodeProto):
    x0 = converter.get_variable(onnx_op.input[0])
    x1 = converter.get_variable(onnx_op.input[1])

    check_broadcast_constraints(x0, x1)

    y = x0 ** x1
    converter.set_variable(onnx_op.output[0], y)


@ONNXConverter.register_handler("PRelu")
def _convert_prelu(converter: ONNXConverter, onnx_op: INodeProto):
    # FIXME: It's possible to support in current version of webdnn
    raise NotImplementedError("[ONNXConverter] Operator \"PRelu\" is not supported yet.")


@ONNXConverter.register_handler("Sigmoid")
def _convert_sigmoid(converter: ONNXConverter, onnx_op: INodeProto):
    x0 = converter.get_variable(onnx_op.input[0])
    y, = Sigmoid(None)(x0)
    converter.set_variable(onnx_op.output[0], y)


@ONNXConverter.register_handler("Max")
def _convert_max(converter: ONNXConverter, onnx_op: INodeProto):
    raise NotImplementedError("[ONNXConverter] Operator \"Max\" is not supported yet.")


@ONNXConverter.register_handler("Min")
def _convert_min(converter: ONNXConverter, onnx_op: INodeProto):
    raise NotImplementedError("[ONNXConverter] Operator \"Min\" is not supported yet.")


@ONNXConverter.register_handler("Sum")
def _convert_sum(converter: ONNXConverter, onnx_op: INodeProto):
    xs = [converter.get_variable(proto) for proto in onnx_op.input]

    check_broadcast_constraints(xs[0], xs[1])
    y = xs[0] + xs[1]
    for x in xs[2:]:
        check_broadcast_constraints(x, y)
        y += x

    converter.set_variable(onnx_op.output[0], y)


@ONNXConverter.register_handler("Softmax")
def _convert_softmax(converter: ONNXConverter, onnx_op: INodeProto):
    x = converter.get_variable(onnx_op.input[0])

    attrs = attribute_dict(onnx_op)
    axis = attrs["axis"].i

    y, = Softmax(None, axis=x.order.axes[axis])(x)
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


@ONNXConverter.register_handler("MaxMul")
def _convert_matmul(converter: ONNXConverter, onnx_op: INodeProto):
    # FIXME: It's possible to support in current version of webdnn
    raise NotImplementedError("[ONNXConverter] Operator \"MaxMul\" is not supported yet.")
