from webdnn.backend.webassembly.optimize_rules.insert_transpose import InsertTranspose
from webdnn.backend.webassembly.optimize_rules.use_eigen import UseEigen
from webdnn.graph.optimize_rule import OptimizeRuleGroup
from webdnn.optimizer.sub_rules.constant_folding import ConstantFolding
from webdnn.optimizer.sub_rules.dump_graph import DumpGraph
from webdnn.optimizer.sub_rules.elementwise_kernel_fusion import ElementwiseKernelFusion
from webdnn.optimizer.sub_rules.merge_sgemm_and_elementwise_mul import MergeSgemmAndElementwiseMul
from webdnn.optimizer.sub_rules.replace_convolution_by_im2col import ReplaceConvolutionByIm2Col
from webdnn.optimizer.sub_rules.replace_deconvolution_by_col2im import ReplaceDeconvolutionByCol2Im
from webdnn.optimizer.sub_rules.replace_linear_by_sgemm import ReplaceLinearBySgemm
from webdnn.optimizer.sub_rules.update_inplace_attribute import UpdateInplaceAttribute
from webdnn.util import flags


class WebassemblyOptimizeRule(OptimizeRuleGroup):
    def __init__(self):
        sub_rules = [
            InsertTranspose(),

            ReplaceConvolutionByIm2Col(),
            MergeSgemmAndElementwiseMul(),
            ConstantFolding(),

            ReplaceDeconvolutionByCol2Im(),
            MergeSgemmAndElementwiseMul(),
            ConstantFolding(),

            ReplaceLinearBySgemm(),
            MergeSgemmAndElementwiseMul(),
            ConstantFolding(),

            UseEigen(),
            ElementwiseKernelFusion(),
            UpdateInplaceAttribute()
        ]

        if flags.DEBUG:
            sub_rules.append(DumpGraph("cg{count}.dot"))

        super(WebassemblyOptimizeRule, self).__init__(sub_rules)
