"""本项目的异常类型。

全部继承自 ``RcaError``,便于上层(V2 的 L1 循环)统一捕获并写入状态表。
每个异常都对应简报第 5 节中「组件职责边界」的一条硬约束,
在边界被越过时**立刻失败**,而不是产出一个看起来合理的错误数字。
"""

from __future__ import annotations


class RcaError(Exception):
    """所有项目异常的基类。"""


class KnowledgeError(RcaError):
    """知识库(YAML)加载或校验失败。"""


class MetricNotFoundError(KnowledgeError):
    """引用了不存在的指标。"""


class DimensionNotFoundError(KnowledgeError):
    """引用了不存在的维度。"""


class DecompositionError(RcaError):
    """分解请求本身不合法(与数据无关)。"""


class MetricNotAdditiveError(DecompositionError):
    """对比率型指标请求了加法维度分解。

    比率指标(CVR / AOV)的各分项之和不等于总体比率,
    强行相加会得到一个算术上错误、但看起来很像结论的数字。
    V0 直接拒绝;mix/rate 拆分留到 V1+。
    """


class FilterNotSupportedError(DecompositionError):
    """过滤条件在本次分解涉及的某张表上不可用。

    典型场景:按 category 过滤去做 UV×CVR×AOV 分解 ——
    category 只存在于 dim_product(经 fact_orders 关联),
    fact_sessions 上根本没有这个概念,UV 无法按同口径过滤,
    恒等式会破裂。
    """


class PeriodError(DecompositionError):
    """时间窗口不合法(顺序颠倒或两窗口重叠)。"""


class NonPositiveValueError(DecompositionError):
    """对数分解要求所有因子在两个窗口内均为正数。"""


class QueryExecutionError(RcaError):
    """SQL 执行失败(观察层 L1)。"""
