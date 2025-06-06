from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # 创建一个列表，用于存储输入值
    vals_list = list(vals)
    
    # 计算 f(x0, ..., xi + epsilon, ..., xn-1)
    vals_list[arg] += epsilon
    f_plus_epsilon = f(*vals_list)
    
    # 计算 f(x0, ..., xi - epsilon, ..., xn-1)
    vals_list[arg] -= 2 * epsilon
    f_minus_epsilon = f(*vals_list)
    
    # 计算中心差分公式
    derivative = (f_plus_epsilon - f_minus_epsilon) / (2 * epsilon)
    
    return derivative


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    ret: Iterable[Variable] = []
    seen = set()
    def visit(v0:Variable):
        if v0.unique_id in seen:
            return
        seen.add(v0.unique_id)
        for v in v0.parents:
            visit(v)
        ret.insert(0, v0)
    visit(variable)
    
    return ret


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    vars = topological_sort(variable)
    d_dict = {}
    for var in vars:
        d_dict[var.unique_id] = 0
    d_dict[variable.unique_id] = deriv
    # print([(v.unique_id, v) for v in vars])
    for var in vars:
        var:Variable
        if var.is_leaf():
            var.accumulate_derivative(d_dict[var.unique_id])
            continue
        
        for father, father_deriv in var.chain_rule(d_dict[var.unique_id]):
            d_dict[father.unique_id] += father_deriv
    # print(d_dict)
    return


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
