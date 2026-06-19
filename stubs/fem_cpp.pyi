from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import typing
__all__: list[str] = ['FEM_1D', 'FEM_2D']
class FEM_1D:
    def __init__(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], arg2: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], arg3: collections.abc.Callable[[typing.SupportsFloat | typing.SupportsIndex], float], arg4: collections.abc.Callable[[typing.SupportsFloat | typing.SupportsIndex], float], arg5: collections.abc.Callable[[typing.SupportsFloat | typing.SupportsIndex], float], arg6: collections.abc.Callable[[typing.SupportsFloat | typing.SupportsIndex], float], arg7: collections.abc.Callable[[typing.SupportsFloat | typing.SupportsIndex], float], arg8: collections.abc.Callable[[typing.SupportsFloat | typing.SupportsIndex], float]) -> None:
        ...
    def assemble_matrix(self, K11: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], K12: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], D1: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        """
        Assemble the global stiffness matrix K and load vector D from the element contributions
        """
    def full_solve(self) -> list[tuple[str, float]]:
        """
        Run the full FEM solve process
        """
    def gen_K11_K12_D1(self) -> tuple[list[float], list[float], list[float]]:
        """
        Generate the K11, K12, and D1 vectors for each element
        """
    def gen_tlist(self) -> None:
        """
        Generate the tlist based on the plist
        """
    def get_Solution(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Get the computed solution vector Sol as a NumPy array
        """
    def reconstruct_solution(self) -> None:
        """
        Reconstruct the full solution vector Sol from the free nodes and Dirichlet boundary conditions
        """
    def solve_LGS(self) -> None:
        """
        Solve the linear system K * Sol_noRW = D
        """
    def validate_sol(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], arg1: typing.SupportsFloat | typing.SupportsIndex) -> tuple[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"], list[float]]:
        """
        Validate the computed solution against an analytical solution at the nodes, returning a vector of errors
        """
    @property
    def RESOLUTION(self) -> float:
        ...
    @RESOLUTION.setter
    def RESOLUTION(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class FEM_2D:
    def __init__(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.int32, "[m, 1]"], arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32, "[m, n]"], arg2: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"], arg3: typing.Annotated[numpy.typing.ArrayLike, numpy.int32, "[m, n]"], arg4: collections.abc.Callable[[typing.SupportsFloat | typing.SupportsIndex, typing.SupportsFloat | typing.SupportsIndex], float], arg5: collections.abc.Callable[[typing.SupportsFloat | typing.SupportsIndex, typing.SupportsFloat | typing.SupportsIndex], float], arg6: collections.abc.Callable[[typing.SupportsFloat | typing.SupportsIndex, typing.SupportsFloat | typing.SupportsIndex], float], arg7: collections.abc.Callable[[typing.SupportsFloat | typing.SupportsIndex, typing.SupportsFloat | typing.SupportsIndex], float], arg8: collections.abc.Callable[[typing.SupportsFloat | typing.SupportsIndex, typing.SupportsFloat | typing.SupportsIndex], float], arg9: collections.abc.Callable[[typing.SupportsFloat | typing.SupportsIndex, typing.SupportsFloat | typing.SupportsIndex], float], arg10: collections.abc.Callable[[typing.SupportsFloat | typing.SupportsIndex, typing.SupportsFloat | typing.SupportsIndex], float]) -> None:
        ...
    def assemble_matrix(self, K11: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], K22: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], K33: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], K12: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], K13: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], K23: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], D1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]) -> None:
        """
        Assemble the global stiffness matrix and load vector
        """
    def full_solve(self) -> list[tuple[str, float]]:
        """
        Run the full FEM solve process
        """
    def gen_b(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Generate b from a vector
        """
    def gen_c(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Generate c from a vector
        """
    def gen_delta_E(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsInt | typing.SupportsIndex, arg2: typing.SupportsInt | typing.SupportsIndex) -> float:
        """
        Generate delta_E
        """
    def gen_local_matrices(self) -> tuple[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]]:
        """
        Generate local matrices
        """
    def get_D(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Get the computed load vector D as a NumPy array
        """
    def get_Solution(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Get the computed solution vector as a NumPy array
        """
    def print_solution(self) -> None:
        """
        Print the computed solution
        """
    def reconstruct_solution(self) -> None:
        """
        Reconstruct the full solution vector
        """
    def solve_LGS(self) -> None:
        """
        Solve the linear system
        """
    def validate_sol(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], arg1: typing.SupportsFloat | typing.SupportsIndex) -> tuple[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"], list[float]]:
        """
        Validate the computed solution against an analytical solution
        """
    @property
    def RESOLUTION(self) -> float:
        ...
    @RESOLUTION.setter
    def RESOLUTION(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
