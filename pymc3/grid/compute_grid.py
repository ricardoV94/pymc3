#   Copyright 2020 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np
from pandas import Series, DataFrame, MultiIndex

from ..model import modelcontext, DictToArrayBijection, ArrayOrdering


def compute_grid(
        intervals,
        model=None,
):
    """

    Parameters
    ----------
    intervals :
    model :

    Returns
    -------
    pymc3.Grid object
    """

    if model is None:
        model = modelcontext(model)

    for variable, value in intervals.items():
        try:
            value = iter(value)
        except TypeError:
            intervals[variable] = [value]

    grid_vars = (model[v] for v in intervals.keys())
    bij = DictToArrayBijection(ArrayOrdering(grid_vars), model.test_point)
    logp_func = bij.mapf(model.fastlogp_nojac)

    grid = np.meshgrid(*intervals.values(), indexing='ij')
    grid_shape = grid[0].shape
    lly = np.zeros(grid[0].size)
    grid = [a.flatten() for a in grid]

    for i, variables in enumerate(zip(*grid)):
        lly[i] = logp_func(variables)

    # grid = {variable: values.reshape(grid_shape) for variable, values in zip(intervals.keys(), grid)}
    lly = lly.reshape(grid_shape)
    return Grid(intervals, lly)


class Grid:
    def __init__(self, coords, lly,):
        self.coords = coords
        self.lly = lly
        self.ly = np.exp(lly - np.max(lly))
        self.var_dims = {var: dim for dim, var in enumerate(coords.keys())}
        self.dims = tuple(self.var_dims.values())

    def marginal(self, variables):
        """ Return marginal probability for the selected variable(s).

        Note: Order of varibles is ignored. Output respects the order of the
        variables with which the grid object was created.

        Parameters
        ----------
        variables: str or list of str
            Name of variable(s) whose marginal probability to compute.

        Returns
        -------
            pd.Series or pd.DataFrame containing the the value of each variable
            as index (or MultiIndex) and a single column "PMF" with the respective
            probability.
        """

        if isinstance(variables, str):
            variables = [variables]
        else:
            variables = list(sorted(variables, key=lambda v: self.var_dims[v]))

        var_dims = [self.var_dims[var] for var in variables]
        marginal_dims = tuple((dim for dim in self.dims if dim not in var_dims))

        marginal_ly = np.sum(self.ly, axis=marginal_dims)
        marginal_ly /= np.sum(marginal_ly)

        if marginal_ly.ndim == 0:
            return Series(data=marginal_ly, name='PMF')
        elif marginal_ly.ndim == 1:
            return Series(index=self.coords[variables[0]], data=marginal_ly, name='PMF')
        else:
            var_coords = [self.coords[var] for var in variables]
            index = MultiIndex.from_product(var_coords, names=variables)
            return DataFrame(data=marginal_ly.flatten(), index=index, columns=['PMF'])

    def sample(self, draws=2000, jitter=False):
        # Return SimpleTrace object
        pass

    def __getitem__(self, key):
        return self.marginal(key)
