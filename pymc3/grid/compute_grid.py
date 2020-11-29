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
from pandas import Series

from ..model import modelcontext, DictToArrayBijection, ArrayOrdering


def compute_grid(
        intervals,
        model=None,
):
    model = modelcontext(model)

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
    def __init__(self, coords, lly, ):
        self.coords = coords
        self.lly = lly
        self.ly = np.exp(lly - np.max(lly))
        self.var_dims = {var: dim for dim, var in enumerate(coords.keys())}
        self.dims = tuple(self.var_dims.values())

    def marginal(self, var):
        var_dim = self.var_dims[var]
        marginal_dims = tuple((dim for dim in self.dims if dim != var_dim))
        if len(marginal_dims) == 1:
            marginal_dims = marginal_dims[0]
        print(marginal_dims)
        marginal_ly = np.sum(self.ly, axis=marginal_dims)

        return Series(index=self.coords[var],
                      data=marginal_ly / np.sum(marginal_ly),
                      name='PMF')

    def joint(self, variables):
        # TODO: Return MultiIndex PMF
        # TODO: Merge with marginal
        var_dims = [self.var_dims[var] for var in variables]
        marginal_dims = [dim for dim in self.dims if dim not in var_dims]

        if len(marginal_dims) == 0:
            return self.ly

        if len(marginal_dims) == 1:
            marginal_dims = marginal_dims[0]

        marginal_ly = np.sum(self.ly, axis=marginal_dims)
        return marginal_ly

    def sample(self, draws=2000, jitter=False):
        # Return SimpleTrace object
        pass

    def __getitem__(self, key):
        return self.marginal(key)
