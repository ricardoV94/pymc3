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
from pprint import pformat

import numpy as np
import theano
import theano.tensor as tt
from theano.graph.toolbox import is_same_graph


def find_infinite_bounds(apply, implicit_bounds=False, bound_nodes=None):
    # Create output list at first lever of iteration
    if bound_nodes is None:
        bound_nodes = []

    # Check if it is a switch node
    # print(apply.op.scalar_op)
    if hasattr(apply.op, "scalar_op") and isinstance(apply.op.scalar_op, theano.scalar.basic.Switch):
        switch_input, *switch_outputs = apply.inputs  # 1 input, 2 outputs

        # Check if it is an explicit bound switch (named)
        if apply.out.name == 'bound_switch':
            bound_nodes.append((switch_input, switch_outputs, 'explicit_bound'))
            # Terminate graph trasversal here
            if not implicit_bounds:
                return bound_nodes
        # Check if it is an implicit bound switch (leads to -inf)
        elif implicit_bounds:
            if switch_outputs[0].eval() == -np.inf:
                bound_nodes.append((switch_input, switch_outputs, 'implicit_bound_true'))
            if switch_outputs[1].eval() == -np.inf:
                bound_nodes.append((switch_input, switch_outputs, 'explicit_bound_false'))

        # Continue to explore downstream of the switch
        child_apply = switch_input.owner
        if child_apply:
            find_infinite_bounds(child_apply, implicit_bounds, bound_nodes)

    else:
        for apply_input in apply.inputs:
            # Check if it is not a terminal node
            child_apply = apply_input.owner
            if child_apply:
                find_infinite_bounds(child_apply, implicit_bounds, bound_nodes)

    return bound_nodes


def find_infinite_bound_logical_conds(apply, logical_conds=None):
    if logical_conds is None:
        logical_conds = []

    if hasattr(apply.op, "scalar_op") and isinstance(apply.op.scalar_op, theano.scalar.basic.LogicalComparison):
        logical_conds.append(apply)

    for child_node in apply.inputs:
        child_apply = child_node.owner
        if child_apply:
            find_infinite_bound_logical_conds(child_apply, logical_conds)

    return logical_conds


def find_logical_cond_input_variables(logical_cond_apply):

    inputs = logical_cond_apply.inputs
    if len(inputs) > 2:
        raise ValueError

    input_variables = []
    for expression in inputs:
        if not expression.owner:
            input_variables.append((expression,))
        else:
            expression_inputs = tuple(theano.graph.basic.graph_inputs(expression.owner.inputs))

            # TODO: Check for 1.0 * var expression introduced by the bound_switch
            # and remove Constant(1) from output
            found_mul_1 = False
            for potential_var in expression_inputs:
                test_expression = 1.0 * potential_var
                if is_same_graph(expression, test_expression):
                    input_variables.append((potential_var,))
                    found_mul_1 = True
            # Otherwise, include all inputs
            if not found_mul_1:
                input_variables.append(expression_inputs)

    return input_variables


# TODO: Less hackish way to mask culprit values
def input_variables_to_string(x, mask):
    def get_masked_values(x):
        return np.squeeze(np.unique(np.resize(x, mask.shape)[mask]))

    def get_value_if_const(x):
        if hasattr(x, 'value'):
            return get_masked_values(x.value)
        if hasattr(x, 'tag'):
            return f"{x} = {get_masked_values(x.tag.test_value)}"
        return x

    if len(x) == 1:
        return str(get_value_if_const(x[0]))
    else:
        return f"f({', '.join(map(str, map(get_value_if_const, x)))})"


theano_logical_comparators_parse_map = {
    theano.scalar.basic.EQ: '==',
    theano.scalar.basic.NEQ: '!=',
    theano.scalar.basic.GT: '>',
    theano.scalar.basic.GE: '>=',
    theano.scalar.basic.LT: '<',
    theano.scalar.basic.LE: '<=',
}

# TODO: Flip logic for implicit bounds with expression_false
# TODO: Print only non-repeated bad values (and trim output)
# TODO: Get parameter / observed names (is this possible)?
# TODO: Add logic for BinaryBitOps ?
def debug_bounds(model, variable, implicit_bounds=False):
    bound_switches = find_infinite_bounds(variable.logpt.owner, implicit_bounds)
    # Test bound
    for bound_switch_input, bound_switch_outputs, bound_description in bound_switches:

        # Check that bound input evaluates to False (otherwise it cannot be responsible)
        bound_fn = model.fn(bound_switch_input)
        bound_output = bound_fn(model.test_point)
        # print(bound_switch, bound_output)
        if np.all(bound_output):
            # If it's an explicit bound_switch, confirm that the nested logp expression is non-finite
            if bound_description == 'explicit_bound':
                logp_branch = bound_switch_outputs[0]
                logp_fn = model.fn(logp_branch)
                logp = logp_fn(model.test_point)
                if not np.all(np.isfinite(logp)):
                    print(
                        f'The logp expression of {variable} is non-finite for the given inputs,',
                        'but no explicit bounds were found to have been violated.'
                     )
            continue

        # If bound is responsible, test which nested logical conditions are responsible
        bound_logical_conds = find_infinite_bound_logical_conds(bound_switch_input.owner)
        first_bound = True

        # 1. Sanity check that disabling all logical conditions leads the switch to be true
        # TODO: Remove?
        no_bound = theano.clone(
            bound_switch_input,
            {
                logical_cond.out: tt.eq(logical_cond.inputs[0], logical_cond.inputs[0])
                for logical_cond in bound_logical_conds
             }
        )
        no_bound_fn = model.fn(no_bound)
        no_bound_output = no_bound_fn(model.test_point)
        if not np.all(no_bound_output):
            raise RuntimeWarning('Disabling all bounds is not working as expected')
            continue

        # 2. Enable one switch at a time (culprit if switch stays false)
        for enabled_logical_cond in bound_logical_conds:
            # TODO: Is there a more clean way to achieve this test?
            new_bound = theano.clone(
                bound_switch_input,
                {
                    logical_cond.out: tt.eq(logical_cond.inputs[0], logical_cond.inputs[0])
                    for logical_cond in bound_logical_conds if logical_cond != enabled_logical_cond
                }
            )
            new_bound_fn = model.fn(new_bound)
            new_bound_output = new_bound_fn(model.test_point)
            # print('Enabled: ', enabled_logical_cond, ' | Result :', new_bound_output)
            if not np.all(new_bound_output):
                if first_bound:
                    first_bound = False
                    explicit = 'explicit' if bound_description == 'explicit_bound' else 'implicit'
                    print(f'The following {explicit} bound(s) of {variable} were violated:')

                # print(enabled_logical_cond.out)
                mask = new_bound_output == 0
                # print(new_bound_output, mask)
                ivs = find_logical_cond_input_variables(enabled_logical_cond)
                ivs = [input_variables_to_string(iv, mask) for iv in ivs]
                logical_cond_type = type(enabled_logical_cond.op.scalar_op)
                logical_comp = theano_logical_comparators_parse_map.get(
                    logical_cond_type,
                    str(logical_cond_type),
                )
                print(f'{ivs[0]} {logical_comp} {ivs[1]}')


def debug_bad_energy(model, implicit_bounds=False):
    test_point = model.test_point
    for variable in model.basic_RVs:
        if not(np.isfinite(variable.logp(test_point))):
            debug_bounds(model, variable, implicit_bounds)
    print('Done')

