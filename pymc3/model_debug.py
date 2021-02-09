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
        # Check if it is an explicit bound switch (named)
        switch_input = apply.inputs[0]
        if apply.out.name == 'bound_switch':
            bound_nodes.append((switch_input, 'bound'))
            # Terminate graph trasversal here
            if not implicit_bounds:
                return bound_nodes
        # Check if it is an implicit bound switch (leads to -inf)
        elif implicit_bounds:
            switch_outputs = apply.inputs[1:]
            if switch_outputs[0].eval() == -np.inf:
                bound_nodes.append((switch_input, 'expression_true'))
            # elif check_leaf_infinite_constant(switch_outputs[1]):
            elif switch_outputs[1].eval() == -np.inf:
                bound_nodes.append((switch_input, 'expression_false'))

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

    # TODO: include unary logic operators?
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
            # TODO: This may fail? Have a fallback
            expression_inputs = tuple(theano.graph.basic.graph_inputs(expression.owner.inputs))

            # Check for 1.0 * var expression introduced by the bound_switch
            for potential_var in expression_inputs:
                test_expression = 1.0 * potential_var
                if is_same_graph(expression, test_expression):
                    input_variables.append((potential_var,))
                    break
            else:  # nobreak
                input_variables.append(expression_inputs)

    return input_variables


def input_variables_to_string(x):
    def get_value_if_const(x):
        if hasattr(x, 'value'):
            return x.value
        if hasattr(x, 'tag'):
            return f"{x}: [{x.tag.test_value}]"
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


def debug_bounds(model, variable, implicit_bounds=False):
    bound_switches = find_infinite_bounds(variable.logpt.owner, implicit_bounds)
    # Test bound
    for bound_switch, bound_description in bound_switches:
        # Check if bound evaluates to False (TODO: Or True for implicit bounds with expression_false)
        bound_fn = model.fn(bound_switch)
        bound_output = bound_fn(model.test_point)
        print(bound_switch, bound_output)
        if np.all(bound_output):
            continue

        bound_logical_conds = find_infinite_bound_logical_conds(bound_switch.owner)
        first_bound = True
        # Test bound logical conditions

        # 1. Sanity check that disabling all logical conditions leads the switch to be true
        no_bound = theano.clone(
            bound_switch,
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
                bound_switch,
                {
                    logical_cond.out: tt.eq(logical_cond.inputs[0], logical_cond.inputs[0])
                    for logical_cond in bound_logical_conds if logical_cond != enabled_logical_cond
                }
            )
            new_bound_fn = model.fn(new_bound)
            new_bound_output = new_bound_fn(model.test_point)
            print('Enabled: ', enabled_logical_cond, ' | Result :', new_bound_output)
            if not np.all(new_bound_output):
                if first_bound:
                    first_bound = False
                    explicit = 'explicit' if bound_description == 'bound' else 'implicit'
                    print(f'The following {explicit} bound(s) of {variable} were violated:')
                    print('')

                # print(enabled_logical_cond.out)
                ivs = find_logical_cond_input_variables(enabled_logical_cond)
                ivs = [input_variables_to_string(iv) for iv in ivs]
                logical_comp = theano_logical_comparators_parse_map[(type(enabled_logical_cond.op.scalar_op))]
                print(f'{ivs[0]} {logical_comp} {ivs[1]}')
                print('')


def debug_bad_energy(model, implicit_bounds=False):
    test_point = model.test_point
    for variable in model.basic_RVs:
        if not(np.isfinite(variable.logp(test_point))):
            debug_bounds(model, variable, implicit_bounds)
    print('Done')
