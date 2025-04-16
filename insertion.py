from dolfin import *
import maxflow
import networkx as nx
import numpy as np
import time
import utils
import csv


def _single_cut(msh, vol_face_fn, bdy_length_fn, int_lengths, int_cells, bdy_lengths, bdy_faces, graph, dual, sign,
                alpha, flog, j, tolerance=1e-10):
    V = FunctionSpace(msh, 'DG', 0)  # PWC

    # initializing coefficients
    coeff = sign * (1 / alpha)

    cut_result = Function(V)
    prep_time = time.time()
    energy, integral, per, labels = _linear_problem(msh, vol_face_fn, bdy_length_fn, int_lengths, int_cells,
                                            bdy_lengths, bdy_faces, graph, dual, coeff, cut_result)
    value = - per - coeff * integral
    flog.write("    The cut and decomposition took %.2f seconds, has value - %.10f\n" % (
        (time.time() - prep_time), value))

    # decomposing the cut
    decomposed_cut = Function(V)
    unique_vals, inverse = np.unique(labels * cut_result.vector(), return_inverse=True)
    decomposed_cut.vector()[:] = inverse

    return cut_result, decomposed_cut, value, per


def _linear_problem(msh, vol_face_fn, bdy_length_fn, int_lengths, int_cells,
                    bdy_lengths, bdy_faces, graph, dual, coeff, cut_result):
    copy_graph = graph.copy()

    dual_vec = dual.vector()
    val = coeff * dual_vec * vol_face_fn.vector() + bdy_length_fn.vector()
    pos_val = np.maximum(0.0, val)
    neg_val = np.maximum(0.0, -val)

    copy_graph.add_grid_tedges(np.arange(msh.num_cells()), pos_val, neg_val)

    energy = copy_graph.maxflow()
    cut_values = copy_graph.get_grid_segments(np.arange(msh.num_cells())).astype(float)
    cut_result.vector()[:] = cut_values

    # Mask of interior cells where both ends are in the same component (not cut)
    not_cut = (cut_values[int_cells[:, 1]] * cut_values[int_cells[:, 2]] == 1)
    same_component = int_cells[not_cut, 1:]

    # Get connected components and node labels
    components, node_labels = get_components(copy_graph, same_component)

    # Energy terms
    integral = assemble(dual * cut_result * dx)
    per = utils.TV(int_lengths, int_cells, bdy_lengths, bdy_faces, cut_result)

    return energy, integral, per, node_labels


def get_components(graph, same_component):
    nx_graph = graph.get_nx_graph()
    nx_graph.remove_nodes_from(['s', 't'])

    # Add symmetric edges at once
    all_edges = np.vstack([same_component, same_component[:, ::-1]])
    nx_graph.add_edges_from(map(tuple, all_edges), weight=1)

    # Strongly connected components
    sccs = list(nx.strongly_connected_components(nx_graph))
    components = [np.fromiter(c, dtype=int) for c in sccs]

    # Label assignment
    node_labels = np.empty(nx_graph.number_of_nodes(), dtype=int)
    for label, nodes in enumerate(components):
        node_labels[list(nodes)] = label

    return components, node_labels


def _Dinkelbach(msh, vol_face_fn, bdy_length_fn, int_lengths, int_cells, bdy_lengths, bdy_faces, graph, dual, sign,
                alpha, flog, j, max_iterations=15, tolerance=1e-10):
    V = FunctionSpace(msh, 'DG', 0)  # PWC
    coeff = np.zeros(max_iterations)
    per = np.zeros(max_iterations)
    integral_value = np.zeros(max_iterations)
    energy = np.zeros(max_iterations)

    # initializing coefficients
    flog.write("  Dinkelbach:\n")
    coeff[0] = sign * (1 / alpha)
    n = 0

    # ---initial cut
    cut_result = Function(V)
    decomposed_cut = Function(V)
    prep_time = time.time()
    energy[n], integral_value[n], per[n], labels = _linear_problem(msh, vol_face_fn, bdy_length_fn, int_lengths, int_cells,
                                                           bdy_lengths, bdy_faces, graph, dual, coeff[n], cut_result)
    flog.write("    The initial cut took %.2f seconds, has value - %.10f\n" % (
        (time.time() - prep_time), per[n] + coeff[n] * integral_value[n]))

    # if it is a zero cut, stop
    if per[n] < tolerance:
        flog.write("    Zero initial cut with coefficient %s \n" % (coeff[0]))
        ext = 0
        return cut_result, cut_result, ext, per[n]
    else:
        flog.write("    The perimeter of the initial cut is %s\n" % per[n])
        n += 1

    # ---Following cuts (main loop)
    while (n == 1) or (
            1 < n < max_iterations and np.abs(per[n - 1] + coeff[n - 1] * integral_value[n - 1]) > tolerance):
        prev_time = time.time()
        coeff[n] = (-per[n - 1] / integral_value[n - 1])
        oldcut = cut_result
        oldlabels = labels
        cut_result = Function(V)
        energy[n], integral_value[n], per[n], labels = _linear_problem(msh, vol_face_fn, bdy_length_fn, int_lengths, int_cells,
                                                               bdy_lengths, bdy_faces, graph, dual, coeff[n],
                                                               cut_result)
        flog.write("    Cut number %s has lambda equal to %s, took %.2f seconds, has value %.10f\n" % (
            n, coeff[n], (time.time() - prev_time), per[n] + coeff[n] * integral_value[n]))
        if per[n] < tolerance:
            flog.write("    Dinkelbach stops after %s steps with lambda %.6e \n" % (n, coeff[n]))
            print("Dinkelbach stops after %s steps with lambda %.6e \n" % (n, coeff[n]))
            ext = -1 / coeff[n]
            decomposed_cut.vector()[:] = oldlabels * oldcut.vector()
            return oldcut, decomposed_cut, ext, per[n - 1]
        flog.write("    Its perimeter is %s\n" % per[n])
        # increment iterations
        n += 1

    flog.write("    Dinkelbach stops after %s steps with lambda %.6e \n" % (n, coeff[n - 1]))
    print("Dinkelbach stops after %s steps with lambda %.6e \n" % (n, coeff[n - 1]))
    ext = -1 / coeff[n - 1]

    # decomposing the cut
    decomposed_cut.vector()[:] = labels * cut_result.vector()

    return cut_result, decomposed_cut, ext, per[n - 1]
