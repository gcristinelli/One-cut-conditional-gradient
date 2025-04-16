__author__ = "Giacomo Cristinelli, José A. Iglesias and Daniel Walter"
__date__ = "April 22nd, 2024"

from dolfin import *
import maxflow
import time
import argparse
import numpy as np
from matplotlib import pyplot as pp
from mshr import *
import scipy.sparse as cp

import utils
from plot import plot_result, plot_energy, plot_convergence
from mesh import _create_mesh
from SSN import _SSN
from insertion import _single_cut, _Dinkelbach


########################################################################################################################
def _main_():
    # ---make a timestamped folder to spam images
    rd = utils.result_directory()

    # FEniCS log level
    set_log_level(30)

    # ---TIMER
    start_time = time.time()

    with open('setup.conf', 'r') as conf_file:
        # Read the contents of the conf file
        conf_content = conf_file.read()

    # Open a text file for writing
    flog = open(rd + '/log.txt', 'w')
    flog.write(conf_content + '\n')

    # ---GEOMETRY LOOP ------------------------------------------------------------------------------------------------

    flog.write("\nGeometry loop:\n")
    print("Starting geometry loop...\n")

    # B1.---MESH GENERATION AND FUNCTION SPACES
    msh = _create_mesh(rdom, d, lx1, lx2, ly1, ly2, lz1, lz2, Nx, Ny, Nz, N, N2, rd)

    V = FunctionSpace(msh, 'DG', 0)  # PWC
    VL = FunctionSpace(msh, 'CG', 1)  # PWL
    VL_vec = VectorFunctionSpace(msh, 'CG', 1, d)  # vector valued PWL
    msh.init()
    msh.init(d - 1, d)
    e2f = msh.topology()(d - 1, d)
    vol_face_fn = Function(V)
    bdy_length_fn = Function(V)
    bdy_length = np.empty(0)
    bdy_faces = np.empty(0)
    facet_size = np.empty(0)

    if d == 2:
        flog.write("  Made a mesh with {} vertices, and {} faces \n".format(msh.num_vertices(), msh.num_faces()))
        plot(msh, linewidth=0.25)
        pp.savefig(rd + '/0_mesh.png', bbox_inches='tight', dpi=300)
        pp.close()
    elif d == 3:
        flog.write(
            "  Made a mesh with {} vertices, {} faces, and {} {}-dimensional cells \n".format(msh.num_vertices(),
                                                                                              msh.num_faces(),
                                                                                              msh.num_cells(), d))
    # dividing into boundary or internal facets
    facets_list = np.arange(msh.num_facets())
    bdy_facets = np.array([facet for facet in facets_list if len(e2f(facet)) == 1], dtype=int)
    internal_facets = np.setdiff1d(facets_list, bdy_facets)

    # Defining dimension dependent size of facets
    if d == 2:
        facet_size = np.array([Edge(msh, edge).length() for edge in facets_list])
    elif d == 3:
        facet_size = np.array([Face(msh, face).area() for face in facets_list])

    int_cells = np.array([[facet, e2f(facet)[0], e2f(facet)[1]] for facet in internal_facets], dtype=int)
    int_lengths = facet_size[internal_facets]
    vol_face_fn.vector()[:] = [Cell(msh, cell).volume() for cell in range(msh.num_cells())]
    mid_cell = [Cell(msh, cell).midpoint().array() for cell in range(0, msh.num_cells())]

    # defining integrating measures
    domains = MeshFunction("size_t", msh, msh.topology().dim(), 0)
    boundaries = MeshFunction("size_t", msh, msh.topology().dim() - 1, 0)
    dx = Measure("dx", domain=msh, subdomain_data=domains)
    ds = Measure("ds", domain=msh, subdomain_data=boundaries)
    nu = FacetNormal(msh)

    # B2.---GRAPH GENERATION, creating graph with (d-1)-facets areas/length as weights
    G = maxflow.GraphFloat()
    G.add_nodes(msh.num_cells())
    G.add_edges(int_cells[:, 1], int_cells[:, 2], facet_size[int_cells[:, 0]], facet_size[int_cells[:, 0]])

    flog.write("  Constructed graph has {} nodes, and {} edges \n".format(G.get_node_count(), G.get_edge_count()))
    flog.write("  Making the mesh and graph took - %.2f seconds \n" % (time.time() - start_time))

    print("Making the mesh of {} vertices, {} {}-dimensional cells, and the graph took - {} seconds \n".format(
        msh.num_vertices(), msh.num_cells(), d, time.time() - start_time))

    # ---PDE AND OPERATORS CONSTRUCTION--------------------------------------------------------------------------------
    v = TestFunction(VL)
    u = TrialFunction(VL)
    dt = T / time_steps

    if d == len(b):
        tr = interpolate(Constant(b), VL_vec)
    else:
        raise Exception("mismatch between transport term and dimension")

    if ell:
        Lap = a * inner(grad(u), grad(v)) * dx + inner(tr, grad(u)) * v * dx + c * v * u * dx
        adj_Lap = a * inner(grad(u), grad(v)) * dx - inner(tr, grad(u)) * v * dx + c * v * u * dx
    else:
        Lap = (v * u * dx + dt * a * inner(grad(u), grad(v)) * dx + dt * inner(tr, grad(u)) * v * dx
               + dt * c * v * u * dx)
        adj_Lap = (v * u * dx + dt * a * inner(grad(u), grad(v)) * dx - dt * inner(tr, grad(u)) * v * dx
                   + dt * c * v * u * dx)

    bdr = DirichletBC(VL, Constant(0), DomainBoundary())

    U0 = interpolate(Constant(1.0), V)
    Yd = Function(VL)
    UD = Function(V)
    Y0 = Function(VL)

    # C1.---REFERENCE CONTROL AND OBSERVATIONS FOR CASTLE
    u0 = U0 * v * dx
    if d == 2:
        Yd = interpolate(Expression('''x[0]>-0.5&&x[0]<0.5&&x[1]>-0.5&&x[1]<0.5? 1.00001:0.0''', degree=0), V)
        # diamond measurement
        # Yd = interpolate(Expression('''
        #         (x[0] >= -0.5 && x[0] <= 0.5 && x[1] >= -0.5 && x[1] <= 0.5 &&
        #          !(x[0] >= 0 && x[1] <= 0.5 && x[0] <= 0.5 && x[1] >= 0.5 - x[0])) ? 1.00001 : 0.0
        #     ''', degree=0), V)
    else:
        Yd = interpolate(
            Expression('''x[0]>-0.5&&x[0]<0.5&&x[1]>-0.5&&x[1]<0.5&&x[2]>-0.5&&x[2]<0.5? 1.00001:0.0''', degree=0), V)

    Yd = interpolate(Yd, VL)
    solve(Lap == u0, Y0, bdr, solver_parameters={'linear_solver': 'mumps'})

    # # C2.---REFERENCE CONTROL AND OBSERVATIONS FOR CIRCLES
    # shift = np.ones((msh.num_cells(), d))
    #
    # if d == 2:
    #     shift = np.hstack([shift, np.zeros((msh.num_cells(), 1))])
    #
    # UD = interpolate(Expression('''x[0]>-0.6&&x[0]<0&&x[1]>-0.6&&x[1]<-0.1? 1.00001:0.0''', degree=0), V)
    # UD.vector()[:] += 2 * (np.linalg.norm(mid_cell - 0.5 * shift, axis=1) < 0.15)
    #
    # energy_D = utils.TV(int_lengths, int_cells, bdy_length, bdy_faces, UD)
    #
    # flog.write("Energy of the toy control is %.8e\n" % energy_D)
    #
    # rhyd = UD * v * dx
    # rhy0 = U0 * v * dx
    #
    # print("Computing the initial state and measurement... \n")
    # if ell:
    #     solve(Lap == rhyd, Yd, bdr, solver_parameters={'linear_solver': 'mumps'})
    #     solve(Lap == rhy0, Y0, bdr, solver_parameters={'linear_solver': 'mumps'})
    # else:
    #     for n in range(time_steps):
    #         solve(Lap == rhyd, Yd, bdr, solver_parameters={'linear_solver': 'mumps'})
    #         solve(Lap == rhy0, Y0, bdr, solver_parameters={'linear_solver': 'mumps'})
    #         rhyd = Yd * v * dx
    #         rhy0 = Y0 * v * dx

    # C2.---PLOTTING INITIAL OBJECTS
    plot_result(msh, int_cells, flog, rd, Yd, 0, "d", d)
    plot_result(msh, int_cells, flog, rd, UD, 2, "d", d)
    plot_result(msh, int_cells, flog, rd, U0, 2, 0, d)

    # C3.---OBJECTS FOR SEMI-SMOOTH NEWTON
    Y0_arr = Y0.vector().get_local()
    Km = np.reshape(Y0_arr, (-1, 1))
    Kl = np.empty(shape=[len(Km), 0])
    mean = np.array([1])
    coefficients = np.empty(0)
    U0 = interpolate(U0, V)
    U0_arr = U0.vector().get_local()
    Ul = np.empty(shape=[len(U0_arr), 0])
    measurements = Yd.vector().get_local()
    mass_form = v * u * dx
    M = utils.assemble_csr(mass_form)

    # C4.---WARM UP ITERATION
    coefficients, mean, opt_val, misfit = _SSN(Kl, Km, coefficients, mean, measurements, alpha, M, flog)

    # --SINGLE CUT CONDITIONAL GRADIENT-------------------------------------------------------------------------------
    k = 0
    Uk = Function(V)
    prev_Uk = Function(V)
    Yk = Function(VL)
    Vk = Function(VL)
    Uk.vector()[:] = mean.flatten() * U0_arr + Ul @ coefficients
    Yk.vector()[:] = mean.flatten() * Y0_arr + Kl @ coefficients

    opt = np.zeros(max_iterations + 1)
    energy = np.zeros(max_iterations + 1)
    rel_change = np.zeros(max_iterations + 1)
    data = []
    total_time = 0

    energy[0] = (1 / alpha) * 0.5 * assemble((Y0 - Yd) ** 2 * dx)
    export = [0, total_time, energy[0], opt_val / alpha, 0, 0]
    data.append(export)

    # D2.---MAIN LOOP
    while (k == 0) or (k <= max_iterations and opt[k - 1] > tolerance):
        start_iteration_time = time.time()
        flog.write("\nIteration {} of conditional gradient: \n".format(k))
        print("Starting iteration %s of OCCG\n" % k)

        prev_Uk.assign(Uk)

        # Solve adjoint equation
        Pk = Function(VL)
        misfit_fun = Yk - Yd
        rhpk = misfit_fun * v * dx

        prev_time = time.time()

        if ell:
            solve(adj_Lap == rhpk, Pk, bdr, solver_parameters={'linear_solver': 'mumps'})
        else:
            for n in range(time_steps):
                solve(adj_Lap == rhpk, Pk, bdr, solver_parameters={'linear_solver': 'mumps'})
                rhpk = Pk * v * dx

        flog.write("  Adjoint PDE took %.2f seconds \n" % (time.time() - prev_time))

        # Making sure adjoint pk has zero average
        Pkp = interpolate(Pk, V)

        # Insertion step
        vm, vmm, opt[k], perm = _single_cut(msh, vol_face_fn, bdy_length_fn, int_lengths, int_cells, bdy_length,
                                            bdy_faces, G, Pkp, 1, alpha, flog, k, tolerance=tolerance)
        if np.abs(perm) == 0:
            data_array = np.array(data, dtype=float)
            fmt = ['%d', '%.2f', '%.10f', '%.6e', '%.6e']
            header = 'iteration, time (seconds), energy, indicator, L1-relative change'
            np.savetxt(rd + '/output.csv', data_array, delimiter=',', fmt=fmt, header=header)
            raise Exception("Zero cuts at iteration %s" % k)

        # solving a new state equation on each decomposable component of the new cut
        nonzero_vals = np.unique(vmm.vector()[vmm.vector().get_local() != 0])
        stacked_masks = np.array([(vmm.vector().get_local() == n).astype(int) for n in nonzero_vals])
        comp = len(stacked_masks)
        for i, mask in enumerate(stacked_masks):
            vk = Function(V)
            vk.vector()[:] = mask
            # plot_result(msh, int_cells, flog, rd, vk, 1, '%s,%i' %(k,i), d)
            pervk = utils.TV(int_lengths, int_cells, bdy_length, bdy_faces, vk)
            rhvk = (vk / pervk) * v * dx
            Ul = np.append(Ul, np.reshape(vk.vector().get_local() / pervk, (-1, 1)), axis=1)

            # New state equation
            prev_time = time.time()

            if ell:
                solve(Lap == rhvk, Vk, bdr, solver_parameters={'linear_solver': 'mumps'})
            else:
                for n in range(time_steps):
                    solve(Lap == rhvk, Vk, bdr, solver_parameters={'linear_solver': 'mumps'})
                    rhvk = Vk * v * dx

            flog.write("  state PDE component %i took %.2f seconds\n" % (i, time.time() - prev_time))

            # Storing new state and updating the coefficients
            Kl = np.append(Kl, np.reshape(Vk.vector().get_local(), (-1, 1)), axis=1)
            coefficients = np.append(coefficients, np.array([1]))

        coefficients, mean, opt_val, misfit = _SSN(Kl, Km, coefficients, mean, measurements, alpha, M, flog)

        # Resulting values
        Ul, Kl, coefficients = utils.sparsify(Ul, Kl, coefficients)

        Uk.vector()[:] = mean.flatten() * U0_arr + Ul @ coefficients
        Yk.vector()[:] = mean.flatten() * Y0_arr + Kl @ coefficients

        energy[k] = (1 / alpha) * misfit + utils.TV(int_lengths, int_cells, bdy_length, bdy_faces, Uk)
        rel_change[k] = assemble(abs(Uk - prev_Uk) * dx) / assemble(abs(Uk) * dx)

        flog.write("  Current surrogate energy value is %.6e, with convergence indicator, %.6e \n" % (opt_val/alpha, opt[k]))
        flog.write("  Current actual energy value is %.6e \n" % (energy[k]))
        print(
            "Step %s of OCCG finished with energy value %.10f and convergence indicator %.6e \n" % (k, opt_val, opt[k]))

        plot_convergence(rd, opt)
        plot_energy(rd, energy)
        if plts:
            plot_result(msh, int_cells, flog, rd, vm, 1, k, d)
            plot_result(msh, int_cells, flog, rd, Uk, 2, k + 1, d)
            plot_result(msh, int_cells, flog, rd, Pk, 3, k, d)
            plot_result(msh, int_cells, flog, rd, Yk, 4, k + 1, d)
            plot_result(msh, int_cells, flog, rd, misfit_fun, 5, k + 1, d)

        total_time += time.time() - start_iteration_time
        export = [k + 1, total_time, energy[k], opt[k], rel_change[k], comp]
        data.append(export)
        k += 1

    data_array = np.array(data, dtype=float)
    energy_change = data_array[:, 2] - data_array[-1, 2]
    data_array = np.hstack((data_array, energy_change[:, np.newaxis]))
    fmt = ['%d', '%.2f', '%.10f', '%.10e', '%.10e', '%d', '%.10e']
    header = 'iteration, time (seconds), energy, indicator, L1-relative change, components, energy difference'
    np.savetxt(rd + '/output.csv', data_array, delimiter=',', fmt=fmt, header=header)

    print("Algorithm converged in %s steps, the total time was %.2f seconds" % (k - 1, time.time() - start_time))
    flog.write("The total time was %.2f seconds" % (time.time() - start_time))
    flog.close()


########################################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to configuration file', default='./setup.conf')
    args = parser.parse_args()

    # Read the configuration file
    (rdom, Nx, Ny, Nz, N, N2, d, alpha, lx1, lx2, ly1, ly2, lz1, lz2, cx, cy, r, ell, a, b, c, time_steps, T,
     max_iterations, tolerance, plts) = (utils.parse_config(args.config))

    _main_()
