import os
from datetime import datetime
import numpy as np
import configparser
from dolfin import *
import scipy.sparse as cp


def result_directory():
    now = datetime.now()
    dt_string = now.strftime("%y%m%d_%H%M%S")
    rd = os.path.join(os.path.dirname(__file__), './results/' + dt_string)
    if os.path.isdir(rd):
        pass
    else:
        os.makedirs(rd)
    return rd


def parse_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)

    mesh_section = config['Mesh']
    rdom = eval(mesh_section.get('Random_mesh', fallback='True'))
    Nx, Ny, Nz = eval(mesh_section.get('Size', fallback='[200,200,200]'))
    N = eval(mesh_section.get('N2D', fallback='100'))
    N2 = eval(mesh_section.get('N3D', fallback='50'))
    d = eval(mesh_section.get('dimension', fallback='2'))

    regularizer_section = config['Regularizer']
    alpha = eval(regularizer_section.get('Parameter', fallback='0.0001'))

    domain_section = config['Regularizer']
    lx1, lx2 = eval(domain_section.get('Domain_x', fallback='[-1,1]'))
    ly1, ly2 = eval(domain_section.get('Domain_y', fallback='[-1,1]'))
    lz1, lz2 = eval(domain_section.get('Domain_z', fallback='[-1,1]'))

    control_section = config['Control']
    cx, cy = eval(control_section.get('Center', fallback='[0,0]'))
    r = eval(control_section.get('radius', fallback='0.2'))

    pde_section = config['PDE']
    ell = eval(pde_section.get('elliptic', fallback='True'))
    a = eval(pde_section.get('a', fallback='1'))
    b = eval(pde_section.get('b', fallback='1'))
    c = eval(pde_section.get('c', fallback='0.5'))
    time_steps = eval(pde_section.get('time-steps', fallback='100'))
    T = eval(pde_section.get('T', fallback='1'))

    gcg_section = config['GCG']
    max_iterations = eval(gcg_section.get('max_iterations', fallback='120'))
    tolerance = eval(gcg_section.get('tolerance', fallback='1e-10'))

    plot_section = config['Plots']
    plts = eval(plot_section.get('return_plots', fallback='False'))

    return (rdom, Nx, Ny, Nz, N, N2, d, alpha, lx1, lx2, ly1, ly2, lz1, lz2, cx, cy, r, ell, a, b, c, time_steps, T,
            max_iterations, tolerance, plts)


def TV(int_lengths, int_cells, bdy_lengths, bdy_faces, cut_result):
    var1 = cut_result.vector()[int_cells[:, 1]] - cut_result.vector()[int_cells[:, 2]]
    TV = sum(int_lengths * np.abs(var1)) + sum(bdy_lengths * np.abs(cut_result.vector()[bdy_faces]))
    return TV


def sparsify(U, K, c):
    ind = np.where(c <= 0.0000001)[0]
    U = np.delete(U, ind, 1)
    K = np.delete(K, ind, 1)
    c = np.delete(c, ind, 0)
    return U, K, c


def assemble_csr(a, bc=None):
    A = assemble(a)
    if bc is not None:
        bc.apply(A)
    mat = as_backend_type(A).mat()
    rows, cols, vals = mat.getValuesCSR()
    return cp.csr_matrix((vals, cols, rows))