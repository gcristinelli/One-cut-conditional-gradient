from dolfin import *
import numpy as np
import time
from matplotlib import pyplot as pp


def plot_result(msh, int_cells, flog, rd, f, n, j, d):
    if d == 2:
        ax = plot(f)
        pp.colorbar(ax, shrink=0.7, format="%01.2f", pad=0.02)
        pp.axis('off')
        # pp.colorbar(ax, shrink=0.55, format='%01.3f')
        if n == 0: pp.savefig(rd + '/input_Yo.png', bbox_inches='tight', dpi=600)
        if n == 1: pp.savefig(rd + '/cut_%s.png' % j, bbox_inches='tight', dpi=600)
        if n == 2: pp.savefig(rd + '/U_%s.png' % j, bbox_inches='tight', dpi=600)
        if n == 3: pp.savefig(rd + '/P_%s.png' % j, bbox_inches='tight', dpi=600)
        if n == 4: pp.savefig(rd + '/Y_%s.png' % j, bbox_inches='tight', dpi=600)
        if n == 5: pp.savefig(rd + '/misfit_%s.png' % j, bbox_inches='tight', dpi=600)
        pp.close()
    elif n == 1 or n == 2:
        _export_ply(msh, int_cells, f, j, n, 1 / 255, rd, flog)


def _export_ply(msh, int_cells, fun, j, index, threshold, rd, flog):
    start_time = time.time()
    msh.init(2, 0)
    f2v = msh.topology()(2, 0)
    var = np.abs(fun.vector()[int_cells[:, 2]] - fun.vector()[int_cells[:, 1]])
    var_index = np.column_stack((int_cells[:, 0], var))
    non_zero_var = var_index[var_index[:, 1] > threshold]
    if len(non_zero_var) == 0:
        return
    normalized_var = (255 * (non_zero_var[:, 1] - np.min(non_zero_var[:, 1])) / (np.max(non_zero_var[:, 1]))).astype(
        int)
    n = len(non_zero_var[:, 0])
    header = "ply\nformat ascii 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nelement " \
             "face {}\nproperty list uchar int vertex_indices\nproperty uchar red \nproperty uchar green \nproperty " \
             "uchar blue \n end_header\n"
    if index == 1:
        f = open(rd + '/cut_%s.ply' % j, "w")
        f.write(header.format(3 * n, n))
    elif index == 2:
        f = open(rd + '/U_%s.ply' % j, "w")
        flog.write("  The maximum variation of U%s between cells is %.6e \n" % (j, np.max(non_zero_var)))
        f.write(header.format(3 * n, n))

    triangles = np.arange(3 * n, dtype=int)
    vertices_per_face = 3 * np.ones((n, 1), dtype=int)
    triangles = np.concatenate((vertices_per_face, triangles.reshape((-1, 3))), axis=1)

    for face_index in range(n):
        vert = f2v(int(non_zero_var[face_index, 0]))
        for vertex in vert:
            f.write(" ".join(str(coord) for coord in msh.coordinates()[vertex]) + "\n")

    for face_index in range(n):
        f.write(" ".join(str(ind) for ind in triangles[face_index, :]) + " " + " ".join(
            str(normalized_var[face_index]) for ind in range(3)) + "\n")

    f.close()
    flog.write("  Creating the ply file took %.2f seconds \n" % (time.time() - start_time))


def plot_convergence(rd, opt):
    # Trim and prepare data
    opt_trimmed = np.trim_zeros(opt)
    iterations = np.arange(1, len(opt_trimmed) + 1)
    log_opt = np.log10(opt_trimmed)

    # Set figure style
    #pp.style.use('seaborn-v0_8-darkgrid')
    pp.figure(figsize=(6, 4))

    # Plot
    with pp.style.context('seaborn-v0_8-darkgrid'):
        pp.plot(iterations, log_opt, marker='o', linewidth=1, color='navy', label=r'$\log_{10}$(indicator)')

    # Labels and grid
    pp.xlabel("Iteration", fontsize=12)
    pp.ylabel(r"$\log_{10}$ Convergence Indicator", fontsize=12)
    pp.title("Convergence Behavior", fontsize=14)
    pp.grid(True, which='both', linestyle='--', linewidth=0.5)
    pp.xticks(fontsize=10)
    pp.yticks(fontsize=10)
    pp.legend(fontsize=10)
    pp.tight_layout()
    pp.savefig(rd + '/indicator.png', dpi=300)
    pp.close()


def plot_energy(rd, energy):
    # Prepare data
    energy_trimmed = np.trim_zeros(energy)
    iterations = np.arange(1, len(energy_trimmed) + 1)

    # Set style
    pp.figure(figsize=(6, 4))

    # Plot with semilog-y to highlight sharp initial decay
    with pp.style.context('seaborn-v0_8-darkgrid'):
        pp.semilogy(iterations, energy_trimmed, marker='s', linewidth=2, color='darkgreen', label="Energy")

    # Customize y-ticks
    pp.yticks(fontsize=10)
    pp.xticks(fontsize=10)

    # Labels and formatting
    pp.xlabel("Iteration", fontsize=12)
    pp.ylabel("Energy (log scale)", fontsize=12)
    pp.title("Energy Decay", fontsize=14)
    pp.grid(True, which='both', linestyle='--', linewidth=0.5)
    pp.legend(fontsize=10)
    pp.tight_layout()
    pp.savefig(rd + '/energy.png', dpi=300)
    pp.close()
