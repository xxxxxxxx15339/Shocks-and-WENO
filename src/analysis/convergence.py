import numpy as np


def smooth_periodic_state(x):
    """A smooth, nontrivial periodic profile for spatial-order studies."""
    return np.exp(np.sin(2*np.pi*x))


def smooth_periodic_derivative(x):
    return 2*np.pi*np.cos(2*np.pi*x)*smooth_periodic_state(x)


def periodic_cell_averages(resolution, quadrature_order=12):
    """Integrate the smooth profile over each finite-volume cell."""
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    dx = 1.0/resolution
    centers = (np.arange(resolution, dtype=float)+0.5)*dx
    points = centers[:,None]+0.5*dx*nodes[None,:]
    averages = 0.5*np.sum(
        weights[None,:]*smooth_periodic_state(points), axis=1
    )
    return centers, averages


def periodic_advection_convergence(scheme_builder, resolutions=(40, 80, 160)):
    """Measure finite-volume cell-average evolution for u_t + u_x = 0."""
    rows = []
    previous_error = None
    previous_resolution = None
    for resolution in resolutions:
        dx = 1.0/resolution
        x, values = periodic_cell_averages(resolution)
        interface_flux = scheme_builder().evalF(values)
        numerical_derivative = (
            interface_flux-np.roll(interface_flux, 1)
        )/dx
        right_faces = (np.arange(resolution, dtype=float)+1)*dx
        left_faces = np.arange(resolution, dtype=float)*dx
        exact_average_derivative = (
            smooth_periodic_state(right_faces)-smooth_periodic_state(left_faces)
        )/dx
        error = np.mean(np.abs(numerical_derivative-exact_average_derivative))
        order = None
        if previous_error is not None:
            order = np.log(previous_error/error)/np.log(
                resolution/previous_resolution
            )
        rows.append({
            'resolution': resolution,
            'l1_error': float(error),
            'order': None if order is None else float(order),
        })
        previous_error = error
        previous_resolution = resolution
    return rows
