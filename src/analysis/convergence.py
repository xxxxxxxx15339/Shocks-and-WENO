import numpy as np


def smooth_periodic_state(x):
    """A smooth, nontrivial periodic profile for spatial-order studies."""
    return np.exp(np.sin(2*np.pi*x))


def smooth_periodic_derivative(x):
    return 2*np.pi*np.cos(2*np.pi*x)*smooth_periodic_state(x)


def periodic_advection_convergence(scheme_builder, resolutions=(40, 80, 160)):
    """Measure the semi-discrete L1 error for u_t + u_x = 0."""
    rows = []
    previous_error = None
    previous_resolution = None
    for resolution in resolutions:
        x = np.arange(resolution, dtype=float)/resolution
        dx = 1.0/resolution
        values = smooth_periodic_state(x)
        interface_flux = scheme_builder().evalF(values)
        numerical_derivative = (
            interface_flux-np.roll(interface_flux, 1)
        )/dx
        error = np.mean(np.abs(
            numerical_derivative-smooth_periodic_derivative(x)
        ))
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
