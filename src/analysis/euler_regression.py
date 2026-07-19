import numpy as np

from ..core.eulerEquations import GAMMA, flux, getEulerFlux, spds
from ..core.SimulationClasses import eulerSimulation
from ..core.TimeSteppingMethods import SSPRK3


def primitive_variables(state):
    density = state[:,0]
    velocity = state[:,1]/density
    pressure = (GAMMA-1)*(state[:,2]-0.5*density*velocity**2)
    return density, velocity, pressure


def run_euler_benchmark(scheme_builder, initial_condition, cells, length,
                        final_time, cfl=0.35):
    x = np.arange(cells, dtype=float)*length/cells
    initial = initial_condition(x)
    wave_speed = np.max(np.abs(spds(initial)))
    steps = int(np.ceil(final_time*wave_speed/(cfl*length/cells)))
    simulation = eulerSimulation(
        cells, steps+1, length, final_time, SSPRK3(),
        getEulerFlux(scheme_builder()), initial_condition, 3,
        max_wave_speed=wave_speed, max_cfl=cfl,
    )
    return x, simulation.runEuler()[:,-1,:], initial


def _pressure_function(pressure, density, initial_pressure):
    sound_speed = np.sqrt(GAMMA*initial_pressure/density)
    if pressure > initial_pressure:
        a = 2/((GAMMA+1)*density)
        b = (GAMMA-1)/(GAMMA+1)*initial_pressure
        root = np.sqrt(a/(pressure+b))
        value = (pressure-initial_pressure)*root
        derivative = root*(1-0.5*(pressure-initial_pressure)/(pressure+b))
    else:
        exponent = (GAMMA-1)/(2*GAMMA)
        ratio = pressure/initial_pressure
        value = 2*sound_speed/(GAMMA-1)*(ratio**exponent-1)
        derivative = ratio**(-(GAMMA+1)/(2*GAMMA))/(density*sound_speed)
    return value, derivative


def exact_sod_solution(x, time, discontinuity=0.5):
    """Exact Riemann solution for the canonical Sod initial condition."""
    rho_l, velocity_l, pressure_l = 1.0, 0.0, 1.0
    rho_r, velocity_r, pressure_r = 0.125, 0.0, 0.1
    pressure_star = 0.5*(pressure_l+pressure_r)
    for _ in range(20):
        f_l, df_l = _pressure_function(pressure_star, rho_l, pressure_l)
        f_r, df_r = _pressure_function(pressure_star, rho_r, pressure_r)
        update = (
            f_l+f_r+velocity_r-velocity_l
        )/(df_l+df_r)
        pressure_star -= update
        if abs(update) < 1e-13:
            break
    f_l, _ = _pressure_function(pressure_star, rho_l, pressure_l)
    f_r, _ = _pressure_function(pressure_star, rho_r, pressure_r)
    velocity_star = 0.5*(velocity_l+velocity_r+f_r-f_l)

    sound_l = np.sqrt(GAMMA*pressure_l/rho_l)
    sound_r = np.sqrt(GAMMA*pressure_r/rho_r)
    density_star_l = rho_l*(pressure_star/pressure_l)**(1/GAMMA)
    ratio = pressure_star/pressure_r
    density_star_r = rho_r*(
        (ratio+(GAMMA-1)/(GAMMA+1))
        / ((GAMMA-1)/(GAMMA+1)*ratio+1)
    )
    sound_star_l = sound_l*(pressure_star/pressure_l)**(
        (GAMMA-1)/(2*GAMMA)
    )
    rarefaction_head = velocity_l-sound_l
    rarefaction_tail = velocity_star-sound_star_l
    shock_speed = velocity_r+sound_r*np.sqrt(
        (GAMMA+1)/(2*GAMMA)*ratio+(GAMMA-1)/(2*GAMMA)
    )
    similarity = (x-discontinuity)/time
    density = np.empty_like(x)
    velocity = np.empty_like(x)
    pressure = np.empty_like(x)
    for i, xi in enumerate(similarity):
        if xi <= rarefaction_head:
            density[i], velocity[i], pressure[i] = rho_l, velocity_l, pressure_l
        elif xi <= rarefaction_tail:
            local_velocity = 2/(GAMMA+1)*(sound_l+xi)
            local_sound = 2/(GAMMA+1)*(
                sound_l-0.5*(GAMMA-1)*xi
            )
            density[i] = rho_l*(local_sound/sound_l)**(2/(GAMMA-1))
            velocity[i] = local_velocity
            pressure[i] = pressure_l*(local_sound/sound_l)**(
                2*GAMMA/(GAMMA-1)
            )
        elif xi <= velocity_star:
            density[i], velocity[i], pressure[i] = (
                density_star_l, velocity_star, pressure_star
            )
        elif xi <= shock_speed:
            density[i], velocity[i], pressure[i] = (
                density_star_r, velocity_star, pressure_star
            )
        else:
            density[i], velocity[i], pressure[i] = rho_r, velocity_r, pressure_r
    energy = pressure/(GAMMA-1)+0.5*density*velocity**2
    return np.column_stack((density, density*velocity, energy))


def _feature_locations(x, density):
    gradient = np.abs(np.gradient(density, x))
    exclusion = max(3, len(x)//40)
    strong = np.flatnonzero(gradient >= 0.5*np.max(gradient))
    shock_index = int(strong[-1])
    if shock_index <= exclusion:
        contact_index = shock_index
    else:
        window_cells = max(exclusion+1, int(np.ceil(1.5/(x[1]-x[0]))))
        window_start = max(0, shock_index-window_cells)
        contact_index = window_start+int(np.argmax(
            gradient[window_start:shock_index-exclusion]
        ))
    return float(x[shock_index]), float(x[contact_index])


def regression_metrics(x, solution, initial, final_time, reference):
    density, _, pressure = primitive_variables(solution)
    reference_density, _, reference_pressure = primitive_variables(reference)
    dx = x[1]-x[0]
    errors = np.mean(np.abs(solution-reference), axis=0)
    initial_integral = dx*np.sum(initial, axis=0)
    final_integral = dx*np.sum(solution, axis=0)
    boundary_change = final_time*(flux(initial[0:1])[0]-flux(initial[-1:])[0])
    shock, contact = _feature_locations(x, density)
    reference_shock, reference_contact = _feature_locations(x, reference_density)
    return {
        'density_l1': float(errors[0]),
        'momentum_l1': float(errors[1]),
        'energy_l1': float(errors[2]),
        'minimum_density': float(np.min(density)),
        'minimum_pressure': float(np.min(pressure)),
        'density_overshoot': float(max(
            0, np.max(density)-np.max(reference_density),
            np.min(reference_density)-np.min(density),
        )),
        'pressure_overshoot': float(max(
            0, np.max(pressure)-np.max(reference_pressure),
            np.min(reference_pressure)-np.min(pressure),
        )),
        'shock_location': shock,
        'contact_location': contact,
        'shock_location_error': abs(shock-reference_shock),
        'contact_location_error': abs(contact-reference_contact),
        'conservation_error': np.abs(
            final_integral-initial_integral-boundary_change
        ).tolist(),
    }


def interpolate_reference(reference_x, reference, target_x):
    return np.column_stack([
        np.interp(target_x, reference_x, reference[:,component])
        for component in range(3)
    ])
