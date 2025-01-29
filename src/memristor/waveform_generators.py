import numpy as np

# generate triangular waveform {{{
def generate_triangular_wave(level, duration, dt, cycle=1):
    """
    Generate a triangular waveform

    Args:
        level (float):    min/max value of the waveform
        duration (float): the duration of a single pulse
        dt (float):       time step
        cycle (int):      number of times to repeat waveform
    """
    half_duration = duration / 2.0
    rising_time = np.arange(0, half_duration, dt)
    falling_time = np.arange(half_duration, duration, dt)

    rising_wave = (level / half_duration) * rising_time
    falling_wave = level - (level / half_duration) * (falling_time - half_duration)

    single_cycle = np.concatenate([rising_wave, falling_wave])

    # Repeat the single cycle to form multiple cycles
    full_wave = np.tile(single_cycle, cycle)

    # Generate the corresponding time array
    total_time_duration = duration * cycle
    time_array = np.arange(0, total_time_duration, dt)

    return time_array, full_wave
# }}}

# generate square waveform {{{
def generate_square_wave(
    level, delay, transition_time, pulse_duration, period, dt, cycles=1
):
    """
    Generate a square waveform

    Args:
        level (float):           min/max value of the waveform
        delay (float):           the time interval before the wave begins its first transition
        transition_time (float): low-to-high and high-to-low transition times
        pulse_duration (float):  duration of a single pulse
        period (float):          duration of a single cycle
        dt (float):              time step
        cycles (int):            number of times to repeat waveform
    """
    rising_edge = np.linspace(0, level, int(transition_time / dt))
    high_level = np.ones(int((pulse_duration - 2 * transition_time) / dt)) * level
    falling_edge = np.linspace(level, 0, int(transition_time / dt))
    low_level = np.zeros(int((period - pulse_duration) / dt))

    # Combine them to create one period
    single_cycle = np.concatenate([rising_edge, high_level, falling_edge, low_level])

    # Repeat the cycle to get the complete waveform and add the delay at the beginning
    pre_delay = np.zeros(int(delay / dt))
    full_wave = np.concatenate((pre_delay, np.tile(single_cycle, cycles)))

    # Generate the corresponding time array
    time_array = np.arange(0, len(full_wave) * dt, dt)

    return time_array, full_wave
# }}}

# generate custom wave {{{
def custom_wave_generator(*wave):
    """
    Generate wave that is a combination of triangular and square waves

    Args:
        wave (tuple): the x and y values of the wave
    """
    combined_time_array = np.array([])
    combined_wave = np.array([])
    last_time_point = 0

    for wave_function in wave:
        time_array, wave_array = wave_function
        time_array_shifted = time_array + last_time_point
        combined_time_array = np.concatenate([combined_time_array, time_array_shifted])
        combined_wave = np.concatenate([combined_wave, wave_array])

        # add dt to the last time point
        last_time_point = combined_time_array[-1] + time_array[1] - time_array[0]

    return combined_time_array, combined_wave
# }}}
