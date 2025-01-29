import numpy as np

################################################################################
# Description: This script implements the Verilog-A model of the Standford RRAM
#              Model 1 in Python.
# URL: https://nanohub.org/publications/19/1
# Author: Mohamed Watfa
# Date: 09-10-2023
################################################################################

class RRAM:
    def __init__(
        self,
        model_switch=0,
        g0=0.25e-9,
        V0=0.25,
        Vel0=10,
        I0=1000e-6,
        alpha=3,
        beta=0.8,
        gamma0=16,
        T_crit=450,
        deltaGap0=0.02,
        T_smth=500,
        Ea=0.6,
        a0=0.25e-9,
        T_ini=273 + 25,
        F_min=1.4e9,
        gap_ini=2e-10,
        gap_min=2e-10,
        gap_max=17e-10,
        Rth=2.1e3,
        tox=12e-9,
        rand_seed_ini=0,
        time_step=1e-9,
        current_limit=1,
        scale_randomness=1,
    ):
        # Constants
        self.kb = 1.3806503e-23  # Boltzmann constant
        self.q = 1.6e-19  # Electron charge

        # Device structure parameters
        self.Ea = Ea  # Activation energy
        self.F_min = F_min  # Minimum field
        self.a0 = a0  # Atomic spacing
        self.tox = tox  # Oxide thickness
        self.T_ini = T_ini  # Device temperature
        self.Rth = Rth  # Thermal resistance

        # Fitting parameters
        self.I0 = I0  # Current level
        self.g0 = g0  # Resistance window
        self.V0 = V0  # Non-linearity of the curve

        # Gap
        self.gap_ini = gap_ini
        self.gap_min = gap_min
        self.gap_max = gap_max

        # Parameters related to gap growth
        self.Vel0 = Vel0
        self.beta = beta
        self.alpha = alpha
        self.gamma0 = gamma0

        # Parameters that describe the intrinsic (cycle-to-cycle and temporal) fluctuations during switching
        self.deltaGap0 = deltaGap0
        self.T_crit = T_crit
        self.T_smth = T_smth

        # Others
        self.current_limit = current_limit
        self.model_switch = model_switch
        self.time_step = time_step

        # Initialize simulation variables
        self.Vtb = 0
        self.Itb = 0
        self.T_cur = self.T_ini
        self.gap = self.gap_ini
        self.gap_ddt = 0

        # Set the random seed
        self.scale_randomness=scale_randomness
        # np.random.seed(rand_seed_ini)

    def limit_current(self):
        # calculate the current
        self.calculate_current()

        # limit the current if it exceeds limit
        if np.abs(self.Itb) > self.current_limit:
            self.Itb = self.current_limit
            self.calculate_voltage()

    def update_temperature(self):
        self.T_cur = self.T_ini + abs(self.Vtb * self.Itb * self.Rth)

    def update_gamma(self):
        self.gamma_ini = self.gamma0
        if self.Vtb < 0:
            self.gamma_ini = 16
        self.gamma = self.gamma_ini - self.beta * ((self.gap / 1e-9) ** self.alpha)
        if (self.gamma * abs(self.Vtb) / self.tox) < self.F_min:
            self.gamma = 0

    def update_gap(self):
        self.gap_ddt = (
            -self.Vel0
            * np.exp(-self.q * self.Ea / self.kb / self.T_cur)
            * np.sinh(
                self.gamma
                * self.a0
                / self.tox
                * self.q
                * self.Vtb
                / self.kb
                / self.T_cur
            )
        )

        # Add variability to gap
        if np.abs(self.Vtb) > 0 and self.model_switch == 1:
            deltaGap = self.deltaGap0
            gap_random_ddt = (
                np.random.normal(0, 1) * self.scale_randomness
                * deltaGap
                / (1 + np.exp((self.T_crit - self.T_cur) / self.T_smth))
            )
        else:
            gap_random_ddt = 0

        # NOTE: this approximation is not accurate, but faster
        # self.gap_ddt_history[self.time_step_index] = self.gap_ddt + gap_random_ddt
        # self.gap = np.trapz(self.gap_ddt_history, self.time_arr) + self.gap_ini

        self.gap += (self.gap_ddt + gap_random_ddt) * self.time_step

        # Keep gap within bounds
        if self.gap < self.gap_min:
            self.gap = self.gap_min
        elif self.gap > self.gap_max:
            self.gap = self.gap_max

    def calculate_current(self):
        self.Itb = self.I0 * np.exp(-self.gap / self.g0) * np.sinh(self.Vtb / self.V0)

    def calculate_voltage(self):
        self.Vtb = self.V0 * np.arcsinh(
            (self.Itb / self.I0) * np.exp(self.gap / self.g0)
        )

    def step(self, input, input_nature):
        if input_nature == "voltage":
            self.Vtb = input
            self.limit_current()
        else:
            self.Itb = input
            self.calculate_voltage()
        self.update_temperature()
        self.update_gamma()
        self.update_gap()

    def transient(self, time_arr, input_arr, input_nature="voltage"):
        # Save these parameters
        history = {
            key: np.zeros_like(input_arr)
            for key in ["gap_ddt", "gamma", "gap", "Itb", "Vtb"]
        }

        self.time_arr = time_arr
        self.gap_ddt_history = np.zeros_like(self.time_arr)
        self.time_step_index = 0

        for i, inp in enumerate(input_arr):
            _ = self.step(inp, input_nature)
            self.time_step_index += 1

            history["gap_ddt"][i] = self.gap_ddt
            history["gamma"][i] = self.gamma
            history["gap"][i] = self.gap
            history["Itb"][i] = self.Itb
            history["Vtb"][i] = self.Vtb

        return history

    def get_conductance(self, Vtb):
        Itb = self.I0 * np.exp(-self.gap / self.g0) * np.sinh(Vtb / self.V0)
        return Itb / Vtb
