import xsimlab as xs
import numpy as np

boltz = 5.67 * 1e-8  # W/m^2/K^4


@xs.process
class Spacecraft:
    radius = xs.variable(
        intent="in",
        description="spacecraft radius [m]",
        default=0.2111,
    )
    emis = xs.variable(
        intent="in", description="spacecraft emissivity", default=0.9, groups="sc_vars"
    )
    absorb = xs.variable(
        intent="in",
        description="spacecraft absorptivity",
        groups="sc_vars",
    )
    mass = xs.variable(
        intent="in", description="spacecraft mass [kg]", default=4.0
    )  # kg
    spec_heat = xs.variable(
        intent="in", description="spacecraft specific heat", default=897.0
    )  # J/(kg*K)
    Q_gen = xs.variable(
        intent="in",
        description="spacecraft power dissipation",
        default=15.0,
        groups="sc_vars",
    )  # @

    A_inc = xs.variable(
        intent="out",
        description="earth/ sun facing area",
        default=1.0,
        groups="sc_vars",
    )
    A_rad = xs.variable(
        intent="out", description="space facing area", default=1.0, groups="sc_vars"
    )

    heat_capacity = xs.variable(
        intent="out", description="spacecraft heat capacity", groups="sc_vars"
    )  # J/K

    def initialize(self):
        self.heat_capacity = self.spec_heat * self.mass
        self.A_inc = np.pi * self.radius ** 2
        self.A_rad = 4 * np.pi * self.radius ** 2


@xs.process
class Orbit:
    R = xs.variable(intent="in", description="body radius", default=6.3781e6)  # meters
    h = xs.variable(intent="in", description="orbit altitude", default=525e3)  # meters
    tau = xs.variable(
        intent="in", description="orbit period", default=90 * 60, groups="orb_vars"
    )  # orbital period, seconds
    case = xs.variable(intent="in", description="hot/ cold case", default="hot")
    beta = xs.variable(intent="in", description="beta, radians", default=0)

    f_E = xs.variable(intent="out", description="eclipse fraction", groups="orb_vars")
    q_sol = xs.variable(intent="out", description="solar loading", groups="orb_vars")
    albedo = xs.variable(intent="out", description="earth albedo", groups="orb_vars")
    q_ir = xs.variable(intent="out", description="earth IR", groups="orb_vars")

    def initialize(self):
        # Solar radiation at parhelion and aphelion
        q_sol = {"hot": 1414, "cold": 1322}  # W/m^2
        self.q_sol = q_sol[self.case]

        beta_star = np.arcsin(self.R / (self.R + self.h))  # beta angle, radians
        if self.beta >= beta_star:
            self.f_E = 0
        else:
            self.f_E = (
                np.arccos(
                    np.sqrt(self.h ** 2 + 2 * self.R * self.h)
                    / ((self.R + self.h) * np.cos(self.beta))
                )
                / np.pi
            )

        if self.beta >= 30:
            self.albedo = 0.19
            self.q_ir = 218
        else:
            self.albedo = 0.14
            self.q_ir = 228


@xs.process
class SingleNode:
    """Wrap single-mode thermal model in a single Process."""

    T_init = xs.variable(intent="in", description="initial temperature", default=290.0)
    T_out = xs.variable(intent="out", description="model temperature")

    orb_vars = xs.group_dict("orb_vars")
    sc_vars = xs.group_dict("sc_vars")

    def initialize(self):
        self.T_out = self.T_init
        self.time = 0
        # self.A_inc = self.sc_vars[("spacecraft", "A_inc")]

    @xs.runtime(args="step_delta")
    def run_step(self, dt):

        # eclipse function
        orb_frac = (self.time % self.orb_vars[("orbit", "tau")]) / self.orb_vars[
            ("orbit", "tau")
        ]
        if orb_frac >= self.orb_vars[("orbit", "f_E")]:
            sol_rad = 1
        else:
            sol_rad = 0

        # heat from incident radiation from Earth IR
        Q1 = self.orb_vars[("orbit", "q_ir")] * self.sc_vars[("spacecraft", "A_inc")]

        # heat from incident radiation from Sun + Earthshine
        Q2 = (
            (1 + self.orb_vars[("orbit", "albedo")])
            * self.orb_vars[("orbit", "q_sol")]
            * self.sc_vars[("spacecraft", "A_inc")]
            * sol_rad
            * self.sc_vars[("spacecraft", "absorb")]
        )

        # internally generated heat
        Q3 = self.sc_vars[("spacecraft", "Q_gen")]

        # heat from emited area to space
        Q4 = (
            self.sc_vars[("spacecraft", "A_rad")]
            * boltz
            * self.sc_vars[("spacecraft", "emis")]
            * self.T_out ** 4
        )  # Modified from paper to A_inc for simple radiaitor

        # total heat flow
        Q_dot = Q1 + Q2 + Q3 - Q4

        dTdt = Q_dot / (self.sc_vars[("spacecraft", "heat_capacity")])

        self.T1 = self.T_out + dTdt * dt
        self.time += dt

    def finalize_step(self):
        self.T_out = self.T1


single_node = xs.Model(
    {"spacecraft": Spacecraft, "orbit": Orbit, "thermal": SingleNode}
)
