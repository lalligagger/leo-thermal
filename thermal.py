import xsimlab as xs
import numpy as np

boltz = 5.67 * 1e-8  # W/m^2/K^4


@xs.process
class Spacecraft:
    """
    A process to ingest and compute spacecraft (SC) properties.
    """

    # In vars
    radius = xs.variable(
        intent="in",
        description="spacecraft radius [m]",
        default=0.2111,
    )
    emis = xs.variable(
        intent="in",
        description="spacecraft emissivity",
        default=0.9,
        groups="sc_vars",
    )
    mass = xs.variable(
        intent="in",
        description="spacecraft mass [kg]",
        default=4.0,
    )
    spec_heat = xs.variable(
        intent="in",
        description="spacecraft specific heat [J/(kg*K)]",
        default=897.0,
    )
    Q_gen = xs.variable(
        intent="in",
        description="spacecraft power dissipation [W]",
        default=15.0,
        groups="sc_vars",
    )

    # Out vars
    A_inc = xs.variable(
        intent="out",
        description="earth/ sun facing area [m^2]",
        default=1.0,
        groups="sc_vars",
    )
    A_rad = xs.variable(
        intent="out",
        description="space facing area [m^2]",
        default=1.0,
        groups="sc_vars",
    )

    heat_capacity = xs.variable(
        intent="out",
        description="spacecraft heat capacity [J/K]",
        groups="sc_vars",
    )

    def initialize(self):
        """
        Initialize Spacecraft.
        """
        self.heat_capacity = self.spec_heat * self.mass
        self.A_inc = np.pi * self.radius ** 2
        self.A_rad = 4 * np.pi * self.radius ** 2


@xs.process
class Orbit:
    """
    A process to ingest and compute spacecraft (SC) properties.
    """

    # In vars
    R = xs.variable(
        intent="in",
        description="orbiting body (Earth) radius [m]",
        default=6.3781e6,
    )
    h = xs.variable(
        intent="in",
        description="satellite orbit altitude [m]",
        default=525e3,
    )
    tau = xs.variable(
        intent="in",
        description="orbit period [s]",
        default=90 * 60,
        groups="orb_vars",
    )
    case = xs.variable(
        intent="in",
        description="chose orbit hot/ cold case",
        default="hot",
    )
    beta = xs.variable(
        intent="in",
        description="orbit plane vs. sun beta angle [rad]",
        default=0,
    )

    # Out vars
    f_E = xs.variable(
        intent="out",
        description="orbit eclipse fraction",
        groups="orb_vars",
    )
    q_sol = xs.variable(
        intent="out",
        description="solar loading [W/m^2]",
        groups="orb_vars",
    )
    q_ir = xs.variable(
        intent="out",
        description="earth IR loading [W/m^2]",
        groups="orb_vars",
    )
    albedo = xs.variable(
        intent="out",
        description="earth albedo loading [W/m^2]",
        groups="orb_vars",
    )

    def initialize(self):
        """
        Define fraction of orbit spent in eclipse based on period, geometry, 
        and beta angle.
        Define the solar, Earth albedo, and Earth infrared loading in [W/m^2].
        """
        # Solar radiation at parhelion and aphelion
        q_sol = {"hot": 1414, "cold": 1322}  # W/m^2
        self.q_sol = q_sol[self.case]

        # critical beta angle, radians
        beta_star = np.arcsin(self.R / (self.R + self.h))
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
    """
    Wrap single-mode thermal model in a single process, 
    attached to Orbit and Spacecraft proceces.
    """

    # In vars
    T_init = xs.variable(
        intent="in",
        description="initial spacecraft temperature [K]",
        default=290.0,
    )
    # Out vars
    T_out = xs.variable(
        intent="out",
        description="spacecraft temperature per time step [K]",
    )

    # Foreign vars
    orb_vars = xs.group_dict("orb_vars")
    sc_vars = xs.group_dict("sc_vars")

    def initialize(self):
        self.T_out = self.T_init
        self.time = 0
        # self.A_inc = self.sc_vars[("spacecraft", "A_inc")]

    @xs.runtime(args="step_delta")
    def run_step(self, dt):
        """
        Runtime function for thermal model. Adds each type of heat loading and
        increments teperature based on heat capacity/ time step.
        """
        # eclipse function
        orbit_mod = self.time % self.orb_vars[("orbit", "tau")]
        orb_frac = orbit_mod / self.orb_vars[("orbit", "tau")]
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
            * self.sc_vars[("spacecraft", "emis")]
        )

        # internally generated heat
        Q3 = self.sc_vars[("spacecraft", "Q_gen")]

        # heat from emited area to space
        Q4 = (
            self.sc_vars[("spacecraft", "A_rad")]
            * boltz
            * self.sc_vars[("spacecraft", "emis")]
            * self.T_out ** 4
        )

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
