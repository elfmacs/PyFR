import math

import numpy as np

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.plugins.common import init_csv
from pyfr.plugins.solver.base import BaseSolverPlugin
from pyfr.quadrules import get_quadrule


class FlowRatePlugin(BaseSolverPlugin):
    name = 'flowrate'
    systems = 'euler|navier-stokes'
    dimensions = '2|3'

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

        comm, rank, root = get_comm_rank_root()

        # Driven momentum component and its target volume average
        self.idx = 1 + 'xyz'.index(self.cfg.get(cfgsect, 'direction', 'x'))
        self.target = self.cfg.getfloat(cfgsect, 'bulk-momentum')

        # Controller relaxation time and update frequency
        self.trelax = self.cfg.getfloat(cfgsect, 'relax-time')
        self.nsteps = self.cfg.getint(cfgsect, 'nsteps')

        # Controller state; may be replaced by serialised data in setup
        self.force = self.cfg.getfloat(cfgsect, 'force-init', 0.0)
        self.eprev = self.tprev = None

        # Quadrature weights at the solution points
        self.wts, vol = [], 0.0
        for etype, eles in intg.system.ele_map.items():
            rname = self.cfg.get(f'solver-elements-{etype}', 'soln-pts')
            r = get_quadrule(etype, rname, eles.nupts)

            self.wts.append(r.wts[:, None] / eles.rcpdjac_at_np(r.pts))
            vol += self.wts[-1].sum()

        self.vol = comm.allreduce(vol, op=mpi.SUM)

        # Uniform body force, a runtime scalar read by the source macro
        for eles in intg.system.ele_map.values():
            eles.add_src_macro('pyfr.plugins.solver.kernels.flowrate',
                               'flowrate', {'fridx': self.idx}, soln=True)

        self._extern_values['frforce'] = self.force
        self._register_externs(intg, ['frforce'])

        # Output file
        if rank == root and self.cfg.hasopt(cfgsect, 'file'):
            self.csv = init_csv(self.cfg, cfgsect, 't,bulk,force',
                                isrestart=intg.isrestart)
        else:
            self.csv = None

    def setup(self, sdata, prevcfg, serialiser):
        if sdata is not None:
            self.force, eprev, tprev = map(float, sdata)

            if not math.isnan(tprev):
                self.eprev, self.tprev = eprev, tprev

        self._set_force()

        serialiser.register(self.sprefix, lambda: np.array([
            self.force,
            math.nan if self.eprev is None else self.eprev,
            math.nan if self.tprev is None else self.tprev
        ]))

    def _set_force(self):
        self._extern_values['frforce'] = self.force
        self.bind_externs()

    def __call__(self, intg):
        if intg.nacptsteps % self.nsteps:
            return

        comm, rank, root = get_comm_rank_root()

        # Volume averaged momentum in the driven direction
        mom = sum(np.sum(w*s[:, self.idx].astype(np.float64))
                  for w, s in zip(self.wts, intg.soln))
        bulk = comm.allreduce(mom, op=mpi.SUM) / self.vol

        t, err = intg.tcurr, bulk - self.target

        # Critically damped update with a double root of exp(-dt/trelax);
        # the err - eprev term matches the force to the mean drag over dt
        if self.tprev is not None and t > self.tprev:
            dt = t - self.tprev
            r = math.exp(-dt / self.trelax)

            self.force -= ((1 - r)**2*err + (1 - r*r)*(err - self.eprev))/dt
            self._set_force()

        self.eprev, self.tprev = err, t

        if self.csv:
            self.csv(t, bulk, self.force)
