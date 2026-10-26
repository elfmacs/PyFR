import numpy as np

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.plugins.soln.base import BaseSolnPlugin


class EFCountPlugin(BaseSolnPlugin):
    # Count the elements touched by the entropy filter; reads the exported
    # ef-filter strength (1 = untouched) after the last stage of each step,
    # so elements filtered only in earlier stages are missed (a lower bound)
    name = 'efcount'
    systems = '.*'
    dimensions = '2|3'

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        if self.nsteps is None:
            self.nsteps = 1
        self._heart = self.cfg.getint(cfgsect, 'heartbeat', 1000)
        self._step0 = intg.nacptsteps
        self._seen = self._ncum = self._nev = 0
        self._ok = True

        _, rank, root = get_comm_rank_root()
        self._root = rank == root
        if self._root:
            self._fh = open(self.cfg.get(cfgsect, 'file'), 'a')
            if self._fh.tell() == 0:
                self._fh.write('t,steps,nfilt,fmin,x,y,z,ncum,nev\n')
            self._fh.write(f'# start t = {intg.tcurr}, '
                           f'accepted steps {intg.nacptsteps}\n')
            self._fh.flush()

        # Never let a failure here stop the run: log it and stop counting
        try:
            self._src = []
            for eles in intg.system.ele_map.values():
                g = [ef.getter for ef in eles.export_fields
                     if ef.name == 'ef-filter']
                if g:
                    cen = eles.ploc_at_np('upts').mean(axis=0).T
                    self._src.append((g[0], cen))

            if not self._src:
                raise ValueError('No ef-filter field; needs [solver] '
                                 'shock-capturing = entropy-filter')
        except Exception as e:
            self._fail(e)

    def _fail(self, e):
        self._ok = False
        if self._root:
            self._fh.write(f'# error, counting stopped: {e!r}\n')
            self._fh.flush()

    def __call__(self, intg):
        if not self._ok or intg.nacptsteps == self._step0:
            return

        try:
            comm, _, _ = get_comm_rank_root()

            n, best = 0, (2.0, 0.0, 0.0, 0.0)
            for get, cen in self._src:
                f = np.asarray(get())
                if (bad := f < 1).any():
                    n += int(bad.sum())
                    i = int(np.argmin(f))
                    best = min(best, (float(f[i]), *map(float, cen[i])))

            n = comm.allreduce(n, op=mpi.SUM)
            best = comm.allreduce(best, op=mpi.MIN)
            self._seen += 1
            self._ncum += n
            self._nev += n > 0

            if self._root and (n > 0 or self._seen % self._heart == 0):
                fmin, *xyz = best if n > 0 else (1.0, *[float('nan')]*3)
                self._fh.write(f'{intg.tcurr},{self._seen},{n},{fmin},'
                               f'{xyz[0]},{xyz[1]},{xyz[2]},{self._ncum},'
                               f'{self._nev}\n')
                self._fh.flush()
        except Exception as e:
            self._fail(e)
