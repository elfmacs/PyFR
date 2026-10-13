import math

import numpy as np

from pyfr.exprs import npeval
from pyfr.mpiutil import get_comm_rank_root
from pyfr.solvers.base.inters import _get_inter_arrays
from pyfr.solvers.navstokes.inters import NavierStokesBaseBCInters
from pyfr.util import CSVStream, first


def _lagrange(nodes, x, deriv=False):
    # Lagrange basis (or its derivative) through nodes evaluated at x
    n = len(nodes)
    V = np.vander(nodes, n, increasing=True)
    C = np.linalg.inv(V)
    if deriv:
        pw = np.array([k*x**(k - 1) if k else 0.0 for k in range(n)])
    else:
        pw = np.array([x**k for k in range(n)])
    return pw @ C


class NavierStokesSupInflowRecycleBCInters(NavierStokesBaseBCInters):
    """Supersonic inflow with recycled/rescaled turbulent fluctuations.

    The mean inflow (rho, u, v, w, p) is fixed and given by expressions;
    density and velocity fluctuations are taken from a downstream plane
    x = x-rec and rescaled following Pirozzoli, Bernardini & Grasso (2010,
    JFM 657, Eqs. 2.4-2.8) with the inner/outer weight of Lund, Wu &
    Squires (1998, JCP 140, Eq. 16).  Mean quantities at the recycle plane
    use the exponentially weighted time average of Lund et al. Eq. (18).
    Optional spanwise reflection follows Morgan et al. (2011, AIAA J 49).
    Serial only.
    """
    type = 'sup-in-recycle'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        comm, rank, root = get_comm_rank_root()
        if comm.size != 1:
            raise NotImplementedError('sup-in-recycle is serial only')
        if self.ndims != 3 or len(elemap) != 1:
            raise NotImplementedError('sup-in-recycle needs 3D, one etype')

        self.etype = first(elemap)
        self.eles = eles = elemap[self.etype]
        self.gamma = self.c['gamma']

        # Options
        g = lambda k, d=None: (cfg.getfloat(cfgsect, k, d) if d is not None
                               else cfg.getfloat(cfgsect, k))
        self.ut_in, self.rhow_in, self.mu_w = g('ut-in', 0.0), g('rhow-in', 0.0), g('mu-w')
        self.alpha, self.bpar = g('blend-alpha', 4.0), g('blend-b', 0.2)
        self.avg_time = g('avg-time')
        self.avg_cumul = g('avg-cumulative-from', math.inf)
        self.ue_y = cfg.getliteral(cfgsect, 'ue-y')
        self.rescale = cfg.getbool(cfgsect, 'rescale', True)
        self.reflect = cfg.get(cfgsect, 'reflect', 'none')
        if self.reflect not in {'none', 'shift', 'constant', 'dynamic'}:
            raise ValueError('Invalid reflect option')

        # Flux point locations in the order used by the kernels
        perm = self.side_perm(lhs)
        fploc, = _get_inter_arrays(lhs, 'get_ploc_for_inters', elemap, perm)

        # Fixed mean inflow state
        cc = cfg.items_as('constants', float)
        loc = dict(x=fploc[:, 0], y=fploc[:, 1], z=fploc[:, 2]) | cc
        defs = {'v': '0', 'w': '0'}
        mean = [npeval(cfg.getexpr(cfgsect, k, defs.get(k)), loc)
                for k in ('rho', 'u', 'v', 'w', 'p')]
        self.mean_in = np.array([np.broadcast_to(m, fploc[:, 0].shape)
                                 for m in mean], dtype=float)

        # Recycle plane: the solution point plane nearest to x-rec
        upts = eles.ploc_at_np('upts')                    # (nupts, 3, neles)
        xs = np.unique(np.round(upts[:, 0], 9))
        xr = xs[np.argmin(np.abs(xs - g('x-rec')))]
        self.x_rec = float(xr)
        uidx, eidx = np.nonzero(np.abs(upts[:, 0] - xr) < 1e-8)
        ry, rz = upts[uidx, 1, eidx], upts[uidx, 2, eidx]

        # Node grid in (y, z) shared by the recycle plane and the inlet
        tol = 1e-7
        self.yn = yn = np.unique(np.round(ry, 9))
        self.zn = zn = np.unique(np.round(rz, 9))
        iy = np.searchsorted(yn, np.round(ry, 9))
        iz = np.searchsorted(zn, np.round(rz, 9))
        if len(uidx) != len(yn)*len(zn):
            raise ValueError('Recycle plane is not a tensor grid')

        # Order the recycle points as a (ny, nz) grid
        order = np.lexsort((iz, iy))
        self._r_uidx, self._r_eidx = uidx[order], eidx[order]
        self.ny, self.nz = len(yn), len(zn)

        # Inlet flux points onto the same grid
        fy = np.searchsorted(yn, np.round(fploc[:, 1], 9))
        fz = np.searchsorted(zn, np.round(fploc[:, 2], 9))
        fy, fz = np.clip(fy, 0, len(yn) - 1), np.clip(fz, 0, len(zn) - 1)
        if (np.abs(yn[fy] - fploc[:, 1]).max() > tol or
            np.abs(zn[fz] - fploc[:, 2]).max() > tol):
            raise ValueError('Inlet flux points are not on the recycle grid')
        self._f_iy, self._f_iz = fy, fz

        # Element rows in y: node groups and element edges
        corners = eles.ploc_at_np(np.array([[-1., -1., -1.], [1., 1., 1.]]))
        ylo = corners[0, 1, self._r_eidx].reshape(self.ny, self.nz)[:, 0]
        yhi = corners[1, 1, self._r_eidx].reshape(self.ny, self.nz)[:, 0]
        self.yedges = np.unique(np.round(np.r_[ylo, yhi], 9))
        self.ygrp = np.searchsorted(self.yedges, yn, side='right') - 1
        self.ytop = self.yedges[-1]

        # Spanwise weights (Gauss) for plane averages; elements are uniform
        zcorner = corners[:, 2, self._r_eidx].reshape(2, self.ny, self.nz)[:, 0]
        self.zedges = np.unique(np.round(zcorner.ravel(), 9))
        self.lz = self.zedges[-1] - self.zedges[0]
        zgrp = np.searchsorted(self.zedges, zn, side='right') - 1
        zw = np.zeros(self.nz)
        for k in np.unique(zgrp):
            m = zgrp == k
            ref = 2*(zn[m] - self.zedges[k])/(self.zedges[k + 1] - self.zedges[k]) - 1
            zw[m] = np.polynomial.legendre.leggauss(m.sum())[1][np.argsort(np.argsort(ref))]
        self.zw = zw/zw.sum()

        # Valid reflection centres map the z nodes onto themselves
        cands = np.unique(np.round(np.r_[self.zedges[:-1], 0.5*(self.zedges[1:] + self.zedges[:-1])], 9))
        self._zc_valid = [c for c in cands if self._zperm(c) is not None]


        # Inlet scales measured from the fixed mean inflow with the same
        # discrete operators as at the recycle plane, so that identical
        # profiles give gamma = 1 and identity coordinate maps
        prof = np.zeros((2, self.ny))
        for i in range(self.ny):
            k = np.nonzero(fy == i)[0][0]
            prof[:, i] = self.mean_in[0, k], self.mean_in[1, k]
        rhow, ut, ue, delta = self._wall_and_edge(prof[0], prof[1])
        self.cfg_ut_in, self.cfg_rhow_in = self.ut_in, self.rhow_in
        self.ut_in, self.rhow_in, self.delta_in = ut, rhow, delta
        self.W = self._weight(self.yn/self.delta_in)

        # Host-supplied inflow state (primitive) at each flux point
        self._rcst = be.matrix((self.nvars, self.ninterfpts),
                               initval=self.mean_in.copy())
        self.set_external('rcst', f'in fpdtype_t[{self.nvars}]',
                          value=self._rcst)

        # Optional log
        self.nlog = cfg.getint(cfgsect, 'log-nsteps', 100)
        if cfg.hasopt(cfgsect, 'file'):
            self.csv = CSVStream(cfg.get(cfgsect, 'file'),
                                 header='t,x_rec,ut_rec,rhow_rec,delta_rec,'
                                        'ue_rec,gamma,zc,avg_T')
        else:
            self.csv = None

        self._kern = None

        # Debug: dump the first computed inflow state for offline checks
        self.dump = cfg.get(cfgsect, 'debug-dump', None)
        self._fploc = fploc

    def _weight(self, Y):
        # Lund et al. (1998) Eq. (16) with W = 1 for Y > 1
        a, b = self.alpha, self.bpar
        with np.errstate(divide='ignore', invalid='ignore'):
            W = 0.5*(1 + np.tanh(a*(Y - b)/((1 - 2*b)*Y + b))/np.tanh(a))
        return np.where(Y > 1, 1.0, W)

    # --- reflection -------------------------------------------------------
    def _zperm(self, zc):
        zs = np.mod(2*zc - self.zn - self.zedges[0], self.lz) + self.zedges[0]
        j = np.searchsorted(self.zn, np.round(zs, 9))
        j = np.clip(j, 0, self.nz - 1)
        jm = np.clip(j - 1, 0, self.nz - 1)
        j = np.where(np.abs(self.zn[jm] - zs) < np.abs(self.zn[j] - zs), jm, j)
        # Periodic wrap at the ends
        d = np.abs(self.zn[j] - zs)
        d = np.minimum(d, self.lz - d)
        return j if d.max() < 1e-7 else None

    def _src_index(self):
        # Spanwise source index for each z node and the sign of w
        if self.reflect == 'none':
            return np.arange(self.nz), 1.0
        elif self.reflect == 'shift':
            zs = np.mod(self.zn - self.zedges[0] + 0.5*self.lz, self.lz) + self.zedges[0]
            j = np.argmin(np.abs(self.zn[None] - zs[:, None]), axis=1)
            if np.abs(self.zn[j] - zs).max() > 1e-7:
                raise ValueError('Lz/2 shift is not aligned with the z nodes')
            return j, 1.0
        else:
            return self._zperm(self.zc), -1.0

    def _new_reflection(self, t):
        # Morgan et al. (2011) Sec. III.F: uniform random centre, normally
        # distributed streak length; held for tau = lambda/(uc U_inf)
        rng = np.random.default_rng([self.seed, self.nevents])
        self.zc = self._zc_valid[rng.integers(len(self._zc_valid))]
        lam = -1.0
        while lam <= 0:
            lam = rng.normal(self.lam_mean, self.lam_std)
        self.t_switch = t + lam/(self.uc*self.uinf)
        self.nevents += 1

    # --- state ------------------------------------------------------------
    def setup(self, sdata, prevcfg):
        cfg, sect = self.cfg, self.cfgsect

        if self.reflect == 'constant':
            self.zc = cfg.getfloat(sect, 'reflect-centre')
            if self._zperm(self.zc) is None:
                raise ValueError('reflect-centre must be an element edge or '
                                 'centre in z')
        elif self.reflect == 'dynamic':
            self.seed = cfg.getint(sect, 'reflect-seed', 1)
            self.lam_mean = cfg.getfloat(sect, 'reflect-length-mean')
            self.lam_std = cfg.getfloat(sect, 'reflect-length-std')
            self.uc = cfg.getfloat(sect, 'reflect-uc', 0.8)
            self.uinf = cfg.getfloat(sect, 'reflect-uinf', 1.0)

        n = 4*self.ny
        if sdata is not None and len(sdata) == n + 6 and sdata[n + 2] > 0:
            self.means = sdata[:n].reshape(4, self.ny).copy()
            self.tprev, self.t0, self.nupdates = sdata[n:n + 3]
            self.zc, self.t_switch, self.nevents = sdata[n + 3:n + 6]
            self.nupdates, self.nevents = int(self.nupdates), int(self.nevents)
            if self.reflect == 'constant':
                self.zc = cfg.getfloat(sect, 'reflect-centre')
        else:
            self.means = None
            self.tprev, self.t0, self.nupdates = None, None, 0
            self.t_switch, self.nevents = -math.inf, 0
            if self.reflect != 'constant':
                self.zc = 0.0

    @classmethod
    def serialisefn(cls, bciface, prefix, srl):
        def sfn():
            b = bciface
            if b.means is None:
                return np.zeros(4*b.ny + 6)
            return np.r_[b.means.ravel(), b.tprev, b.t0, b.nupdates, b.zc,
                         b.t_switch, b.nevents].astype(float)

        srl.register(prefix, sfn if bciface else None)

    @classmethod
    def hookfns(cls, bciface, mesh, elemap):
        return (None, bciface.advance) if bciface else (None, None)

    # --- recycle plane gather (device) -------------------------------------
    def _build_kernel(self, intg):
        be = intg.backend
        eles = self.eles
        etidx = intg.system.ele_types.index(self.etype)

        be.pointwise.register('pyfr.plugins.soln.kernels.sample')
        npts = len(self._r_eidx)
        wts = np.zeros((npts, eles.nupts))
        wts[np.arange(npts), self._r_uidx] = 1.0
        self._k_wts = be.const_matrix(wts.T, tags={'align'})
        self._k_out = be.matrix((self.nvars, npts), tags={'align'})
        self._k_tpl = dict(ndims=self.ndims, nvars=self.nvars,
                           nsvars=self.nvars, has_grads=False, primitive=True,
                           c=self.cfg.items_as('constants', float),
                           eos_mod=eles.eos_kernel_module, nupts=eles.nupts)
        self._k_banks = intg.system.ele_banks[etidx]
        self._kerns = {}

    def _gather(self, intg):
        be = intg.backend
        uidx = intg.idxcurr
        if uidx not in self._kerns:
            bank = self._k_banks[uidx]
            n = len(self._r_eidx)
            view = be.view(np.full(n, bank.mid), np.zeros(n, dtype=np.int32),
                           self._r_eidx, np.ones(n, dtype=np.int32),
                           vshape=(self.eles.nupts, self.nvars))
            self._kerns[uidx] = be.pointwise.sample(
                tplargs=self._k_tpl, dims=[n], u=view, gradu=None,
                wts=self._k_wts, out=self._k_out
            )
        be.run_kernels([self._kerns[uidx]])
        out = self._k_out.get()                           # (nvars, npts)
        return out.reshape(self.nvars, self.ny, self.nz)

    # --- rescaling (host) ---------------------------------------------------
    def _interp_y(self, F, yt):
        # F: (nv, ny, nz) nodal values; yt: (m,) targets -> (nv, m, nz)
        out = np.zeros((F.shape[0], len(yt), F.shape[2]))
        for i, y in enumerate(yt):
            if y < 0 or y >= self.ytop:
                continue
            k = min(np.searchsorted(self.yedges, y, side='right') - 1,
                    len(self.yedges) - 2)
            idx = np.nonzero(self.ygrp == k)[0]
            out[:, i] = np.einsum('n,vnz->vz', _lagrange(self.yn[idx], y),
                                  F[:, idx])
        return out

    def _wall_and_edge(self, rho=None, u=None):
        if rho is None:
            rho, u = self.means[0], self.means[1]
        idx = np.nonzero(self.ygrp == 0)[0]
        rhow = _lagrange(self.yn[idx], 0.0) @ rho[idx]
        dudy = _lagrange(self.yn[idx], 0.0, deriv=True) @ u[idx]
        ut = math.sqrt(max(self.mu_w*dudy, 0.0)/rhow)
        y0, y1 = self.ue_y
        sel = (self.yn >= y0) & (self.yn <= y1)
        ue = u[sel].mean()
        k = np.argmax(u >= 0.99*ue)
        delta = (np.interp(0.99*ue, u[k - 1:k + 1], self.yn[k - 1:k + 1])
                 if k > 0 else self.yn[0])
        return rhow, ut, ue, delta

    def advance(self, intg):
        t = intg.tcurr
        if self.tprev is not None and t <= self.tprev:
            return

        if self._kern is None:
            self._build_kernel(intg)
            self._kern = True

        # Recycle plane primitive state on the (y, z) node grid
        P = self._gather(intg)
        prho, pu, pv, pw = P[0], P[1], P[2], P[3]

        # Spanwise averages and Lund Eq. (18) time average
        inst = np.array([f @ self.zw for f in (prho, pu, pv, pw)])
        if self.means is None:
            self.means, self.t0 = inst.copy(), t
        else:
            dt = t - self.tprev
            T = (self.avg_time if t < self.avg_cumul
                 else self.avg_time + t - self.avg_cumul)
            a = min(dt/T, 1.0)
            self.means = a*inst + (1 - a)*self.means
        self.tprev = t
        self.nupdates += 1

        # Fluctuations at the recycle plane
        F = np.array([prho, pu, pv, pw]) - self.means[:, :, None]

        # Scaling parameters
        rhow, ut, ue, delta = self._wall_and_edge()
        if self.rescale:
            gam = (self.ut_in*math.sqrt(self.rhow_in))/(ut*math.sqrt(rhow))
            nu_in, nu_rec = self.mu_w/self.rhow_in, self.mu_w/rhow
            y_inn = self.yn*(self.ut_in/nu_in)*(nu_rec/ut)
            y_out = self.yn*delta/self.delta_in
        else:
            gam, y_inn, y_out = 1.0, self.yn, self.yn

        # Spanwise source (reflection / shift)
        if self.reflect == 'dynamic' and t >= self.t_switch:
            self._new_reflection(t)
        jsrc, wsign = self._src_index()
        F = F[:, :, jsrc]
        F[3] *= wsign

        # Inner/outer fluctuations and their blend (P2010a Eqs. 2.5-2.7)
        Fi = self._interp_y(F, y_inn)
        Fo = self._interp_y(F, y_out)
        W = self.W[None, :, None]
        Fb = Fi*(1 - W) + Fo*W
        Fb[1:] *= gam

        # Inflow primitive state at the flux points
        st = self.mean_in.copy()
        fy, fz = self._f_iy, self._f_iz
        st[0] += Fb[0][fy, fz]
        st[1] += Fb[1][fy, fz]
        st[2] += Fb[2][fy, fz]
        st[3] += Fb[3][fy, fz]
        if st[0].min() <= 0:
            raise RuntimeError(f'sup-in-recycle: non-positive inflow density '
                               f'at t = {t}')
        self._rcst.set(st)

        if self.dump and self.nupdates == 1:
            np.savez(self.dump, t=t, st=st, fploc=self._fploc, P=P,
                     means=self.means, yn=self.yn, zn=self.zn, zw=self.zw,
                     ut=ut, rhow=rhow, delta=delta, ue=ue, gam=gam,
                     zc=self.zc, x_rec=self.x_rec, ut_in=self.ut_in,
                     rhow_in=self.rhow_in, delta_in=self.delta_in, W=self.W)

        self.last = dict(t=t, ut=ut, rhow=rhow, delta=delta, ue=ue, gam=gam)
        if self.csv and self.nupdates % self.nlog == 0:
            T = (self.avg_time if t < self.avg_cumul
                 else self.avg_time + t - self.avg_cumul)
            self.csv(t, self.x_rec, ut, rhow, delta, ue, gam, self.zc, T)
