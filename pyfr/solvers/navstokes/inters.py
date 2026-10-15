import math
import numpy as np

from pyfr.exprs import npeval
from pyfr.solvers.baseadvecdiff import (BaseAdvectionDiffusionBCInters,
                                        BaseAdvectionDiffusionIntInters,
                                        BaseAdvectionDiffusionMPIInters)
from pyfr.solvers.euler.inters import MassFlowBCMixin, PressureBCMixin


class TplargsMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        visc_corr = self.cfg.get('solver', 'viscosity-correction', 'none')
        shock_capturing = self.cfg.get('solver', 'shock-capturing', 'none')
        if shock_capturing == 'entropy-filter':
            self.p_min = self.cfg.getfloat('solver-entropy-filter', 'p-min',
                                           1e-6)
        else:
            self.p_min = self.cfg.getfloat('solver-interfaces', 'p-min',
                                           5*self._be.fpdtype_eps)

        self._tplargs = dict(ndims=self.ndims, nvars=self.nvars,
                             rsolver=rsolver, visc_corr=visc_corr,
                             shock_capturing=shock_capturing, c=self.c,
                             p_min=self.p_min)


class NavierStokesIntInters(TplargsMixin,
                            BaseAdvectionDiffusionIntInters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._tplargs['rperiodic'] = self._rperiodic

        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.intconu')
        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.intcflux')

        self.kernels['con_u'] = lambda: self._be.kernel(
            'intconu', tplargs=self._tplargs, dims=[self.ninterfpts],
            ulin=self.scal_lhs, urin=self.scal_rhs,
            ulout=self._comm_lhs, urout=self._comm_rhs, rmat=self._rotmat
        )
        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'intcflux', tplargs=self._tplargs, dims=[self.ninterfpts],
            ul=self.scal_lhs, ur=self.scal_rhs,
            gradul=self._vect_lhs, gradur=self._vect_rhs,
            artvisc=self.artvisc, nl=self._pnorm_lhs, rmat=self._rotmat
        )


class NavierStokesMPIInters(TplargsMixin,
                            BaseAdvectionDiffusionMPIInters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.mpiconu')
        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.mpicflux')

        # Our side of the common solution jump is zero if β = -0.5
        if self.c['ldg-beta'] != -0.5:
            self.kernels['con_u'] = lambda: self._be.kernel(
                'mpiconu', tplargs=self._tplargs, dims=[self.ninterfpts],
                ulin=self.scal_lhs, urin=self.scal_rhs, ulout=self._comm_lhs
            )
        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'mpicflux', tplargs=self._tplargs, dims=[self.ninterfpts],
            ul=self.scal_lhs, ur=self.scal_rhs,
            gradul=self._vect_lhs, gradur=self._vect_rhs,
            artvisc=self.artvisc, nl=self._pnorm_lhs
        )


class NavierStokesBaseBCInters(TplargsMixin, BaseAdvectionDiffusionBCInters):
    cflux_state = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Additional BC specific template arguments
        self._tplargs['bctype'] = self.type
        self._tplargs['bccfluxstate'] = self.cflux_state

        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.bcconu')
        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.bccflux')

        self.kernels['con_u'] = lambda: self._be.kernel(
            'bcconu', tplargs=self._tplargs, dims=[self.ninterfpts],
            extrns=self._external_args, ulin=self.scal_lhs,
            ulout=self._comm_lhs, nlin=self._pnorm_lhs,
            **self._external_vals
        )
        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'bccflux', tplargs=self._tplargs, dims=[self.ninterfpts],
            extrns=self._external_args, ul=self.scal_lhs,
            gradul=self._vect_lhs, nl=self._pnorm_lhs,
            artvisc=self.artvisc, **self._external_vals
        )

    def comm_entropy_kernel(self, entmin_lhs):
        # Physics-specific callback for entropy filtering
        self._be.pointwise.register(
            'pyfr.solvers.navstokes.kernels.bccent'
        )

        return lambda: self._be.kernel(
            'bccent', tplargs=self._tplargs, dims=[self.ninterfpts],
            extrns=self._external_args, entmin_lhs=entmin_lhs,
            nl=self._pnorm_lhs, ul=self.scal_lhs, **self._external_vals
        )


class NavierStokesNoSlpIsotWallBCInters(NavierStokesBaseBCInters):
    type = 'no-slp-isot-wall'
    cflux_state = 'ghost-imperm'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c['cpTw'], = self._eval_opts(['cpTw'])
        self.c |= self._exp_opts('uvw'[:self.ndims], lhs,
                                 default={'u': 0, 'v': 0, 'w': 0})

    @classmethod
    def common_consts(cls, cfg, cfgsect, ndims):
        # Mirrors bc_ldg_state in no-slp-isot-wall.mako
        consts = cfg.items_as('constants', float)

        c = {'gamma': consts['gamma'],
             'cpTw': npeval(cfg.getexpr(cfgsect, 'cpTw'), consts)}
        for v in 'uvw'[:ndims]:
            c[v] = npeval(cfg.getexpr(cfgsect, v, '0'), consts)

        return c

    @staticmethod
    def common_pri_state(pl, nl, c):
        rho = pl[0]
        uw = [np.full_like(rho, c[v]) for v in 'uvw'[:len(pl) - 2]]
        p = (c['gamma'] - 1)/c['gamma']*c['cpTw']*rho

        return [rho, *uw, p]


class NavierStokesNoSlpAdiaWallBCInters(NavierStokesBaseBCInters):
    type = 'no-slp-adia-wall'
    cflux_state = 'ghost-imperm'

    @staticmethod
    def common_pri_state(pl, nl, c):
        # No-slip wall: zero velocity, as bc_ldg_state in the mako
        return [pl[0], *(np.zeros_like(v) for v in pl[1:-1]), pl[-1]]


class NavierStokesSlpAdiaWallBCInters(NavierStokesBaseBCInters):
    type = 'slp-adia-wall'
    cflux_state = None

    @staticmethod
    def common_pri_state(pl, nl, c):
        # Slip wall: remove the normal velocity, as bc_ldg_state in the mako
        vel = np.asarray(pl[1:-1])
        un = np.linalg.vecdot(vel, nl, axis=0)

        return [pl[0], *(vel - un*nl), pl[-1]]


class NavierStokesCharRiemInvBCInters(NavierStokesBaseBCInters):
    type = 'char-riem-inv'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c |= self._exp_opts(
            ['rho', 'p', 'u', 'v', 'w'][:self.ndims + 2], lhs
        )


class NavierStokesSupInflowBCInters(NavierStokesBaseBCInters):
    type = 'sup-in-fa'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c |= self._exp_opts(
            ['rho', 'p', 'u', 'v', 'w'][:self.ndims + 2], lhs
        )


class NavierStokesSupOutflowBCInters(NavierStokesBaseBCInters):
    type = 'sup-out-fn'
    cflux_state = 'ghost'


class NavierStokesSubInflowFrvBCInters(NavierStokesBaseBCInters):
    type = 'sub-in-frv'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c |= self._exp_opts(
            ['rho', 'u', 'v', 'w'][:self.ndims + 1], lhs,
            default={'u': 0, 'v': 0, 'w': 0}
        )


class NavierStokesSubInflowFtpttangBCInters(NavierStokesBaseBCInters):
    type = 'sub-in-ftpttang'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        gamma = self.cfg.getfloat('constants', 'gamma')

        # Pass boundary constants to the backend
        self.c['cpTt'], = self._eval_opts(['cpTt'])
        self.c |= self._exp_opts(['pt'], lhs)
        self.c['Rdcp'] = (gamma - 1.0)/gamma

        # Calculate u, v velocity components from the inflow angle
        theta = self._eval_opts(['theta'])[0]*np.pi/180.0
        velcomps = np.array([np.cos(theta), np.sin(theta), 1.0])

        # Adjust u, v and calculate w velocity components for 3-D
        if self.ndims == 3:
            phi = self._eval_opts(['phi'])[0]*np.pi/180.0
            velcomps[:2] *= np.sin(phi)
            velcomps[2] *= np.cos(phi)

        self.c['vc'] = velcomps[:self.ndims]


class NavierStokesSubOutflowBCInters(NavierStokesBaseBCInters):
    type = 'sub-out-fp'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c |= self._exp_opts(['p'], lhs)


class NavierStokesCharRiemInvMassFlowBCInters(MassFlowBCMixin,
                                              NavierStokesBaseBCInters):
    type = 'char-riem-inv-mass-flow'
    cflux_state = 'ghost'


class NavierStokesCharRiemInvPressureBCInters(PressureBCMixin,
                                              NavierStokesBaseBCInters):
    type = 'char-riem-inv-pressure'
    cflux_state = 'ghost'


class NavierStokesWMAdiaWallBCInters(NavierStokesBaseBCInters):
    type = 'wm-adia-wall'
    cflux_state = None

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        if self._tplargs['visc_corr'] != 'none':
            raise ValueError('wm-adia-wall requires constant viscosity')

        if any(etype != 'hex' for etype, *_ in lhs.foreach()):
            raise ValueError('wm-adia-wall only supports hex elements')

        # Velocity input: the state at one solution point (sample = point,
        # sample-point k) or the mean along the column of solution points
        # normal to the wall (sample = column-mean), matched to the mean of
        # the wall law over the wall-element height
        sample = cfg.get(cfgsect, 'sample', 'point')
        if sample not in {'point', 'column-mean'}:
            raise ValueError(f'wm-adia-wall: unknown sample {sample}')

        wmint = sample == 'column-mean'
        k = 1 if wmint else cfg.getint(cfgsect, 'sample-point')
        debug = cfg.getbool(cfgsect, 'debug', False)

        wmlaw = cfg.get(cfgsect, 'wall-law', 'log-exp')
        if wmlaw not in {'linear', 'log-exp', 'fixed', 'zero-stress'}:
            raise ValueError(f'wm-adia-wall: unknown wall-law {wmlaw}')

        # LOG-EXP defaults: Zhang, Zhou, Yang & Zhang, arXiv:2104.09054,
        # eq. (10); kappa = 0.4 from their section I
        wmc = {'niters': cfg.getint(cfgsect, 'newton-iters', 4),
               'umin': 1e-30}
        for cn, cv in [('kappa', 0.4), ('A', 11.630), ('B', 7.194),
                       ('C', -4.472), ('D', 2.766)]:
            wmc[cn] = cfg.getfloat(cfgsect, cn, cv)

        # Diagnostic: wall shear stress of fixed magnitude tau-fixed along the
        # sampled tangential velocity
        if wmlaw == 'fixed':
            wmc['tau'] = cfg.getfloat(cfgsect, 'tau-fixed')

        # Ghost state used by the inviscid Riemann solve at the wall
        wmghost = cfg.get(cfgsect, 'inviscid-state', 'slip')
        if wmghost not in {'slip', 'noslip'}:
            raise ValueError(f'wm-adia-wall: unknown inviscid-state {wmghost}')

        # Optional suppression of wall-tangential velocity oscillations: an
        # extra flux damp-k*rho*(u_t - l*du_t/dn) with l = slip-ratio times
        # the near-wall element height (= h/g_k for Gauss-Legendre points)
        wmdamp = cfg.getfloat(cfgsect, 'damp-k', 0.0)
        wmlfac = 0.0
        if wmdamp and wmint:
            raise ValueError('wm-adia-wall: damp-k requires sample = point')
        if wmdamp:
            p = elemap['hex'].basis.order
            gk = (1 + np.sort(np.polynomial.legendre.leggauss(p + 1)[0])[k - 1])/2
            wmlfac = cfg.getfloat(cfgsect, 'slip-ratio') / gk

        self._tplargs['wmlaw'] = wmlaw
        self._tplargs['wmc'] = wmc
        self._tplargs['wmghost'] = wmghost
        self._tplargs['wmdamp'] = wmdamp
        self._tplargs['wmlfac'] = wmlfac
        self._tplargs['wmdebug'] = debug
        self._tplargs['wmint'] = wmint

        # Taylor series of the mean LOG-EXP law, used for Y < wmy0
        kp, a, b, cc, d = (wmc[cn] for cn in ['kappa', 'A', 'B', 'C', 'D'])
        self._tplargs['wmy0'] = 0.3
        self._tplargs['wmser'] = [
            ((-1)**(n + 1)*math.factorial(n - 1)*kp**(n - 1) - a*(-1/b)**n
             - cc*(-1/d)**n) / math.factorial(n + 1) for n in range(1, 13)
        ]

        # Solution points of the normal column above each flux point (nearest
        # first) and their wall distances, ordered exactly as the face arrays
        # (cf. _get_inter_arrays)
        tab, tref = self._sample_table(elemap['hex'], k)
        cols = range(tab.shape[1]) if wmint else [k - 1]
        parts, reorder = [], []
        for etype, fidx, eidxs, idx in lhs.foreach():
            ele = elemap[etype]
            fpts = ele.srtd_face_fpts[fidx][eidxs]
            ec = eidxs[:, None]
            xw = ele.plocfpts[fpts, ec]
            nw = ele._pnorm_fpts[fpts, ec]
            nw = nw / np.linalg.norm(nw, axis=-1)[..., None]

            uis, hs = [], []
            for j in cols:
                ui = tab[fpts, j]
                xu = ele.ploc_at_np('upts')[ui, :, ec]
                h = -np.einsum('...i,...i', xu - xw, nw)

                if np.any(h <= 0):
                    raise RuntimeError('wm-adia-wall: non-positive sample '
                                       'height')

                uis.append(ui.ravel())
                hs.append(h.ravel())

            parts.append((np.array(uis), np.repeat(eidxs, fpts.shape[1]),
                          np.array(hs)))
            reorder.append(np.repeat(idx, ele.nfacefpts[fidx]))

        ro = np.argsort(np.concatenate(reorder), kind='stable')[self._perm]
        uidxs = np.concatenate([p[0] for p in parts], axis=1)[:, ro]
        cidx = np.concatenate([p[1] for p in parts])[ro]
        hcol = np.concatenate([p[2] for p in parts], axis=1)[:, ro]

        if wmint and tab.shape[1] > 9:
            raise ValueError('wm-adia-wall: column-mean supports order <= 8')

        if wmint:
            # Wall-element height from each column point, h_j = H t_j/2 with
            # t_j the reference distance in [0, 2]; the column must be affine
            Hs = 2*hcol / tref[:, None]
            hs = Hs.mean(axis=0)
            if np.abs(Hs/hs - 1).max() > 1e-8:
                raise RuntimeError('wm-adia-wall: column-mean requires '
                                   'elements affine along the wall normal')

            # Interpolatory weights of the column nodes over [0, 2], halved
            V = np.vander(tref, increasing=True).T
            m = np.array([2**(i + 1)/(i + 1) for i in range(len(tref))])
            self._tplargs['wmw'] = [float(w) for w in np.linalg.solve(V, m)/2]
        else:
            self._tplargs['wmw'] = [1.0]
            hs = hcol[0]

        # One view of the sampled state per solution register bank.  Views
        # need committed storage, but the banks' extents are otherwise only
        # committed after the kernels are generated; so commit them here
        banks = elemap['hex'].scal_upts
        bids = {id(b) for b in banks}
        for name, ext in list(be._extents.items()):
            if any(id(o) in bids for o, _ in ext._primary._pending):
                be.commit_extent(ext)
                del be._extents[name]

        self._uwms = [
            [be.view(np.full(len(cidx), b.mid), uidx, cidx,
                     vshape=(self.nvars,)) for uidx in uidxs]
            for b in banks
        ]
        self._hwm = be.const_matrix(hs[None, :])

        self._uwnames = ['uwms'] + [f'uwms{j}' for j in range(1, len(uidxs))]
        extrns = self._external_args | {
            n: f'in view fpdtype_t[{self.nvars}]' for n in self._uwnames
        } | {'hwm': 'in fpdtype_t'}
        extvals = self._external_vals | {'hwm': self._hwm}

        if debug:
            nd = self.nvars + self.ndims
            self.wmdbg = be.matrix((nd, self.ninterfpts))
            extrns['wmdbg'] = f'out fpdtype_t[{nd}]'
            extvals['wmdbg'] = self.wmdbg

        self.kernels['comm_flux'] = lambda uin: self._be.kernel(
            'bccflux', tplargs=self._tplargs, dims=[self.ninterfpts],
            extrns=extrns, ul=self.scal_lhs, gradul=self._vect_lhs,
            nl=self._pnorm_lhs, artvisc=self.artvisc,
            **dict(zip(self._uwnames, self._uwms[uin])), **extvals
        )

    @staticmethod
    def _sample_table(ele, k):
        basis = ele.basis
        upts, fpts = np.array(basis.upts), np.array(basis.fpts)
        tab = np.empty((len(fpts), basis.order + 1), dtype=int)
        tref = None

        for (kind, proj, norm), ffpts in zip(basis.faces, basis.facefpts):
            nref = np.array(norm, dtype=float)

            for i in ffpts:
                d = upts - fpts[i]
                tang = d - np.outer(d @ nref, nref)
                col = np.where(np.linalg.norm(tang, axis=1) < 1e-10)[0]

                if len(col) != basis.order + 1:
                    raise ValueError('wm-adia-wall requires solution points '
                                     'aligned with face flux points')
                if not 1 <= k <= len(col):
                    raise ValueError(f'sample-point must be in '
                                     f'1..{len(col)}')

                # Order the column by distance from the face
                t = -(d[col] @ nref)
                o = np.argsort(t)
                tab[i] = col[o]

                if tref is None:
                    tref = t[o]
                elif np.abs(t[o] - tref).max() > 1e-10:
                    raise ValueError('wm-adia-wall: columns differ')

        return tab, tref
