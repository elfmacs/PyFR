import numpy as np

from pyfr.exprs import resolve_aux
from pyfr.solvers.base.elements import ExportableField
from pyfr.solvers.baseadvec import BaseAdvectionElements


class BaseFluidElements:
    eos_kernel_module = 'pyfr.solvers.euler.kernels.eos'

    @staticmethod
    def privars(ndims, cfg):
        if ndims == 2:
            return ['rho', 'u', 'v', 'p']
        elif ndims == 3:
            return ['rho', 'u', 'v', 'w', 'p']

    @staticmethod
    def convars(ndims, cfg):
        if ndims == 2:
            return ['rho', 'rhou', 'rhov', 'E']
        elif ndims == 3:
            return ['rho', 'rhou', 'rhov', 'rhow', 'E']

    dualcoeffs = convars

    @staticmethod
    def visvars(ndims, cfg):
        if ndims == 2:
            return {
                'density': ['rho'],
                'velocity': ['u', 'v'],
                'pressure': ['p']
            }
        elif ndims == 3:
            return {
                'density': ['rho'],
                'velocity': ['u', 'v', 'w'],
                'pressure': ['p']
            }

    @staticmethod
    def auxvars(ndims, cfg):
        c = cfg.items_as('constants', float)
        g = c['gamma']

        vm2 = 'u*u + v*v' if ndims == 2 else 'u*u + v*v + w*w'
        aux = {
            'cpT': f'{g}*p/({g - 1}*rho)',
            'mach': f'sqrt(({vm2})*rho/({g}*p))',
            'p0': f'p*pow(1.0 + {0.5*(g - 1)}*mach*mach, {g / (g - 1)})',
            'cpT0': f'cpT*(1.0 + {0.5*(g - 1)}*mach*mach)'
        }

        # Gradients of cp*T via the equation of state
        k = g / (g - 1)
        for x in 'xyz'[:ndims]:
            aux[f'grad_cpT_{x}'] = (f'{k}*(grad_p_{x}*rho - '
                                    f'p*grad_rho_{x})/(rho*rho)')

        # Temperatures are normalised by the wall state cp*Tw
        if 'cpTw' in c:
            aux['T'] = f'cpT/{c['cpTw']}'
            aux['T0'] = f'cpT0/{c['cpTw']}'

            for x in 'xyz'[:ndims]:
                aux[f'grad_T_{x}'] = f'grad_cpT_{x}/{c['cpTw']}'

        return resolve_aux(aux)

    @staticmethod
    def pri_to_con(pris, cfg):
        rho, p = pris[0], pris[-1]

        # Multiply velocity components by rho
        rhovs = [rho*c for c in pris[1:-1]]

        # Compute the energy
        gamma = cfg.getfloat('constants', 'gamma')
        E = p/(gamma - 1) + 0.5*rho*sum(c*c for c in pris[1:-1])

        return [rho, *rhovs, E]

    @staticmethod
    def con_to_pri(cons, cfg):
        rho, E = cons[0], cons[-1]

        # Divide momentum components by rho
        vs = [rhov/rho for rhov in cons[1:-1]]

        # Compute the pressure
        gamma = cfg.getfloat('constants', 'gamma')
        p = (gamma - 1)*(E - 0.5*rho*sum(v**2 for v in vs))

        return [rho, *vs, p]

    @staticmethod
    def diff_con_to_pri(cons, diff_cons, cfg):
        rho, *rhouvw = cons[:-1]
        diff_rho, *diff_rhouvw, diff_E = diff_cons

        # Divide momentum components by ρ
        uvw = [rhov / rho for rhov in rhouvw]

        # Velocity gradients: ∂u⃗ = 1/ρ·[∂(ρu⃗) - u⃗·∂ρ]
        diff_uvw = [(diff_rhov - v*diff_rho) / rho
                    for diff_rhov, v in zip(diff_rhouvw, uvw)]

        # Pressure gradient: ∂p = (γ - 1)·[∂E - 1/2*(u⃗·∂(ρu⃗) + ρu⃗·∂u⃗)]
        gamma = cfg.getfloat('constants', 'gamma')
        diff_p = diff_E - 0.5*(sum(u*dru for u, dru in zip(uvw, diff_rhouvw)) +
                               sum(ru*du for ru, du in zip(rhouvw, diff_uvw)))
        diff_p *= gamma - 1

        return [diff_rho, *diff_uvw, diff_p]

    @staticmethod
    def con_is_admissible(cons, cfg):
        rho, E = cons[0], cons[-1]

        # Given a positive density p > 0 reduces to 2*rho*E > |rho*u|^2
        msq = sum(rhov*rhov for rhov in cons[1:-1])

        return (rho > 0) & (2*rho*E > msq)

    @staticmethod
    def stats_tables(cfg):
        return ['pyfr.solvers.euler']

    @classmethod
    def eos_tplargs(cls, ndims, cfg):
        return {
            'ndims': ndims,
            'nvars': len(cls.convars(ndims, cfg)),
            'c': cfg.items_as('constants', float),
        }

    def set_backend(self, *args, **kwargs):
        super().set_backend(*args, **kwargs)

        # Register wavespeed kernel for CFL-based time stepping
        self._be.pointwise.register('pyfr.solvers.euler.kernels.wavespeed')

    def init_wavespeed(self):
        self._wspd = self._be.matrix((1, self.neles), tags={'align'})
        self.kernels['wavespeed'] = lambda uin: self._wavespeed_kernel(uin)

        def cfl_getter():
            with np.errstate(divide='ignore'):
                wspd = (2*self.basis.order + 1)*self._wspd.get()[0]
                return np.nan_to_num(1/wspd, posinf=0.0)

        self.export_fields.append(ExportableField(
            name='dt-cfl', shape=(), getter=cfl_getter
        ))
        return self._wspd

    def _wavespeed_kernel(self, uin):
        r, s = self.mesh_regions, self._slice_mat

        tplargs = {
            'ndims': self.ndims,
            'nvars': self.nvars,
            'nverts': len(self.basis.linspts),
            'c': self.cfg.items_as('constants', float),
            'jac_exprs': self.basis.jac_exprs
        }

        wkerns = []
        for rgn in ('curved', 'linear'):
            if rgn not in r:
                continue

            if rgn == 'curved':
                kw = {'smats': self.curved_smat_at('upts'),
                      'rcpdjac': self.rcpdjac_at('upts', 'curved')}
            else:
                kw = {'verts': self.ploc_at('linspts', 'linear'),
                      'upts': self.upts}

            wkerns.append(self._be.kernel(
                'wavespeed', tplargs=tplargs | {'ktype': rgn},
                dims=[self.nupts, r[rgn]], u=s(self.scal_upts[uin], rgn),
                wspd=s(self._wspd, rgn), **kw
            ))

        if len(wkerns) > 1:
            return self._be.unordered_meta_kernel(wkerns)
        else:
            return wkerns[0]


    def _set_ec_kernels(self, viscous=False):
        from pyfr.backends.base import NullKernel
        from pyfr.quadrules import get_quadrule

        cfg, basis = self.cfg, self.basis
        vflux = cfg.get('solver', 'volume-flux')
        if vflux != 'entropy-conservative':
            raise ValueError(f'Invalid volume-flux {vflux!r}')

        # Restrictions of the first implementation
        if self.basis.name != 'hex':
            raise ValueError('volume-flux requires hexahedral elements')
        if cfg.get('solver-elements-hex', 'soln-pts') != 'gauss-legendre':
            raise ValueError('volume-flux requires gauss-legendre soln-pts')
        if cfg.get('solver-interfaces-quad', 'flux-pts') != 'gauss-legendre':
            raise ValueError('volume-flux requires gauss-legendre flux-pts')
        if self.antialias:
            raise ValueError('volume-flux is incompatible with anti-aliasing')
        if cfg.get('solver', 'shock-capturing', 'none') != 'none':
            raise ValueError('volume-flux is incompatible with shock-capturing')
        if self._be.fpdtype != np.float64:
            raise ValueError('volume-flux is only validated in double precision')

        # Metric terms must be constant within each element (affine)
        smats = self.smat_at_np('upts')
        rcpdjac = self.rcpdjac_at_np('upts')
        if (np.abs(smats - smats[:, :1]).max() > 1e-10*np.abs(smats).max() or
            np.abs(rcpdjac - rcpdjac[:1]).max() > 1e-10*np.abs(rcpdjac).max()):
            raise ValueError('volume-flux requires affine elements')

        # One dimensional operators on the Gauss points
        n1 = basis.order + 1
        q = get_quadrule('line', 'gauss-legendre', n1)
        x1, w1 = q.pts.ravel(), q.wts
        D1 = np.zeros((n1, n1))
        for i in range(n1):
            for j in range(n1):
                if i != j:
                    D1[i, j] = (np.prod([x1[i] - x1[k] for k in range(n1)
                                         if k not in (i, j)]) /
                                np.prod([x1[j] - x1[k] for k in range(n1)
                                         if k != j]))
            D1[i, i] = sum(1/(x1[i] - x1[k]) for k in range(n1) if k != i)

        Vf = np.array([[np.prod([(s - x1[k])/(x1[j] - x1[k])
                                 for k in range(n1) if k != j])
                        for j in range(n1)] for s in (-1.0, 1.0)])
        B = np.diag([-1.0, 1.0])
        S0 = np.diag(w1) @ D1 - 0.5*Vf.T @ B @ Vf

        # Lines of solution points and their end flux points per direction
        upts, fpts, nfpts = basis.upts, basis.fpts, basis.norm_fpts
        pos = lambda c: int(np.argmin(np.abs(x1 - c)))
        vix, fix = [], []
        for d in range(self.ndims):
            o = [e for e in range(self.ndims) if e != d]
            lines = {}
            for k, xk in enumerate(upts):
                key = tuple(pos(xk[e]) for e in o)
                lines.setdefault(key, [None]*n1)[pos(xk[d])] = k

            ends = {}
            for f, xf in enumerate(fpts):
                if abs(abs(nfpts[f, d]) - 1) < 1e-12:
                    key = tuple(pos(xf[e]) for e in o)
                    ends.setdefault(key, [None, None])[int(nfpts[f, d] > 0)] = f

            keys = sorted(lines)
            vix.append([lines[k] for k in keys])
            fix.append([ends[k] for k in keys])

            if (len(keys) != n1**2 or
                any(None in r for r in vix[-1] + fix[-1])):
                raise RuntimeError('Unable to identify solution point lines')

        tplargs = {
            'ndims': self.ndims, 'nvars': self.nvars,
            'nupts': self.nupts, 'nfpts': self.nfpts,
            'c': self.cfg.items_as('constants', float),
            'S0': S0, 'Vf': Vf, 'w1': w1, 'vix': vix, 'fix': fix
        }

        be = self._be
        be.pointwise.register('pyfr.solvers.euler.kernels.entvar')
        be.pointwise.register('pyfr.solvers.euler.kernels.entproj')

        be.pointwise.register('pyfr.solvers.euler.kernels.tfluxdiff')

        # Per-element (constant) metric terms and scratch space
        self._ec_smats = be.const_matrix(
            np.ascontiguousarray(smats[:, 0]), tags={'align'}
        )
        self._ec_vupts = be.matrix((self.nupts, self.nvars, self.neles),
                                   tags={'align'})
        self._ec_div = be.matrix((self.nupts, self.nvars, self.neles),
                                 tags={'align'})
        self._ec_ufpts = be.matrix((self.nfpts, self.nvars, self.neles),
                                   tags={'align'})

        # Entropy projected interpolation to the flux points
        def disu(uin):
            return be.ordered_meta_kernel([
                be.kernel('entvar', tplargs=tplargs,
                          dims=[self.nupts, self.neles],
                          u=self.scal_upts[uin], v=self._ec_vupts),
                be.kernel('mul', self.opmat('M0'), self._ec_vupts,
                          out=self._scal_fpts),
                be.kernel('entproj', tplargs=tplargs,
                          dims=[self.nfpts, self.neles], u=self._scal_fpts,
                          uc=self._ec_ufpts)
            ])

        self.kernels['disu'] = disu

        # Without viscous terms there is nothing left for tdisf/tdivtpcorf;
        # with them these kernels handle the viscous flux only
        if not viscous:
            self.kernels['tdisf'] = lambda uin: NullKernel()
            self.kernels['tdivtpcorf'] = lambda fout: NullKernel()
        self.kernels['tfluxdiff'] = lambda uin: be.kernel(
            'tfluxdiff', tplargs=tplargs, dims=[self.neles],
            u=self.scal_upts[uin], uf=self._ec_ufpts,
            smats=self._ec_smats, div=self._ec_div
        )
        if not viscous:
            self.kernels['tdivtconf'] = lambda fout: be.kernel(
                'mul', self.opmat('M3'), self._scal_fpts,
                out=self.scal_upts[fout], beta=0.0
            )

        # Add the flux differencing divergence in negdivconf
        self._srctplargs['esdiv'] = True
        self.set_external('esdiv', f'in fpdtype_t[{self.nvars}]',
                          value=self._ec_div)


class EulerElements(BaseFluidElements, BaseAdvectionElements):
    def set_backend(self, *args, **kwargs):
        super().set_backend(*args, **kwargs)

        # Can elide interior flux calculations at p = 0
        if self.basis.order == 0:
            return

        # Register our flux kernels
        self._be.pointwise.register('pyfr.solvers.euler.kernels.tflux')

        # Template parameters for the flux kernels
        tplargs = {
            'ndims': self.ndims,
            'nvars': self.nvars,
            'nverts': len(self.basis.linspts),
            'c': self.cfg.items_as('constants', float),
            'jac_exprs': self.basis.jac_exprs
        }

        # Helpers
        tdisf = []
        c, l = 'curved', 'linear'
        r, s = self.mesh_regions, self._slice_mat
        slicedk = self._make_sliced_kernel

        if c in r and 'flux' not in self.antialias:
            tdisf.append(lambda uin: self._be.kernel(
                'tflux', tplargs=tplargs | {'ktype': 'curved'},
                dims=[self.nupts, r[c]], u=s(self.scal_upts[uin], c),
                f=s(self._vect_upts, c), smats=self.curved_smat_at('upts')
            ))
        elif c in r:
            tdisf.append(lambda: self._be.kernel(
                'tflux', tplargs=tplargs | {'ktype': 'curved'},
                dims=[self.nqpts, r[c]], u=s(self._scal_qpts, c),
                f=s(self._vect_qpts, c), smats=self.curved_smat_at('qpts')
            ))

        if l in r and 'flux' not in self.antialias:
            tdisf.append(lambda uin: self._be.kernel(
                'tflux', tplargs=tplargs | {'ktype': 'linear'},
                dims=[self.nupts, r[l]], u=s(self.scal_upts[uin], l),
                f=s(self._vect_upts, l), verts=self.ploc_at('linspts', l),
                upts=self.upts
            ))
        elif l in r:
            tdisf.append(lambda: self._be.kernel(
                'tflux', tplargs=tplargs | {'ktype': 'linear'},
                dims=[self.nqpts, r[l]], u=s(self._scal_qpts, l),
                f=s(self._vect_qpts, l), verts=self.ploc_at('linspts', l),
                upts=self.qpts
            ))

        if 'flux' not in self.antialias:
            self.kernels['tdisf'] = lambda uin: slicedk(k(uin) for k in tdisf)
        else:
            self.kernels['tdisf'] = lambda: slicedk(k() for k in tdisf)

        # Entropy conservative flux differencing replaces tdisf/tdivtpcorf
        if self.cfg.get('solver', 'volume-flux', 'standard') != 'standard':
            self._set_ec_kernels()
