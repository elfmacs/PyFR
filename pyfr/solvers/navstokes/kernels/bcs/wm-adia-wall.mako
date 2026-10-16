<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.euler.kernels.rsolvers.${rsolver}'/>
<%include file='pyfr.solvers.navstokes.kernels.bcs.common'/>

// Ghost state for the inviscid Riemann solve: slip mirror
<%pyfr:macro name='bc_rsolve_state' params='ul, nl, ur'>
    fpdtype_t nor = ${' + '.join(f'ul[{i + 1}]*nl[{i}]' for i in range(ndims))};
    ur[0] = ul[0];
% for i in range(ndims):
    ur[${i + 1}] = ul[${i + 1}] - 2*nor*nl[${i}];
% endfor
    ur[${nvars - 1}] = ul[${nvars - 1}];
</%pyfr:macro>

// Boundary state for the gradients: as slp-adia-wall (develop semantics,
// where bc_ldg_state returns the boundary value rather than a ghost)
<%pyfr:macro name='bc_ldg_state' params='ul, nl, ur'>
    fpdtype_t nor = ${' + '.join(f'ul[{i + 1}]*nl[{i}]' for i in range(ndims))};
    ur[0] = ul[0];
% for i in range(ndims):
    ur[${i + 1}] = ul[${i + 1}] - nor*nl[${i}];
% endfor
    ur[${nvars - 1}] = ul[${nvars - 1}] - (0.5/ul[0])*nor*nor;
</%pyfr:macro>

// Wall shear stress vector (acting on the wall) from the sampled state
<%pyfr:macro name='bc_wm_tau' params='us, h, nl, rhow, tau'>
    fpdtype_t rcprho = 1.0/us[0];
    fpdtype_t vn = rcprho*(${' + '.join(f'us[{i + 1}]*nl[{i}]' for i in range(ndims))});
    fpdtype_t vt[${ndims}];
% for i in range(ndims):
    vt[${i}] = rcprho*us[${i + 1}] - vn*nl[${i}];
% endfor

    // LOG-EXP law (Zhang et al., arXiv:2104.09054, eq. 10) solved for u_tau
    // by Newton from a linear-law start
<% k, a, b, cc, d = wmc['kappa'], wmc['A'], wmc['B'], wmc['C'], wmc['D'] %>
    fpdtype_t nu = ${c['mu']}*rcprho;
    fpdtype_t us_t = fmax(sqrt(${pyfr.dot('vt[{i}]', i=ndims)}), ${wmc['umin']});
% if wmint:
    // Column-mean form: U = u_tau G(Y), Y = H u_tau/nu, G(Y) the mean of the
    // LOG-EXP law over [0, Y] (analytic; Taylor series for Y < ${wmy0} to avoid
    // cancellation).  d(u_tau G)/du_tau = F(Y), so Newton divides by F
    fpdtype_t ut = sqrt(nu*us_t/(${wmser[0]}*h));
% for it in range(wmc['niters']):
    {
        fpdtype_t Y = h*ut/nu;
        fpdtype_t F = log1p(${k}*Y)*${1/k}
                    - ${a}*expm1(-Y*${1/b}) - ${cc}*expm1(-Y*${1/d});
        fpdtype_t G;
        if (Y < ${wmy0})
            G = ${'('*len(wmser)}${')*Y + '.join(str(cm) for cm in reversed(wmser))})*Y;
        else
            G = ((1 + ${k}*Y)*log1p(${k}*Y) - ${k}*Y)*${1/k**2}/Y
              + ${a}*(1 + ${b}*expm1(-Y*${1/b})/Y)
              + ${cc}*(1 + ${d}*expm1(-Y*${1/d})/Y);
        ut -= (ut*G - us_t)/F;
    }
% endfor
% else:
    // Point form: U = u_tau F(y+), y+ = h u_tau/nu; Newton on u_tau F - U
    fpdtype_t ut = sqrt(nu*us_t/h);
% for it in range(wmc['niters']):
    {
        fpdtype_t yp = h*ut/nu;
        fpdtype_t F = log1p(${k}*yp)*${1/k}
                    - ${a}*expm1(-yp*${1/b}) - ${cc}*expm1(-yp*${1/d});
        fpdtype_t dF = 1/(1 + ${k}*yp) + ${a/b}*exp(-yp*${1/b})
                     + ${cc/d}*exp(-yp*${1/d});
        ut -= (ut*F - us_t)/(F + yp*dF);
    }
% endfor
% endif
    // tau = rho_w u_tau^2 t, written so that it tends to zero with |u_t|
    fpdtype_t s = rhow*ut*ut/us_t;
% for i in range(ndims):
    tau[${i}] = s*vt[${i}];
% endfor
</%pyfr:macro>

<%pyfr:macro name='bc_common_flux_state' params='ul, gradul, artviscl, nl, magnl'
              externs='uwms, hwm'>
    // Inviscid part: slip wall
    fpdtype_t ur[${nvars}];
    ${pyfr.expand('bc_rsolve_state', 'ul', 'nl', 'ur')};

    fpdtype_t ficomm[${nvars}];
    ${pyfr.expand('rsolve', 'ul', 'ur', 'nl', 'ficomm')};

    // Viscous part: momentum flux into the wall from the wall model
    fpdtype_t tau[${ndims}];
    // Sampled state: a weighted sum over the normal column of solution
    // points above this flux point, the column mean (weights sum to one) or
    // a single point (weight one)
    fpdtype_t usm[${nvars}];
% for i in range(nvars):
    usm[${i}] = ${' + '.join(f'{w!r}*uwms[{j}][{i}]' for j, w in enumerate(wmw) if w)};
% endfor
    ${pyfr.expand('bc_wm_tau', 'usm', 'hwm', 'nl', 'ul[0]', 'tau')};

    ul[0] = magnl*ficomm[0];
% for i in range(ndims):
    ul[${i + 1}] = magnl*(ficomm[${i + 1}] + tau[${i}]);
% endfor
    ul[${nvars - 1}] = magnl*ficomm[${nvars - 1}];
</%pyfr:macro>
