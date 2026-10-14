<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%include file='pyfr.solvers.euler.kernels.rsolvers.${rsolver}'/>
<%include file='pyfr.solvers.navstokes.kernels.bcs.common'/>

// Ghost state for the inviscid Riemann solve: slip mirror (default) or,
// with inviscid-state = noslip, all velocity components reversed
<%pyfr:macro name='bc_rsolve_state' params='ul, nl, ur'>
% if wmghost == 'noslip':
    ur[0] = ul[0];
% for i in range(ndims):
    ur[${i + 1}] = -ul[${i + 1}];
% endfor
    ur[${nvars - 1}] = ul[${nvars - 1}];
% else:
    fpdtype_t nor = ${' + '.join(f'ul[{i + 1}]*nl[{i}]' for i in range(ndims))};
    ur[0] = ul[0];
% for i in range(ndims):
    ur[${i + 1}] = ul[${i + 1}] - 2*nor*nl[${i}];
% endfor
    ur[${nvars - 1}] = ul[${nvars - 1}];
% endif
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
% if wmlaw == 'linear':
% for i in range(ndims):
    tau[${i}] = ${c['mu']}*vt[${i}]/h;
% endfor
% elif wmlaw == 'log-exp':
<% k, a, b, cc, d = wmc['kappa'], wmc['A'], wmc['B'], wmc['C'], wmc['D'] %>
    // LOG-EXP law (Zhang et al., arXiv:2104.09054, eq. 10) solved for
    // u_tau with Newton on g = u_tau*F(y+) - |u_t|, linear-law start
    fpdtype_t nu = ${c['mu']}*rcprho;
    fpdtype_t us_t = fmax(sqrt(${pyfr.dot('vt[{i}]', i=ndims)}), ${wmc['umin']});
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
    // tau = rho_w u_tau^2 t, written so that it tends to zero with |u_t|
    fpdtype_t s = rhow*ut*ut/us_t;
% for i in range(ndims):
    tau[${i}] = s*vt[${i}];
% endfor
% elif wmlaw == 'fixed':
    // Diagnostic only: fixed magnitude along the sampled tangential velocity
    fpdtype_t us_t = fmax(sqrt(${pyfr.dot('vt[{i}]', i=ndims)}), ${wmc['umin']});
    fpdtype_t s = ${wmc['tau']}/us_t;
% for i in range(ndims):
    tau[${i}] = s*vt[${i}];
% endfor
% elif wmlaw == 'none':
    // Diagnostic only: zero wall shear stress
% for i in range(ndims):
    tau[${i}] = 0;
% endfor
% endif
</%pyfr:macro>

<%pyfr:macro name='bc_common_flux_state' params='ul, gradul, artviscl, nl, magnl'
              externs='uwms, hwm, wmdbg'>
    // Inviscid part: slip wall
    fpdtype_t ur[${nvars}];
    ${pyfr.expand('bc_rsolve_state', 'ul', 'nl', 'ur')};

    fpdtype_t ficomm[${nvars}];
    ${pyfr.expand('rsolve', 'ul', 'ur', 'nl', 'ficomm')};

    // Viscous part: momentum flux into the wall from the wall model
    fpdtype_t tau[${ndims}];
    ${pyfr.expand('bc_wm_tau', 'uwms', 'hwm', 'nl', 'ul[0]', 'tau')};

% if wmdamp:
    // Oscillation suppression: penalise the mismatch between the wall-face
    // tangential velocity and l times its normal derivative into the fluid
    // (nl points out of the fluid, so -d/dn_in = nl.grad)
    fpdtype_t pen[${ndims}];
    {
        fpdtype_t rcpr = 1.0/ul[0], lw = ${wmlfac}*hwm;
        fpdtype_t uf[${ndims}], gn[${ndims}], ufn = 0, gnn = 0;
% for i in range(ndims):
        uf[${i}] = rcpr*ul[${i + 1}];
        gn[${i}] = rcpr*(${' + '.join(f'nl[{j}]*(gradul[{j}][{i + 1}] - uf[{i}]*gradul[{j}][0])' for j in range(ndims))});
        ufn += uf[${i}]*nl[${i}];
        gnn += gn[${i}]*nl[${i}];
% endfor
% for i in range(ndims):
        pen[${i}] = ${wmdamp}*ul[0]*(uf[${i}] - ufn*nl[${i}] + lw*(gn[${i}] - gnn*nl[${i}]));
% endfor
    }
% endif

    ul[0] = magnl*ficomm[0];
% for i in range(ndims):
% if wmdamp:
    ul[${i + 1}] = magnl*(ficomm[${i + 1}] + tau[${i}] + pen[${i}]);
% else:
    ul[${i + 1}] = magnl*(ficomm[${i + 1}] + tau[${i}]);
% endif
% endfor
    ul[${nvars - 1}] = magnl*ficomm[${nvars - 1}];

% if wmdebug:
    // Debug output: sampled state followed by the wall-model tau
% for i in range(nvars):
    wmdbg[${i}] = uwms[${i}];
% endfor
% for i in range(ndims):
    wmdbg[${nvars + i}] = tau[${i}];
% endfor
% endif
</%pyfr:macro>
