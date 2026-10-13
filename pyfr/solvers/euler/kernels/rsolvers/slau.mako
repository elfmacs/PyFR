<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.flux'/>

<%pyfr:macro name='rsolve' params='ul, ur, n, nf'>
    // Primitive quantities and normal velocities
    fpdtype_t fl[${ndims}][${nvars}], fr[${ndims}][${nvars}];
    fpdtype_t vl[${ndims}], vr[${ndims}];
    fpdtype_t pl, pr;

    ${pyfr.expand('inviscid_flux', 'ul', 'fl', 'pl', 'vl')};
    ${pyfr.expand('inviscid_flux', 'ur', 'fr', 'pr', 'vr')};

    fpdtype_t unl = ${pyfr.dot('n[{i}]', 'vl[{i}]', i=ndims)};
    fpdtype_t unr = ${pyfr.dot('n[{i}]', 'vr[{i}]', i=ndims)};
    fpdtype_t al = sqrt(${c['gamma']}*pl/ul[0]);
    fpdtype_t ar = sqrt(${c['gamma']}*pr/ur[0]);
    fpdtype_t af = 0.5*(al + ar);
    fpdtype_t ml = unl/af, mr = unr/af;

    // SLAU all-speed Mach correction
    fpdtype_t v2 = 0.5*(${pyfr.dot('vl[{i}]', i=ndims)}
                       + ${pyfr.dot('vr[{i}]', i=ndims)});
    fpdtype_t mt = fmin(1.0, sqrt(v2)/af);
    fpdtype_t chi = (1.0 - mt)*(1.0 - mt);

    // Normal-velocity dissipation and pressure-difference correction
    fpdtype_t frho = -fmax(fmin(ml, 0.0), -1.0)
                      *fmin(fmax(mr, 0.0), 1.0);
    fpdtype_t vnmag = (ul[0]*fabs(unl) + ur[0]*fabs(unr))
                    / (ul[0] + ur[0]);
    fpdtype_t vnmagl = (1.0 - frho)*vnmag + frho*fabs(unl);
    fpdtype_t vnmagr = (1.0 - frho)*vnmag + frho*fabs(unr);
    fpdtype_t mdot = 0.5*(ul[0]*(unl + vnmagl)
                         + ur[0]*(unr - vnmagr)
                         - (chi/af)*(pr - pl));

    // Pressure splitting
    fpdtype_t betal, betar;
    if (fabs(ml) <= 1.0)
        betal = 0.25*(2.0 - ml)*(ml + 1.0)*(ml + 1.0);
    else
        betal = (ml >= 0.0) ? 1.0 : 0.0;

    if (fabs(mr) <= 1.0)
        betar = 0.25*(2.0 + mr)*(mr - 1.0)*(mr - 1.0);
    else
        betar = (mr >= 0.0) ? 0.0 : 1.0;

    fpdtype_t pface = 0.5*(pl + pr) + 0.5*(betal - betar)*(pl - pr)
                    + (1.0 - chi)*(betal + betar - 1.0)*0.5*(pl + pr);

    // Upwind convective transport plus the pressure contribution
    fpdtype_t mdotabs = fabs(mdot);
    nf[0] = mdot;
% for i in range(ndims):
    nf[${i + 1}] = 0.5*(mdot*(vl[${i}] + vr[${i}])
                         + mdotabs*(vl[${i}] - vr[${i}]))
                   + pface*n[${i}];
% endfor

    fpdtype_t hl = (ul[${nvars - 1}] + pl)/ul[0];
    fpdtype_t hr = (ur[${nvars - 1}] + pr)/ur[0];
    nf[${nvars - 1}] = 0.5*(mdot*(hl + hr) + mdotabs*(hl - hr));
</%pyfr:macro>
