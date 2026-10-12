<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.ecflux'/>

// Entropy stable interface flux: entropy conservative flux plus local
// Lax-Friedrichs dissipation with the wave speed estimate of rusanov.mako.
// With entropy projected face states this dissipates entropy since
// [[v]].[[u]] >= 0 for a convex entropy
<%pyfr:macro name='rsolve' params='ul, ur, n, nf'>
    ${pyfr.expand('ec_flux', 'ul', 'ur', 'n', 'nf')};

    fpdtype_t rl = ul[0], rr = ur[0];
    fpdtype_t nv = (${' + '.join(f'n[{i}]*ul[{i + 1}]' for i in range(ndims))})/rl
                 + (${' + '.join(f'n[{i}]*ur[{i + 1}]' for i in range(ndims))})/rr;
    fpdtype_t pl = ${c['gamma'] - 1}*(ul[${nvars - 1}]
                 - 0.5*(${pyfr.dot('ul[{i}]', i=(1, ndims + 1))})/rl);
    fpdtype_t pr = ${c['gamma'] - 1}*(ur[${nvars - 1}]
                 - 0.5*(${pyfr.dot('ur[{i}]', i=(1, ndims + 1))})/rr);

    // Estimate the maximum wave speed / 2
    fpdtype_t a = sqrt(${0.25*c['gamma']}*(pl + pr)/(rl + rr))
                + 0.25*fabs(nv);

% for i in range(nvars):
    nf[${i}] += a*(ul[${i}] - ur[${i}]);
% endfor
</%pyfr:macro>
