<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.ecflux'/>

// Entropy conservative interface flux (no dissipation); only meaningful
// together with [solver] volume-flux = entropy-conservative, where the
// face states are entropy projected
<%pyfr:macro name='rsolve' params='ul, ur, n, nf'>
    ${pyfr.expand('ec_flux', 'ul', 'ur', 'n', 'nf')};
</%pyfr:macro>
