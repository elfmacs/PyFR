<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.bcs.common'/>

## Supersonic inflow whose primitive state (rho, u, v, w, p) at each flux
## point is supplied by the host through the rcst external
<%pyfr:macro name='bc_rsolve_state' params='ul, nl, ur' externs='rcst'>
    ur[0] = rcst[0];
% for i in range(ndims):
    ur[${i + 1}] = rcst[0]*rcst[${i + 1}];
% endfor
    ur[${nvars - 1}] = rcst[${nvars - 1}]*${1/(c['gamma'] - 1)} +
                       0.5*(1.0/ur[0])*${pyfr.dot('ur[{i}]', i=(1, ndims + 1))};
</%pyfr:macro>

<%pyfr:alias name='bc_ldg_state' func='bc_rsolve_state'/>
<%pyfr:alias name='bc_ldg_grad_state' func='bc_common_grad_zero'/>
