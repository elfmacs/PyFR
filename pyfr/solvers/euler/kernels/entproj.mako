<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.ecflux'/>

// Map interpolated entropy variables back to conservative variables (the
// entropy projection); a non-negative v[nvars - 1] = -rho/p has no
// admissible state, so the point is explicitly poisoned with NaN.  A copy
// is kept in uc since the interface kernels overwrite u with fluxes
<%pyfr:kernel name='entproj' ndim='2'
              u='inout fpdtype_t[${str(nvars)}]'
              uc='out fpdtype_t[${str(nvars)}]'>
    fpdtype_t v[] = ${pyfr.array('u[{i}]', i=nvars)};

    if (v[${nvars - 1}] < 0)
    {
        ${pyfr.expand('ec_ent_to_cons', 'v', 'u')};
    }
    else
    {
% for i in range(nvars):
        u[${i}] = nan("");
% endfor
    }

% for i in range(nvars):
    uc[${i}] = u[${i}];
% endfor
</%pyfr:kernel>
