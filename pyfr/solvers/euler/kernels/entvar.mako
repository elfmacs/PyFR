<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.ecflux'/>

<%pyfr:kernel name='entvar' ndim='2'
              u='in fpdtype_t[${str(nvars)}]'
              v='out fpdtype_t[${str(nvars)}]'>
    ${pyfr.expand('ec_cons_to_ent', 'u', 'v')};
</%pyfr:kernel>
