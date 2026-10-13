<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='flowrate' params='t, u, ploc, src' externs='frforce'>
    src[${fridx}] += frforce;
    src[${nvars - 1}] += frforce*u[${fridx}]/u[0];
</%pyfr:macro>
