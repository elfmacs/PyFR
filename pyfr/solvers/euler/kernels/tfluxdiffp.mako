<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.ecflux'/>

<%
    n1 = len(w1)
    nlines = len(vix[0])
    arr1 = lambda a: '{' + ', '.join(repr(float(x)) for x in a) + '}'
    arr2 = lambda a: '{' + ', '.join(arr1(r) for r in a) + '}'
    iarr2 = lambda a: '{' + ', '.join('{' + ', '.join(str(int(x)) for x in r) + '}' for r in a) + '}'
%>

// Point-parallel variant of tfluxdiff (one thread per solution point and
// element) for GPUs.  pinfo holds, for each direction, the position of the
// point along its line and the line index.  Two-point fluxes are always
// evaluated with the lower line index first and the summation order is the
// same as in tfluxdiff, so both kernels evaluate the same expression
<%pyfr:kernel name='tfluxdiffp' ndim='2'
              u='in broadcast-col fpdtype_t[${str(nupts)}][${str(nvars)}]'
              uf='in broadcast-col fpdtype_t[${str(nfpts)}][${str(nvars)}]'
              smats='in broadcast-col fpdtype_t[${str(ndims)}][${str(ndims)}]'
              pinfo='in broadcast-row fpdtype_t[${str(2*ndims)}]'
              div='out fpdtype_t[${str(nvars)}]'>
    const fpdtype_t S0[${n1}][${n1}] = ${arr2(S0)};
    const fpdtype_t Vf[2][${n1}] = ${arr2(Vf)};
    const fpdtype_t rw[${n1}] = ${arr1(1/w1)};

    fpdtype_t dv[${nvars}];
    for (int k = 0; k < ${nvars}; k++)
        dv[k] = 0;

% for d in range(ndims):
    {
        const int vix[${nlines}][${n1}] = ${iarr2(vix[d])};
        const int fix[${nlines}][2] = ${iarr2(fix[d])};
        fpdtype_t sd[] = ${pyfr.array(f'smats[{d}][{{i}}]', i=ndims)};

        int a = (int) pinfo[${2*d}], l = (int) pinfo[${2*d + 1}];
        fpdtype_t acc[${nvars}], ua[${nvars}], ub[${nvars}], fs[${nvars}];

        for (int k = 0; k < ${nvars}; k++)
        {
            acc[k] = 0;
            ua[k] = u[vix[l][a]][k];
        }

        // Volume-volume coupling
        for (int b = 0; b < ${n1}; b++)
        {
            if (b == a)
                continue;

            for (int k = 0; k < ${nvars}; k++)
                ub[k] = u[vix[l][b]][k];

            if (b < a)
            {
                ${pyfr.expand('ec_flux', 'ub', 'ua', 'sd', 'fs')};
                for (int k = 0; k < ${nvars}; k++)
                    acc[k] -= 2*S0[b][a]*fs[k];
            }
            else
            {
                ${pyfr.expand('ec_flux', 'ua', 'ub', 'sd', 'fs')};
                for (int k = 0; k < ${nvars}; k++)
                    acc[k] += 2*S0[a][b]*fs[k];
            }
        }

        // Volume-face coupling with the entropy projected face states
        for (int m = 0; m < 2; m++)
        {
            fpdtype_t bm = (m == 0) ? -1 : 1;
            fpdtype_t um[${nvars}], gm[${nvars}];

            for (int k = 0; k < ${nvars}; k++)
            {
                um[k] = uf[fix[l][m]][k];
                gm[k] = 0;
            }

            for (int b = 0; b < ${n1}; b++)
            {
                for (int k = 0; k < ${nvars}; k++)
                    ub[k] = u[vix[l][b]][k];

                ${pyfr.expand('ec_flux', 'ub', 'um', 'sd', 'fs')};

                for (int k = 0; k < ${nvars}; k++)
                {
                    if (b == a)
                        acc[k] += bm*Vf[m][a]*fs[k];
                    gm[k] += Vf[m][b]*fs[k];
                }
            }

            for (int k = 0; k < ${nvars}; k++)
                acc[k] -= bm*Vf[m][a]*gm[k];
        }

        for (int k = 0; k < ${nvars}; k++)
            dv[k] += rw[a]*acc[k];
    }
% endfor

    for (int k = 0; k < ${nvars}; k++)
        div[k] = dv[k];
</%pyfr:kernel>
