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

// Entropy conservative flux differencing on Gauss points with decoupled
// SBP operators (Chan, Del Rey Fernandez and Carpenter, arXiv:1809.01178,
// eqs. (9) and (18)), applied line by line; writes the reference-space
// divergence excluding the interface term, which tdivtconf supplies
<%pyfr:kernel name='tfluxdiff' ndim='1'
              u='in fpdtype_t[${str(nupts)}][${str(nvars)}]'
              uf='in fpdtype_t[${str(nfpts)}][${str(nvars)}]'
              smats='in fpdtype_t[${str(ndims)}][${str(ndims)}]'
              div='out fpdtype_t[${str(nupts)}][${str(nvars)}]'>
    const fpdtype_t S0[${n1}][${n1}] = ${arr2(S0)};
    const fpdtype_t Vf[2][${n1}] = ${arr2(Vf)};
    const fpdtype_t rw[${n1}] = ${arr1(1/w1)};

    for (int i = 0; i < ${nupts}; i++)
        for (int k = 0; k < ${nvars}; k++)
            div[i][k] = 0;

% for d in range(ndims):
    {
        const int vix[${nlines}][${n1}] = ${iarr2(vix[d])};
        const int fix[${nlines}][2] = ${iarr2(fix[d])};
        fpdtype_t sd[] = ${pyfr.array(f'smats[{d}][{{i}}]', i=ndims)};

        for (int l = 0; l < ${nlines}; l++)
        {
            fpdtype_t acc[${n1}][${nvars}];
            fpdtype_t ua[${nvars}], ub[${nvars}], fs[${nvars}];

            for (int a = 0; a < ${n1}; a++)
                for (int k = 0; k < ${nvars}; k++)
                    acc[a][k] = 0;

            // Volume-volume coupling: 2 sum_b S0[a][b] f_S(u_a, u_b)
            for (int a = 0; a < ${n1}; a++)
            {
                for (int k = 0; k < ${nvars}; k++)
                    ua[k] = u[vix[l][a]][k];

                for (int b = a + 1; b < ${n1}; b++)
                {
                    for (int k = 0; k < ${nvars}; k++)
                        ub[k] = u[vix[l][b]][k];

                    ${pyfr.expand('ec_flux', 'ua', 'ub', 'sd', 'fs')};

                    for (int k = 0; k < ${nvars}; k++)
                    {
                        acc[a][k] += 2*S0[a][b]*fs[k];
                        acc[b][k] -= 2*S0[a][b]*fs[k];
                    }
                }
            }

            // Volume-face coupling with the entropy projected face states
            for (int m = 0; m < 2; m++)
            {
                fpdtype_t bm = (m == 0) ? -1 : 1;
                fpdtype_t gm[${nvars}];

                for (int k = 0; k < ${nvars}; k++)
                {
                    ub[k] = uf[fix[l][m]][k];
                    gm[k] = 0;
                }

                for (int a = 0; a < ${n1}; a++)
                {
                    for (int k = 0; k < ${nvars}; k++)
                        ua[k] = u[vix[l][a]][k];

                    ${pyfr.expand('ec_flux', 'ua', 'ub', 'sd', 'fs')};

                    for (int k = 0; k < ${nvars}; k++)
                    {
                        acc[a][k] += bm*Vf[m][a]*fs[k];
                        gm[k] += Vf[m][a]*fs[k];
                    }
                }

                for (int a = 0; a < ${n1}; a++)
                    for (int k = 0; k < ${nvars}; k++)
                        acc[a][k] -= bm*Vf[m][a]*gm[k];
            }

            for (int a = 0; a < ${n1}; a++)
                for (int k = 0; k < ${nvars}; k++)
                    div[vix[l][a]][k] += rw[a]*acc[a][k];
        }
    }
% endfor
</%pyfr:kernel>
