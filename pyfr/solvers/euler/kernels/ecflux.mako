<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

// Logarithmic mean (a - b)/(log(a) - log(b)) following Algorithm 1 of
// Ranocha et al., arXiv:2112.10517 (after Ismail and Roe 2009, App. B);
// the series threshold 1e-4 is the value given there for FP64
<%pyfr:macro name='ec_lnmean' params='a, b, out'>
    {
        fpdtype_t xi = a/b, fz = (xi - 1)/(xi + 1), uz = fz*fz;
        fpdtype_t Fz = (uz < 1e-4)
                     ? 1 + uz*(${1/3} + uz*(${1/5} + uz*${1/7}))
                     : log(xi)/(2*fz);
        out = (a + b)/(2*Fz);
    }
</%pyfr:macro>

// Chandrashekar entropy conservative two-point flux projected onto the
// (not necessarily unit) vector n; Gassner, Winters and Kopriva,
// arXiv:1604.06618, eqs. (3.18)-(3.20)
<%pyfr:macro name='ec_flux' params='ul, ur, n, f'>
    {
        fpdtype_t rl = ul[0], rr = ur[0];
        fpdtype_t vl[${ndims}], vr[${ndims}], va[${ndims}];
% for i in range(ndims):
        vl[${i}] = ul[${i + 1}]/rl;
        vr[${i}] = ur[${i + 1}]/rr;
        va[${i}] = 0.5*(vl[${i}] + vr[${i}]);
% endfor
        fpdtype_t ql = ${pyfr.dot('vl[{i}]', i=ndims)};
        fpdtype_t qr = ${pyfr.dot('vr[{i}]', i=ndims)};
        fpdtype_t pl = ${c['gamma'] - 1}*(ul[${nvars - 1}] - 0.5*rl*ql);
        fpdtype_t pr = ${c['gamma'] - 1}*(ur[${nvars - 1}] - 0.5*rr*qr);
        fpdtype_t bl = 0.5*rl/pl, br = 0.5*rr/pr;

        fpdtype_t rln, bln;
        ${pyfr.expand('ec_lnmean', 'rl', 'rr', 'rln')};
        ${pyfr.expand('ec_lnmean', 'bl', 'br', 'bln')};

        fpdtype_t ph = 0.5*(rl + rr)/(bl + br);
        fpdtype_t hh = ${1/(2*(c['gamma'] - 1))}/bln - 0.25*(ql + qr)
                     + ph/rln + ${pyfr.dot('va[{i}]', i=ndims)};
        fpdtype_t vn = ${pyfr.dot('n[{i}]', 'va[{i}]', i=ndims)};
        fpdtype_t mf = rln*vn;

        f[0] = mf;
% for i in range(ndims):
        f[${i + 1}] = mf*va[${i}] + ph*n[${i}];
% endfor
        f[${nvars - 1}] = mf*hh;
    }
</%pyfr:macro>

// Entropy variables (Gassner, Winters and Kopriva, eq. (3.30))
<%pyfr:macro name='ec_cons_to_ent' params='u, v'>
    {
        fpdtype_t rho = u[0], rcprho = 1/u[0];
        fpdtype_t q = rcprho*(${pyfr.dot('u[{i}]', i=(1, ndims + 1))});
        fpdtype_t p = ${c['gamma'] - 1}*(u[${nvars - 1}] - 0.5*q);
        fpdtype_t s = log(p) - ${c['gamma']}*log(rho);
        fpdtype_t rcpp = 1/p;

        v[0] = ${c['gamma']/(c['gamma'] - 1)} - s*${1/(c['gamma'] - 1)}
             - 0.5*q*rcpp;
% for i in range(ndims):
        v[${i + 1}] = u[${i + 1}]*rcpp;
% endfor
        v[${nvars - 1}] = -rho*rcpp;
    }
</%pyfr:macro>

// Inverse of ec_cons_to_ent; requires v[nvars - 1] = -rho/p < 0
<%pyfr:macro name='ec_ent_to_cons' params='v, u'>
    {
        fpdtype_t v5 = v[${nvars - 1}];
        fpdtype_t vel[${ndims}];
% for i in range(ndims):
        vel[${i}] = -v[${i + 1}]/v5;
% endfor
        fpdtype_t q = ${pyfr.dot('vel[{i}]', i=ndims)};
        fpdtype_t s = ${c['gamma']} - ${c['gamma'] - 1}*(v[0] - 0.5*v5*q);
        fpdtype_t rho = exp((s + log(-v5))*${1/(1 - c['gamma'])});
        fpdtype_t p = -rho/v5;

        u[0] = rho;
% for i in range(ndims):
        u[${i + 1}] = rho*vel[${i}];
% endfor
        u[${nvars - 1}] = p*${1/(c['gamma'] - 1)} + 0.5*rho*q;
    }
</%pyfr:macro>
