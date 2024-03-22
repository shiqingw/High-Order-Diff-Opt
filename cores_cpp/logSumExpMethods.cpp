#include <cmath>
#include <tuple>
#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>


std::tuple<double, xt::xarray<double>, xt::xarray<double>, xt::xarray<double>>
getLogSumExpDerivaties(const xt::xarray<double>& p, const xt::xarray<double>& A,
const xt::xarray<double>& b, const double kappa) {
    // Compute F(p) = log[sum(exp[k(A p + b)])/len(A)] + 1
    // p: input vector of dimension 2 or 3
    // A: real matrix of dimension N x dim(p)
    // b: real vector of dimension N

    int dim_p = p.shape()[0], dim_z = A.shape()[0];
    xt::xarray<double> z = kappa * (xt::linalg::dot(A, p) + b);
    double c = xt::amax(z)();
    z = xt::exp(z - c);
    double sum_z = xt::sum(z)();
    double F = log(sum_z) + c - log(double(dim_z)) + 1;

    xt::xarray<double> zT_A = xt::linalg::dot(z, A); // shape dim_p
    xt::xarray<double> F_dp = kappa * zT_A / sum_z; // shape dim_p

    xt::xarray<double> diag_z = xt::diag(z);
    xt::xarray<double> diag_z_A = xt::linalg::dot(diag_z, A);
    xt::xarray<double> AT_diag_z_A = xt::linalg::dot(xt::transpose(A), diag_z_A);
    xt::xarray<double> AT_z_zT_A = xt::linalg::outer(zT_A, zT_A);
    xt::xarray<double> F_dpdp = pow(kappa,2) * (AT_diag_z_A/sum_z - AT_z_zT_A/pow(sum_z,2));

    xt::xarray<double> F_dpdpdp = xt::zeros<double>({dim_p, dim_p, dim_p});
    xt::xarray<double> z_dp = kappa * diag_z_A;
    // part 1 of F_dpdpdp
    for (int i = 0; i < dim_p; ++i){
        xt::xarray<double> z_dp_i = xt::view(z_dp, xt::all(), i);
        xt::xarray<double> tmp = pow(kappa,2)*xt::linalg::dot(xt::transpose(A), xt::linalg::dot(xt::diag(z_dp_i) ,A))/sum_z;
        tmp -= pow(kappa,2) * xt::sum(z_dp_i)() * AT_diag_z_A/pow(sum_z,2);
        xt::view(F_dpdpdp, xt::all(), xt::all(), i) += tmp; 
    }

    // part 2 of F_dpdpdp
    xt::xarray<double> big = xt::zeros<double>({dim_z, dim_z, dim_z});
    xt::xarray<double> identity = xt::eye(dim_z);
    for (int i = 0; i < dim_z; ++i){
        xt::xarray<double> e_i = xt::view(identity, i, xt::all());
        xt::view(big, xt::all(), xt::all(), i) += xt::linalg::outer(e_i, z) + xt::linalg::outer(z, e_i);
    }

    for (int i = 0; i < dim_p; ++i){
        xt::xarray<double> z_dp_i = xt::view(z_dp, xt::all(), i);
        xt::xarray<double> tmp = xt::linalg::tensordot(xt::transpose(A), big, {1}, {0}); // shape dim_p x dim_z x dim_z
        tmp = xt::linalg::tensordot(tmp, A, {1}, {0}); // shape dim_p x dim_p x dim_z
        tmp = pow(kappa,2) * xt::linalg::tensordot(tmp, z_dp_i, {2}, {0}) / pow(sum_z,2); // shape dim_p x dim_p
        tmp -= 2 * pow(kappa,2) * xt::sum(z_dp_i)() * AT_z_zT_A / pow(sum_z,3);
        xt::view(F_dpdpdp, xt::all(), xt::all(), i) -= tmp;
    }

    return std::make_tuple(F, F_dp, F_dpdp, F_dpdpdp);
}

std::tuple<double, xt::xarray<double>, xt::xarray<double>, xt::xarray<double>> getGradientAndHessianEllipsoidAndLogSumExp(
    const xt::xarray<double>& a, const xt::xarray<double>& q, const xt::xarray<double>& D,
    const xt::xarray<double>& R, const xt::xarray<double>& B, const xt::xarray<double>& b){
        
    }