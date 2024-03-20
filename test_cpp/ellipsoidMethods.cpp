#include <cmath>
#include <tuple>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>

#include "diffOptHelper.h"

/*
    We consider F(p) = (p - a)^T @ R @ D @ R.T @ (p - a) where a is a vector and
    R = [[[2*(qw**2+qx**2)-1, 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz), 2*(qw**2+qy**2)-1, 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 2*(qw**2+qz**2)-1]].
    D = diag(D11, D22, D33).
    Note A = R @ D @ R.T for simplicity.
    The parameters of F are x = [qx, qy, qz, qw, ax, ay, az]. This class calculates
    various derivatives of F w.r.t. x and p.
*/

Eigen::VectorXd rimonMethod(const Eigen::MatrixXd& A, const Eigen::VectorXd& a, const Eigen::MatrixXd& B, const Eigen::VectorXd& b) {
    int nv = A.rows();
    Eigen::LLT<Eigen::MatrixXd> lltOfA(A); // Compute the Cholesky decomposition of A
    Eigen::MatrixXd A_sqrt = lltOfA.matrixL(); // Extract the lower triangular matrix

    // C = inv(A_sqrt) * B * inv(A_sqrt).T
    Eigen::MatrixXd A_sqrt_inv = A_sqrt.inverse();
    Eigen::MatrixXd C = A_sqrt_inv * B * A_sqrt_inv.transpose();
    
    Eigen::VectorXd c = A_sqrt.transpose() * (b - a);
    
    Eigen::LLT<Eigen::MatrixXd> lltOfC(C); // Cholesky decomposition of C
    Eigen::MatrixXd C_sqrt = lltOfC.matrixL(); // Lower triangular matrix of C
    Eigen::VectorXd c_tilde = C_sqrt.triangularView<Eigen::Lower>().solve(c);
    
    // Computing C_tilde
    Eigen::MatrixXd C_sqrt_inv = C_sqrt.inverse();
    Eigen::MatrixXd C_tilde = C_sqrt_inv * C_sqrt_inv.transpose();
    
    // Construct matrix M
    Eigen::MatrixXd M(2*nv, 2*nv);
    M << C_tilde, -Eigen::MatrixXd::Identity(nv, nv),
        -c_tilde * c_tilde.transpose(), C_tilde; // doubt
    
    // Compute the smallest eigenvalue of M
    Eigen::EigenSolver<Eigen::MatrixXd> es(M);
    double lambda_min = es.eigenvalues().real().minCoeff();
    
    // Solve for x_rimon
    Eigen::VectorXd x_rimon = (lambda_min * C - Eigen::MatrixXd::Identity(nv, nv)).ldlt().solve(C * c);
    x_rimon = a + lambda_min * A_sqrt.transpose().triangularView<Eigen::Upper>().solve(x_rimon);
    
    return x_rimon;
}

xt::xarray<double> rimonMethodXtensor(const xt::xarray<double>& A, const xt::xarray<double>& a, const xt::xarray<double>& B, const xt::xarray<double>& b) {
    int nv = A.shape()[0];

    xt::xarray<double> A_sqrt = xt::linalg::cholesky(A); 

    xt::xarray<double> A_sqrt_inv = xt::linalg::inv(A_sqrt);
    
    // Eigen::MatrixXd C = A_sqrt_inv * B * A_sqrt_inv.transpose();
    // xt::xarray<double> C = xt::linalg::solve_triangular(A_sqrt, B);
    // C = xt::linalg::solve_triangular(A_sqrt, xt::transpose(C));
    xt::xarray<double> C = xt::linalg::dot(xt::linalg::dot(A_sqrt_inv, B), xt::transpose(A_sqrt_inv));

    // Eigen::VectorXd c = A_sqrt.transpose() * (b - a);
    xt::xarray<double> c = xt::linalg::dot(xt::transpose(A_sqrt), b - a);

    // Placeholder for Cholesky decomposition on C
    xt::xarray<double> C_sqrt = xt::linalg::cholesky(C); 
    
    // Eigen::VectorXd c_tilde = C_sqrt.triangularView<Eigen::Lower>().solve(c);
    xt::xarray<double> c_tilde = xt::linalg::solve(C_sqrt, c); 

    // Eigen::MatrixXd C_sqrt_inv = C_sqrt.inverse();
    xt::xarray<double> C_sqrt_inv = xt::linalg::inv(C_sqrt);
    
    // Eigen::MatrixXd C_tilde = C_sqrt_inv * C_sqrt_inv.transpose();
    xt::xarray<double> C_tilde = xt::linalg::dot(C_sqrt_inv, xt::transpose(C_sqrt_inv));

    // Construct matrix M
    xt::xarray<double> M = xt::zeros<double>({2*nv, 2*nv}); // Initialize M with zeros
    xt::view(M, xt::range(0, nv), xt::range(0, nv)) = C_tilde;
    xt::view(M, xt::range(nv, 2*nv), xt::range(nv, 2*nv)) = C_tilde;
    xt::view(M, xt::range(0, nv), xt::range(nv, 2*nv)) = -xt::eye(nv);
    xt::view(M, xt::range(nv, 2*nv), xt::range(0, nv)) = xt::linalg::outer(-c_tilde, xt::transpose(c_tilde));

    // Compute the smallest eigenvalue of M
    xt::xarray<double> eigenval_real = xt::real(xt::linalg::eigvals(M));
    double lambda_min = std::numeric_limits<double>::max();
    for (double& val : eigenval_real) {
        if (val < lambda_min) {
            lambda_min = val;
        }
    }

    // Solve for x_rimon
    // xt::xarray<double> L = -xt::linalg::cholesky(-lambda_min * C + xt::eye(nv));
    xt::xarray<double> x_rimon = xt::linalg::solve(lambda_min * C - xt::eye(nv), xt::linalg::dot(C, c)); 
    x_rimon = a + lambda_min * xt::linalg::solve(xt::transpose(A_sqrt), x_rimon);
    
    return x_rimon;
}

xt::xarray<double> F(const xt::xarray<double>& p, const xt::xarray<double>& a, const xt::xarray<double>& A) {
    // Compute F(p)
    // p: input vector of dimension 3
    // a: center of the ellipsoid, dimension 3
    // A: real symmetric quadratic coefficient matrix, dimension 3 x 3

    return xt::linalg::dot(p - a, xt::linalg::dot(A, p - a));
}

xt::xarray<double> F_dp(const xt::xarray<double>& p, const xt::xarray<double>& a, const xt::xarray<double>& A) {
    // Compute dF/dp
    // p: input vector of dimension 3
    // a: center of the ellipsoid, dimension 3
    // A: real symmetric quadratic coefficient matrix, dimension 3 x 3

    return 2 * xt::linalg::dot(A, p - a); // Since A is symmetric, A.transpose() is the same as A
}

xt::xarray<double> F_dpdp(const xt::xarray<double>& A) {
    // Compute d^2F/dpdp
    // A: real symmetric quadratic coefficient matrix, dimension 3 x 3

    return 2 * A; // Since A is symmetric, A.transpose() is the same as A
}

xt::xarray<double> F_dpdpdp() {
    // Compute d^3F/dpdpdp

    return xt::zeros<double>({3, 3, 3});;
}

xt::xarray<double> F_dy(const xt::xarray<double>& p, const xt::xarray<double>& a, const xt::xarray<double>& A) {
    // Compute dF/dy, where y = [A11, A12, A13, A22, A23, A33, a1, a2, a3]
    // p: input vector of dimension 3
    // a: center of the ellipsoid, dimension 3
    // A: real symmetric quadratic coefficient matrix, dimension 3 x 3
    int dim_y = 9, dim_A_flat = 6;
    xt::xarray<double> vector = a - p;
    xt::xarray<double> outer_prod = xt::linalg::outer(vector, vector);
    xt::xarray<double> F_dy = xt::zeros<double>({dim_y});
    F_dy(0) = outer_prod(0, 0);
    F_dy(1) = 2 * outer_prod(0, 1);
    F_dy(2) = 2 * outer_prod(0, 2);
    F_dy(3) = outer_prod(1, 1);
    F_dy(4) = 2 * outer_prod(1, 2);
    F_dy(5) = outer_prod(2, 2);
    xt::view(F_dy, xt::range(dim_A_flat, dim_y)) = 2 * xt::linalg::dot(A, vector);

    return F_dy;
}

xt::xarray<double> F_dpdy(const xt::xarray<double>& p, const xt::xarray<double>& a, const xt::xarray<double>& A) {
    // Compute d^2F/dpdy, where y = [A11, A12, A13, A22, A23, A33, a1, a2, a3]
    // p: input vector of dimension 3
    // a: center of the ellipsoid, dimension 3
    // A: real symmetric quadratic coefficient matrix, dimension 3 x 3
    int dim_p = 3, dim_y = 9, dim_A_flat = 6;

    xt::xarray<double> F_dpdy = xt::zeros<double>({dim_p, dim_y});
    xt::xarray<double> vector = p - a;
    F_dpdy(0,0) = 2 * vector(0);
    F_dpdy(0,1) = 2 * vector(1);
    F_dpdy(0,2) = 2 * vector(2);
    F_dpdy(1,1) = 2 * vector(0);
    F_dpdy(1,3) = 2 * vector(1);
    F_dpdy(1,4) = 2 * vector(2);
    F_dpdy(2,2) = 2 * vector(0);
    F_dpdy(2,4) = 2 * vector(1);
    F_dpdy(2,5) = 2 * vector(2);
    xt::view(F_dpdy, xt::all(), xt::range(dim_A_flat, dim_y)) = -2 * A;

    return F_dpdy;
}

xt::xarray<double> F_dydy(const xt::xarray<double>& p, const xt::xarray<double>& a, const xt::xarray<double>& A) {
    // Compute d^2F/dydy, where y = [A11, A12, A13, A22, A23, A33, a1, a2, a3]
    // p: input vector of dimension 3
    // a: center of the ellipsoid, dimension 3
    // A: real symmetric quadratic coefficient matrix, dimension 3 x 3
    int dim_p = 3, dim_y = 9, dim_A_flat = 6;
    xt::xarray<double> F_dydy = xt::zeros<double>({dim_y, dim_y});
    xt::xarray<double> vector = a - p;
    xt::xarray<double> tmp = xt::zeros<double>({dim_p, dim_A_flat});
    tmp(0,0) = 2 * vector(0);
    tmp(0,1) = 2 * vector(1);
    tmp(0,2) = 2 * vector(2);
    tmp(1,1) = 2 * vector(0);
    tmp(1,3) = 2 * vector(1);
    tmp(1,4) = 2 * vector(2);
    tmp(2,2) = 2 * vector(0);
    tmp(2,4) = 2 * vector(1);
    tmp(2,5) = 2 * vector(2);
    xt::view(F_dydy, xt::range(dim_A_flat, dim_y), xt::range(0, dim_A_flat)) = tmp;
    xt::view(F_dydy, xt::range(0, dim_A_flat), xt::range(dim_A_flat, dim_y)) = xt::transpose(tmp);
    xt::view(F_dydy, xt::range(dim_A_flat, dim_y), xt::range(dim_A_flat, dim_y)) = 2 * A;

    return F_dydy;
}

xt::xarray<double> F_dpdpdy(const xt::xarray<double>& A) {
    // Compute d^3F/dpdpdy, where y = [A11, A12, A13, A22, A23, A33, a1, a2, a3]
    // p: input vector of dimension 3
    // a: center of the ellipsoid, dimension 3
    // A: real symmetric quadratic coefficient matrix, dimension 3 x 3
    int dim_p = 3, dim_y = 9, dim_A_flat = 6;
    xt::xarray<double> F_dpdpdy = xt::zeros<double>({dim_p, dim_p, dim_y});
    for (int i = 0; i < dim_A_flat; ++i){
        F_dpdpdy(i/dim_p,i%dim_p,i) = 2;
        F_dpdpdy(i%dim_p,i/dim_p,i) = 2;
    }

    return F_dpdpdy;
}

xt::xarray<double> F_dpdydy(const xt::xarray<double>& A) {
    // Compute d^3F/dpdpdy, where y = [A11, A12, A13, A22, A23, A33, a1, a2, a3]
    // p: input vector of dimension 3
    // a: center of the ellipsoid, dimension 3
    // A: real symmetric quadratic coefficient matrix, dimension 3 x 3
    int dim_p = 3, dim_y = 9, dim_A_flat = 6;
    xt::xarray<double> F_dpdpdy = xt::zeros<double>({dim_p, dim_y, dim_y});
    xt::xarray<double> tmp = xt::zeros<double>({dim_p, dim_p, dim_A_flat});
    for (int i = 0; i < dim_A_flat; ++i){
        tmp(i/dim_p,i%dim_p,i) = -2;
        tmp(i%dim_p,i/dim_p,i) = -2;
    }

    xt::view(F_dpdpdy, xt::all(), xt::range(dim_A_flat, dim_y), xt::range(0, dim_A_flat)) = tmp;
    xt::view(F_dpdpdy, xt::all(), xt::range(0, dim_A_flat), xt::range(dim_A_flat, dim_y)) = xt::transpose(tmp, {0,2,1});

    return F_dpdpdy;
}