#include <Eigen/Dense>
#include <xtensor/xarray.hpp>
Eigen::VectorXd rimonMethod(const Eigen::MatrixXd& A, const Eigen::VectorXd& a, const Eigen::MatrixXd& B, const Eigen::VectorXd& b);
xt::xarray<double> rimonMethodXtensor(const xt::xarray<double>& A, const xt::xarray<double>& a, const xt::xarray<double>& B, const xt::xarray<double>& b);
double F(const xt::xarray<double>& p, const xt::xarray<double>& a, const xt::xarray<double>& A);
xt::xarray<double> F_dp(const xt::xarray<double>& p, const xt::xarray<double>& a, const xt::xarray<double>& A);
xt::xarray<double> F_dpdp(const xt::xarray<double>& A);
xt::xarray<double> F_dpdpdp();
xt::xarray<double> F_dy(const xt::xarray<double>& p, const xt::xarray<double>& a, const xt::xarray<double>& A);
xt::xarray<double> F_dpdy(const xt::xarray<double>& p, const xt::xarray<double>& a, const xt::xarray<double>& A);
xt::xarray<double> F_dydy(const xt::xarray<double>& p, const xt::xarray<double>& a, const xt::xarray<double>& A);
xt::xarray<double> F_dpdpdy(const xt::xarray<double>& A);
xt::xarray<double> F_dpdydy(const xt::xarray<double>& A);
xt::xarray<double> RDRT_dq(const xt::xarray<double>& q, const xt::xarray<double>& D, const xt::xarray<double>& R);
xt::xarray<double> RDRT_dqdq(const xt::xarray<double>& q, const xt::xarray<double>& D);
std::tuple<double, xt::xarray<double>, xt::xarray<double>> getGradientEllipsoid(const xt::xarray<double>& a,
    const xt::xarray<double>& q, const xt::xarray<double>& D, const xt::xarray<double>& R,
    const xt::xarray<double>& B, const xt::xarray<double>& b);
std::tuple<double, xt::xarray<double>, xt::xarray<double>, xt::xarray<double>> getGradientAndHessianEllipsoid(
    const xt::xarray<double>& a, const xt::xarray<double>& q, const xt::xarray<double>& D,
    const xt::xarray<double>& R, const xt::xarray<double>& B, const xt::xarray<double>& b);