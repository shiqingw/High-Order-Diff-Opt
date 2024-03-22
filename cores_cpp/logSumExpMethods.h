#include <tuple>
#include <xtensor/xarray.hpp>
std::tuple<double, xt::xarray<double>, xt::xarray<double>, xt::xarray<double>>
getLogSumExpDerivaties(const xt::xarray<double>& p, const xt::xarray<double>& A,
const xt::xarray<double>& b, const double kappa);