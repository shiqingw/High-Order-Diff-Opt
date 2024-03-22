#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor-blas/xlinalg.hpp>

int main()
{   xt::xarray<double> A {{4,0,0},
                          {0,4,0},
                          {0,0,4}};
    xt::xarray<double> L {{2,0,0},
                          {0,2,0},
                          {0,0,2}};
    xt::xarray<double> B {{1,0,0},
                          {0,2,0},
                          {0,0,3}};

    std::cout << xt::linalg::solve(A, B) << std::endl;
    std::cout << xt::linalg::solve_triangular(A, B) << std::endl;
    std::cout << xt::linalg::solve_cholesky(L, B) << std::endl;
}