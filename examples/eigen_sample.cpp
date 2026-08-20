#include <Eigen/Dense>

#include <cmath>
#include <iostream>

int main() {
    Eigen::Matrix2d coefficients;
    coefficients << 3.0, 2.0,
                    1.0, 2.0;
    Eigen::Vector2d expected_solution(2.0, 1.0);
    Eigen::Vector2d right_hand_side = coefficients * expected_solution;
    Eigen::Vector2d solution = coefficients.colPivHouseholderQr().solve(right_hand_side);
    const double error = (solution - expected_solution).norm();

    std::cout << "Eigen " << EIGEN_WORLD_VERSION << '.' << EIGEN_MAJOR_VERSION << '.'
              << EIGEN_MINOR_VERSION << "\n";
    std::cout << "solution = [" << solution.x() << ", " << solution.y() << "]\n";
    std::cout << "error = " << error << "\n";

    if (error > 1e-10) {
        std::cerr << "Eigen sample failed: solution error is too large\n";
        return 1;
    }

    std::cout << "Eigen sample passed\n";
    return 0;
}
