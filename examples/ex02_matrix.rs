use autodiff::backwards::{Graph, Rev64};
use autodiff::matrix::Matrix;

fn main() {
    let x: Matrix<Rev64, 3, 3> =
        Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).into();
    let mut a1 = Rev64::new(1.0);
    let mut a2 = Rev64::new(2.0);
    let mut a3 = Rev64::new(3.0);
    let mut a = Matrix::<Rev64, 3, 1>::new([[a1], [a2], [a3]]);

    let mut b = 2.0 * (x * a);
    b[0][0].backward();

    a1.update();
    a2.update();
    a3.update();

    println!("∂b0/∂c1 = {}", a1.grad);
    println!("∂b0/∂c2 = {}", a2.grad);
    println!("∂b0/∂c3 = {}", a3.grad);

    Graph::instance().reset_gradients();

    b[1][0].backward();

    a1.update();
    a2.update();
    a3.update();

    println!("∂b1/∂c1 = {}", a1.grad);
    println!("∂b1/∂c2 = {}", a2.grad);
    println!("∂b1/∂c3 = {}", a3.grad);

    Graph::instance().reset_gradients();

    b[2][0].backward();

    a1.update();
    a2.update();
    a3.update();

    println!("∂b2/∂c1 = {}", a1.grad);
    println!("∂b2/∂c2 = {}", a2.grad);
    println!("∂b2/∂c3 = {}", a3.grad);

    let jacobi = a.jacobi(&mut b);
    println!("∂b/∂c =\n{}", jacobi);

    // finite difference check
    let b = 2.0 * (x * a);
    let eps = 1e-6;

    let mut a1 = a.clone();
    a1[0][0].value += eps;

    let b1 = 2.0 * (x * a1);
    let diff1 = b1 - b;

    let mut a2 = a.clone();
    a2[1][0].value += eps;

    let b2 = 2.0 * (x * a2);
    let diff2 = b2 - b;

    let mut a3 = a.clone();
    a3[2][0].value += eps;

    let b3 = 2.0 * (x * a3);
    let diff3 = b3 - b;

    let jacobi_fd = Matrix::new([
        [
            diff1[0][0].value / eps,
            diff2[0][0].value / eps,
            diff3[0][0].value / eps,
        ],
        [
            diff1[1][0].value / eps,
            diff2[1][0].value / eps,
            diff3[1][0].value / eps,
        ],
        [
            diff1[2][0].value / eps,
            diff2[2][0].value / eps,
            diff3[2][0].value / eps,
        ],
    ]);

    println!("∂b/∂c (finite difference) =\n{}", jacobi_fd);
}
