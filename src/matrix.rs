use std::fmt::{self, Display, Formatter};
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, Neg, Sub, SubAssign};

use num_traits::Float;

#[derive(Debug, Clone, Copy)]
pub struct Matrix<T: Float, const ROWS: usize, const COLS: usize> {
    pub data: [[T; COLS]; ROWS],
}

impl<T: Float, const ROWS: usize, const COLS: usize> Default for Matrix<T, ROWS, COLS> {
    fn default() -> Self {
        Self::zeros()
    }
}

impl<T: Float + Display, const ROWS: usize, const COLS: usize> Display for Matrix<T, ROWS, COLS> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        for i in 0..ROWS {
            for j in 0..COLS {
                write!(f, "{:.2} ", self.data[i][j])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl<T: Float, const ROWS: usize, const COLS: usize> Matrix<T, ROWS, COLS> {
    pub fn new(data: [[T; COLS]; ROWS]) -> Self {
        Self { data }
    }

    pub fn from_diag(diag: [T; ROWS]) -> Self {
        let mut result = Self::zeros();
        for i in 0..ROWS {
            result.data[i][i] = diag[i];
        }
        result
    }

    pub fn transpose(&self) -> Matrix<T, COLS, ROWS> {
        let mut result = Matrix::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                result.data[j][i] = self.data[i][j];
            }
        }
        result
    }

    pub fn zeros() -> Self {
        Self {
            data: [[T::zero(); COLS]; ROWS],
        }
    }

    pub fn identity() -> Self {
        let mut result = Self::zeros();
        for i in 0..ROWS {
            result.data[i][i] = T::one();
        }
        result
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut result = Vec::with_capacity(ROWS * COLS);
        // let mut result = [0.0; ROWS * COLS];// would be nice to use array instead of vec, but can't do math with const generics
        for i in 0..ROWS {
            for j in 0..COLS {
                result.push(self.data[i][j]);
            }
        }
        result
    }

    pub fn from_vec(data: Vec<T>) -> Self {
        let mut result = Self::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                result.data[i][j] = data[i * COLS + j];
            }
        }
        result
    }

    pub fn frobenius_norm(&self) -> T {
        let mut result = T::zero();
        for i in 0..ROWS {
            for j in 0..COLS {
                result = result + self.data[i][j] * self.data[i][j];
            }
        }
        result.sqrt()
    }
}

impl<T: Float, const ROWS: usize, const COLS: usize> Index<usize> for Matrix<T, ROWS, COLS> {
    type Output = [T; COLS];

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T: Float, const ROWS: usize, const COLS: usize> IndexMut<usize> for Matrix<T, ROWS, COLS> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T: Float, const R_LHS: usize, const C_LHS: usize, const R_RHS: usize, const C_RHS: usize>
    Mul<Matrix<T, R_RHS, C_RHS>> for Matrix<T, R_LHS, C_LHS>
{
    type Output = Matrix<T, R_LHS, C_RHS>;

    fn mul(self, rhs: Matrix<T, R_RHS, C_RHS>) -> Matrix<T, R_LHS, C_RHS> {
        let mut result = Matrix::zeros();

        for i in 0..R_LHS {
            for j in 0..C_RHS {
                let mut sum = T::zero();
                for k in 0..C_LHS {
                    sum = sum + self[i][k] * rhs[k][j];
                }
                result.data[i][j] = sum;
            }
        }

        result
    }
}

impl<T: Float, const ROWS: usize, const COLS: usize> Add for Matrix<T, ROWS, COLS> {
    type Output = Matrix<T, ROWS, COLS>;
    fn add(self, rhs: Matrix<T, ROWS, COLS>) -> Matrix<T, ROWS, COLS> {
        let mut result = Matrix::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                result.data[i][j] = self.data[i][j] + rhs.data[i][j];
            }
        }
        result
    }
}

impl<T: Float, const ROWS: usize, const COLS: usize> AddAssign<Matrix<T, ROWS, COLS>>
    for Matrix<T, ROWS, COLS>
{
    fn add_assign(&mut self, rhs: Matrix<T, ROWS, COLS>) {
        for i in 0..ROWS {
            for j in 0..COLS {
                self.data[i][j] = self.data[i][j] + rhs.data[i][j];
            }
        }
    }
}

impl<T: Float, const ROWS: usize, const COLS: usize> Sub for Matrix<T, ROWS, COLS> {
    type Output = Matrix<T, ROWS, COLS>;
    fn sub(self, rhs: Matrix<T, ROWS, COLS>) -> Matrix<T, ROWS, COLS> {
        let mut result = Matrix::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                result.data[i][j] = self.data[i][j] - rhs.data[i][j];
            }
        }
        result
    }
}

impl<T: Float, const ROWS: usize, const COLS: usize> SubAssign<Matrix<T, ROWS, COLS>>
    for Matrix<T, ROWS, COLS>
{
    fn sub_assign(&mut self, rhs: Matrix<T, ROWS, COLS>) {
        for i in 0..ROWS {
            for j in 0..COLS {
                self.data[i][j] = self.data[i][j] - rhs.data[i][j];
            }
        }
    }
}

// unary minus
impl<T: Float, const ROWS: usize, const COLS: usize> Neg for Matrix<T, ROWS, COLS> {
    type Output = Matrix<T, ROWS, COLS>;
    fn neg(self) -> Matrix<T, ROWS, COLS> {
        let mut result = Matrix::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                result.data[i][j] = -self.data[i][j];
            }
        }
        result
    }
}

impl<T: Float, const ROWS: usize, const COLS: usize> Mul<T> for Matrix<T, ROWS, COLS> {
    type Output = Matrix<T, ROWS, COLS>;
    fn mul(self, rhs: T) -> Matrix<T, ROWS, COLS> {
        let mut result = Matrix::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                result.data[i][j] = self.data[i][j] * rhs;
            }
        }
        result
    }
}

impl<T: Float> Matrix<T, 3, 3> {
    pub fn rx(angle: T) -> Self {
        let (sin, cos) = angle.sin_cos();
        Matrix::new([
            [T::one(), T::zero(), T::zero()],
            [T::zero(), cos, sin],
            [T::zero(), -sin, cos],
        ])
    }

    pub fn ry(angle: T) -> Self {
        let (sin, cos) = angle.sin_cos();
        Matrix::new([
            [cos, T::zero(), -sin],
            [T::zero(), T::one(), T::zero()],
            [sin, T::zero(), cos],
        ])
    }

    pub fn rz(angle: T) -> Self {
        let (sin, cos) = angle.sin_cos();
        Matrix::new([
            [cos, sin, T::zero()],
            [-sin, cos, T::zero()],
            [T::zero(), T::zero(), T::one()],
        ])
    }
}

// impl From<&Quaternion> for Matrix<3, 3> {
//     fn from(q: &Quaternion) -> Self {
//         let q0 = q.w;
//         let q1 = q.x;
//         let q2 = q.y;
//         let q3 = q.z;

//         let q0_2 = q0 * q0;
//         let q1_2 = q1 * q1;
//         let q2_2 = q2 * q2;
//         let q3_2 = q3 * q3;
//         let q12 = q1 * q2;
//         let q03 = q0 * q3;
//         let q13 = q1 * q3;
//         let q02 = q0 * q2;
//         let q23 = q2 * q3;
//         let q01 = q0 * q1;

//         Matrix {
//             data: [
//                 [
//                     2.0 * (q0_2 + q1_2 - 0.5),
//                     2.0 * (q12 + q03),
//                     2.0 * (q13 - q02),
//                 ],
//                 [
//                     2.0 * (q12 - q03),
//                     2.0 * (q0_2 + q2_2 - 0.5),
//                     2.0 * (q23 + q01),
//                 ],
//                 [
//                     2.0 * (q13 + q02),
//                     2.0 * (q23 - q01),
//                     2.0 * (q0_2 + q3_2 - 0.5),
//                 ],
//             ],
//         }
//     }
// }

impl<T: Float> Matrix<T, 3, 1> {
    pub fn cross(&self, rhs: Matrix<T, 3, 1>) -> Matrix<T, 3, 1> {
        let x1 = self.data[0][0];
        let y1 = self.data[1][0];
        let z1 = self.data[2][0];

        let x2 = rhs.data[0][0];
        let y2 = rhs.data[1][0];
        let z2 = rhs.data[2][0];

        Matrix::new([
            [y1 * z2 - z1 * y2],
            [z1 * x2 - x1 * z2],
            [x1 * y2 - y1 * x2],
        ])
    }

    pub fn dot(&self, rhs: Matrix<T, 3, 1>) -> T {
        let mut result = T::zero();
        for i in 0..3 {
            result = result + self.data[i][0] * rhs.data[i][0];
        }
        result
    }

    pub fn cross_matrix(&self) -> Matrix<T, 3, 3> {
        let x = self.data[0][0];
        let y = self.data[1][0];
        let z = self.data[2][0];

        Matrix::new([[T::zero(), -z, y], [z, T::zero(), -x], [-y, x, T::zero()]])
    }
}
