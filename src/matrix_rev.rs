use std::ops::Mul;

use crate::{
    backwards::{Graph, Rev64},
    matrix::Matrix,
};

impl<const ROWS: usize, const COLS: usize> Mul<Matrix<Rev64, ROWS, COLS>> for Rev64 {
    type Output = Matrix<Rev64, ROWS, COLS>;
    fn mul(self, rhs: Matrix<Rev64, ROWS, COLS>) -> Matrix<Rev64, ROWS, COLS> {
        let mut result = Matrix::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                result.data[i][j] = self * rhs.data[i][j];
            }
        }
        result
    }
}

impl<const ROWS: usize, const COLS: usize> Mul<Matrix<Rev64, ROWS, COLS>> for f64 {
    type Output = Matrix<Rev64, ROWS, COLS>;
    fn mul(self, rhs: Matrix<Rev64, ROWS, COLS>) -> Matrix<Rev64, ROWS, COLS> {
        let mut result = Matrix::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                result.data[i][j] = self * rhs.data[i][j];
            }
        }
        result
    }
}

impl<const ROWS: usize, const COLS: usize> From<Matrix<f64, ROWS, COLS>>
    for Matrix<Rev64, ROWS, COLS>
{
    fn from(m: Matrix<f64, ROWS, COLS>) -> Self {
        let mut result = Matrix::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                result.data[i][j] = Rev64::new(m.data[i][j]);
            }
        }
        result
    }
}

impl<const ROW_IN: usize> Matrix<Rev64, ROW_IN, 1> {
    pub fn jacobi<const ROW_OUT: usize>(
        &mut self,
        output: &mut Matrix<Rev64, ROW_OUT, 1>,
    ) -> Matrix<f64, ROW_OUT, ROW_IN> {
        let mut result = Matrix::zeros();
        let graph = Graph::instance();
        for i in 0..ROW_OUT {
            graph.reset_gradients();
            output[i][0].backward();
            for j in 0..ROW_IN {
                self.data[j][0].update();
                result.data[i][j] = self.data[j][0].grad;
            }
        }
        result
    }
}
