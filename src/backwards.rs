use num_traits::{Float, Num, NumCast, One, ToPrimitive, Zero};
use std::{
    collections::HashMap,
    ops::{Add, Div, Mul, Neg, Rem, Sub},
    sync::{Arc, Mutex},
};

#[derive(Clone, Copy, Debug)]
pub struct Rev64 {
    pub id: usize,
    pub value: f64,
    pub grad: f64,
}

impl Rev64 {
    pub fn new(value: f64) -> Rev64 {
        let id = Graph::instance().add_variable(value);
        Rev64 {
            id,
            value,
            grad: 0.0,
        }
    }

    /// Performs the backward pass from this variable.
    pub fn backward(&mut self) {
        self.grad = 1.0; // Seed the gradient of the output variable with 1
        Graph::instance().update_variable(*self);

        // Use a queue to manage the nodes that need their gradients updated
        let mut agenda: Vec<usize> = vec![self.id];

        // Visit each variable in the agenda to perform the backward pass
        while let Some(var_id) = agenda.pop() {
            let (var_grad, predecessors) = Graph::instance().get_grad_and_predecessors(var_id);
            for (pred_id, backward_fn) in predecessors {
                let mut pred_var = Graph::instance().get_variable(pred_id);
                backward_fn(var_grad, &mut pred_var);
                agenda.push(pred_id);
                Graph::instance().update_variable(pred_var);
            }
        }
    }
    pub fn update(&mut self) {
        // gets the variable stored in the graph
        let var = Graph::instance().get_variable(self.id);
        // then updates self
        self.value = var.value;
        self.grad = var.grad;
    }
}

impl Add for Rev64 {
    type Output = Rev64;

    fn add(self, rhs: Rev64) -> Rev64 {
        let output = Rev64::new(self.value + rhs.value);
        Graph::instance().add_operation(
            output.id,
            vec![self.id, rhs.id],
            Arc::new(|o_grad, a: &mut Rev64| {
                a.grad += o_grad;
            }),
        );
        output
    }
}

impl Mul for Rev64 {
    type Output = Rev64;

    fn mul(self, rhs: Rev64) -> Rev64 {
        let output = Rev64::new(self.value * rhs.value);
        Graph::instance().add_operation(
            output.id,
            vec![self.id, rhs.id],
            Arc::new(move |o_grad, v: &mut Rev64| {
                if v.id == self.id {
                    v.grad += o_grad * rhs.value;
                } else {
                    v.grad += o_grad * self.value;
                }
            }),
        );
        output
    }
}

impl Rem for Rev64 {
    type Output = Self;

    fn rem(self, other: Self) -> Self {
        let output = Rev64::new(self.value % other.value);
        Graph::instance().add_operation(
            output.id,
            vec![self.id, other.id],
            Arc::new(move |o_grad, v: &mut Rev64| {
                if v.id == self.id {
                    v.grad += o_grad;
                } else {
                    v.grad -= o_grad * (self.value / other.value).trunc();
                }
            }),
        );
        output
    }
}

impl Div for Rev64 {
    type Output = Rev64;

    fn div(self, other: Rev64) -> Rev64 {
        let output = Rev64::new(self.value / other.value);
        Graph::instance().add_operation(
            output.id,
            vec![self.id, other.id],
            Arc::new(move |o_grad, v: &mut Rev64| {
                if v.id == self.id {
                    v.grad += o_grad / other.value;
                } else {
                    v.grad -= o_grad * self.value / (other.value * other.value);
                }
            }),
        );
        output
    }
}

impl Sub for Rev64 {
    type Output = Rev64;

    fn sub(self, other: Rev64) -> Rev64 {
        let output = Rev64::new(self.value - other.value);
        Graph::instance().add_operation(
            output.id,
            vec![self.id, other.id],
            Arc::new(move |o_grad, v: &mut Rev64| {
                if v.id == self.id {
                    v.grad += o_grad;
                } else {
                    v.grad -= o_grad;
                }
            }),
        );
        output
    }
}

impl Neg for Rev64 {
    type Output = Rev64;

    fn neg(self) -> Rev64 {
        let output = Rev64::new(-self.value);
        Graph::instance().add_operation(
            output.id,
            vec![self.id],
            Arc::new(|o_grad, v: &mut Rev64| {
                v.grad -= o_grad;
            }),
        );
        output
    }
}

pub struct Graph {
    variables: Mutex<HashMap<usize, Rev64>>,
    operations: Mutex<HashMap<usize, Vec<(usize, Arc<dyn Fn(f64, &mut Rev64) + Send + Sync>)>>>,
    next_id: Mutex<usize>,
}

impl Graph {
    pub fn instance() -> Arc<Graph> {
        static INSTANCE: once_cell::sync::Lazy<Arc<Graph>> = once_cell::sync::Lazy::new(|| {
            Arc::new(Graph {
                variables: Mutex::new(HashMap::new()),
                operations: Mutex::new(HashMap::new()),
                next_id: Mutex::new(0),
            })
        });
        INSTANCE.clone()
    }

    fn add_variable(&self, value: f64) -> usize {
        let mut id_lock = self.next_id.lock().unwrap();
        let id = *id_lock;
        *id_lock += 1;
        drop(id_lock);

        let mut vars = self.variables.lock().unwrap();
        vars.insert(
            id,
            Rev64 {
                id,
                value,
                grad: 0.0,
            },
        );
        id
    }

    pub fn get_variable(&self, id: usize) -> Rev64 {
        let vars = self.variables.lock().unwrap();
        *vars.get(&id).unwrap()
    }

    pub fn update_variable(&self, var: Rev64) {
        let mut vars = self.variables.lock().unwrap();
        vars.insert(var.id, var);
    }

    fn add_operation<F>(&self, var_id: usize, predecessors: Vec<usize>, backward_fn: Arc<F>)
    where
        F: 'static + Fn(f64, &mut Rev64) + Send + Sync,
    {
        let mut ops = self.operations.lock().unwrap();
        for pred_id in predecessors {
            ops.entry(var_id).or_insert_with(Vec::new).push((
                pred_id,
                Arc::clone(&backward_fn) as Arc<dyn Fn(f64, &mut Rev64) + Send + Sync>,
            ));
        }
    }

    fn get_grad_and_predecessors(
        &self,
        var_id: usize,
    ) -> (
        f64,
        Vec<(usize, Arc<dyn Fn(f64, &mut Rev64) + Send + Sync>)>,
    ) {
        let vars = self.variables.lock().unwrap();
        let var = vars.get(&var_id).unwrap();
        let ops = self.operations.lock().unwrap();
        let preds = ops.get(&var_id).cloned().unwrap_or_default();
        (var.grad, preds)
    }

    pub fn reset_gradients(&self) {
        let mut vars = self.variables.lock().unwrap();
        for (_, var) in vars.iter_mut() {
            var.grad = 0.0;
        }
    }

    pub fn size(&self) -> usize {
        self.variables.lock().unwrap().len()
    }
}

impl Mul<Rev64> for f64 {
    type Output = Rev64;

    fn mul(self, rhs: Rev64) -> Rev64 {
        let output = Rev64::new(self * rhs.value);
        let scalar = self;
        Graph::instance().add_operation(
            output.id,
            vec![rhs.id],
            Arc::new(move |o_grad, b: &mut Rev64| {
                b.grad += o_grad * scalar;
            }),
        );
        output
    }
}

impl PartialOrd for Rev64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl PartialEq for Rev64 {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl NumCast for Rev64 {
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        n.to_f64().map(Rev64::new)
    }
}

impl ToPrimitive for Rev64 {
    fn to_isize(&self) -> Option<isize> {
        self.value.to_isize()
    }

    fn to_i8(&self) -> Option<i8> {
        self.value.to_i8()
    }

    fn to_i16(&self) -> Option<i16> {
        self.value.to_i16()
    }

    fn to_i32(&self) -> Option<i32> {
        self.value.to_i32()
    }

    fn to_i64(&self) -> Option<i64> {
        self.value.to_i64()
    }

    fn to_i128(&self) -> Option<i128> {
        self.value.to_i128()
    }

    fn to_usize(&self) -> Option<usize> {
        self.value.to_usize()
    }

    fn to_u8(&self) -> Option<u8> {
        self.value.to_u8()
    }

    fn to_u16(&self) -> Option<u16> {
        self.value.to_u16()
    }

    fn to_u32(&self) -> Option<u32> {
        self.value.to_u32()
    }

    fn to_u64(&self) -> Option<u64> {
        self.value.to_u64()
    }

    fn to_u128(&self) -> Option<u128> {
        self.value.to_u128()
    }

    fn to_f32(&self) -> Option<f32> {
        self.value.to_f32()
    }

    fn to_f64(&self) -> Option<f64> {
        self.value.to_f64()
    }
}

impl Num for Rev64 {
    type FromStrRadixErr = <f64 as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        f64::from_str_radix(str, radix).map(Rev64::new)
    }
}

impl One for Rev64 {
    fn one() -> Self {
        Rev64::new(1.0)
    }
}

impl Zero for Rev64 {
    fn zero() -> Self {
        Rev64::new(0.0)
    }

    fn is_zero(&self) -> bool {
        self.value == 0.0
    }
}

impl Float for Rev64 {
    fn nan() -> Self {
        Rev64::new(f64::nan())
    }

    fn infinity() -> Self {
        Rev64::new(f64::infinity())
    }

    fn neg_infinity() -> Self {
        Rev64::new(f64::neg_infinity())
    }

    fn neg_zero() -> Self {
        Rev64::new(-0.0)
    }

    fn is_nan(self) -> bool {
        self.value.is_nan()
    }

    fn is_infinite(self) -> bool {
        self.value.is_infinite()
    }

    fn is_finite(self) -> bool {
        self.value.is_finite()
    }

    fn is_normal(self) -> bool {
        self.value.is_normal()
    }

    fn classify(self) -> std::num::FpCategory {
        self.value.classify()
    }

    fn floor(self) -> Self {
        let output = Rev64::new(self.value.floor());
        Graph::instance().add_operation(
            output.id,
            vec![self.id],
            Arc::new(|_o_grad, v: &mut Rev64| {
                v.grad += 0.0;
            }),
        );
        output
    }

    fn ceil(self) -> Self {
        let output = Rev64::new(self.value.ceil());
        Graph::instance().add_operation(
            output.id,
            vec![self.id],
            Arc::new(|_o_grad, v: &mut Rev64| {
                v.grad = 0.0;
            }),
        );
        output
    }

    fn round(self) -> Self {
        let output = Rev64::new(self.value.round());
        Graph::instance().add_operation(
            output.id,
            vec![self.id],
            Arc::new(|_o_grad, v: &mut Rev64| {
                v.grad = 0.0;
            }),
        );
        output
    }

    fn trunc(self) -> Self {
        let output = Rev64::new(self.value.trunc());
        Graph::instance().add_operation(
            output.id,
            vec![self.id],
            Arc::new(|_o_grad, v: &mut Rev64| {
                v.grad = 0.0;
            }),
        );
        output
    }

    fn fract(self) -> Self {
        let output = Rev64::new(self.value.fract());
        Graph::instance().add_operation(
            output.id,
            vec![self.id],
            Arc::new(|o_grad, v: &mut Rev64| {
                v.grad += o_grad;
            }),
        );
        output
    }

    fn abs(self) -> Self {
        let output = Rev64::new(self.value.abs());
        Graph::instance().add_operation(
            output.id,
            vec![self.id],
            Arc::new(|o_grad, v: &mut Rev64| {
                if v.value.is_sign_positive() {
                    v.grad += o_grad;
                } else {
                    v.grad -= o_grad;
                }
            }),
        );
        output
    }

    fn signum(self) -> Self {
        let output = Rev64::new(self.value.signum());
        Graph::instance().add_operation(
            output.id,
            vec![self.id],
            Arc::new(|_o_grad, v: &mut Rev64| {
                v.grad = 0.0;
            }),
        );
        output
    }

    fn is_sign_positive(self) -> bool {
        self.value.is_sign_positive()
    }

    fn is_sign_negative(self) -> bool {
        self.value.is_sign_negative()
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        let output = Rev64::new(self.value.mul_add(a.value, b.value));
        Graph::instance().add_operation(
            output.id,
            vec![self.id, a.id, b.id],
            Arc::new(move |o_grad, v: &mut Rev64| {
                if v.id == self.id {
                    v.grad += o_grad * a.value;
                } else if v.id == a.id {
                    v.grad += o_grad * self.value;
                } else if v.id == b.id {
                    v.grad += o_grad;
                }
            }),
        );
        output
    }

    fn recip(self) -> Self {
        let output = Rev64::new(self.value.recip());
        Graph::instance().add_operation(
            output.id,
            vec![self.id],
            Arc::new(|o_grad, v: &mut Rev64| {
                v.grad -= o_grad / (v.value * v.value);
            }),
        );
        output
    }

    fn powi(self, n: i32) -> Self {
        let output = Rev64::new(self.value.powi(n));
        Graph::instance().add_operation(
            output.id,
            vec![self.id],
            Arc::new(move |o_grad, v: &mut Rev64| {
                v.grad += o_grad * (n as f64) * self.value.powi(n - 1);
            }),
        );
        output
    }

    fn powf(self, n: Self) -> Self {
        let output = Rev64::new(self.value.powf(n.value));
        Graph::instance().add_operation(
            output.id,
            vec![self.id, n.id],
            Arc::new(move |o_grad, v: &mut Rev64| {
                if v.id == self.id {
                    v.grad += o_grad * n.value * self.value.powf(n.value - 1.0);
                } else if v.id == n.id {
                    v.grad += o_grad * self.value.powf(n.value) * self.value.ln();
                }
            }),
        );
        output
    }

    fn sqrt(self) -> Self {
        let output = Rev64::new(self.value.sqrt());
        Graph::instance().add_operation(
            output.id,
            vec![self.id],
            Arc::new(|o_grad, v: &mut Rev64| {
                v.grad += o_grad / (2.0 * v.value.sqrt());
            }),
        );
        output
    }

    fn exp(self) -> Self {
        let output = Rev64::new(self.value.exp());
        Graph::instance().add_operation(
            output.id,
            vec![self.id],
            Arc::new(|o_grad, v: &mut Rev64| {
                v.grad += o_grad * v.value.exp();
            }),
        );
        output
    }

    fn exp2(self) -> Self {
        let output = Rev64::new(self.value.exp2());
        Graph::instance().add_operation(
            output.id,
            vec![self.id],
            Arc::new(|o_grad, v: &mut Rev64| {
                v.grad += o_grad * v.value.exp2() * std::f64::consts::LN_2;
            }),
        );
        output
    }

    fn ln(self) -> Self {
        let output = Rev64::new(self.value.ln());
        Graph::instance().add_operation(
            output.id,
            vec![self.id],
            Arc::new(|o_grad, v: &mut Rev64| {
                v.grad += o_grad / v.value;
            }),
        );
        output
    }

    fn log(self, base: Self) -> Self {
        let output = Rev64::new(self.value.log(base.value));
        Graph::instance().add_operation(
            output.id,
            vec![self.id, base.id],
            Arc::new(move |o_grad, v: &mut Rev64| {
                if v.id == self.id {
                    v.grad += o_grad / (self.value * base.value.ln());
                } else if v.id == base.id {
                    v.grad -= o_grad * self.value.ln() / (base.value * base.value.ln());
                }
            }),
        );
        output
    }

    fn log2(self) -> Self {
        let output = Rev64::new(self.value.log2());
        Graph::instance().add_operation(
            output.id,
            vec![self.id],
            Arc::new(|o_grad, v: &mut Rev64| {
                v.grad += o_grad / (v.value * std::f64::consts::LN_2);
            }),
        );
        output
    }

    fn log10(self) -> Self {
        let output = Rev64::new(self.value.log10());
        Graph::instance().add_operation(
            output.id,
            vec![self.id],
            Arc::new(|o_grad, v: &mut Rev64| {
                v.grad += o_grad / (v.value * std::f64::consts::LN_10);
            }),
        );
        output
    }

    fn max(self, other: Self) -> Self {
        if self > other {
            self
        } else {
            other
        }
    }

    fn min(self, other: Self) -> Self {
        if self < other {
            self
        } else {
            other
        }
    }

    fn abs_sub(self, other: Self) -> Self {
        Rev64::new((self.value - other.value).abs_sub(other.value))
    }

    fn cbrt(self) -> Self {
        let output = Rev64::new(self.value.cbrt());
        Graph::instance().add_operation(
            output.id,
            vec![self.id],
            Arc::new(|o_grad, v: &mut Rev64| {
                v.grad += o_grad / (3.0 * v.value.powf(2. / 3.));
            }),
        );
        output
    }

    fn hypot(self, other: Self) -> Self {
        let output = Rev64::new(self.value.hypot(other.value));
        Graph::instance().add_operation(
            output.id,
            vec![self.id, other.id],
            Arc::new(move |o_grad, v: &mut Rev64| {
                if v.id == self.id {
                    v.grad += o_grad * self.value / self.value.hypot(other.value);
                } else if v.id == other.id {
                    v.grad += o_grad * other.value / self.value.hypot(other.value);
                }
            }),
        );
        output
    }

    fn sin(self) -> Self {
        let output = Rev64::new(self.value.sin());

        Graph::instance().add_operation(
            output.id,
            vec![self.id],
            Arc::new(move |o_grad, v: &mut Rev64| {
                v.grad += o_grad * self.value.cos();
            }),
        );
        output
    }

    fn cos(self) -> Self {
        let output = Rev64::new(self.value.cos());
        Graph::instance().add_operation(
            output.id,
            vec![self.id],
            Arc::new(move |o_grad, v: &mut Rev64| {
                v.grad -= o_grad * self.value.sin();
            }),
        );
        output
    }

    fn tan(self) -> Self {
        let output = Rev64::new(self.value.tan());
        Graph::instance().add_operation(
            output.id,
            vec![self.id],
            Arc::new(move |o_grad, v: &mut Rev64| {
                v.grad += o_grad / (self.value.cos() * self.value.cos());
            }),
        );
        output
    }

    fn asin(self) -> Self {
        let output = Rev64::new(self.value.asin());
        Graph::instance().add_operation(
            output.id,
            vec![self.id],
            Arc::new(move |o_grad, v: &mut Rev64| {
                v.grad += o_grad / (1.0 - self.value * self.value).sqrt();
            }),
        );
        output
    }

    fn acos(self) -> Self {
        let output = Rev64::new(self.value.acos());
        Graph::instance().add_operation(
            output.id,
            vec![self.id],
            Arc::new(move |o_grad, v: &mut Rev64| {
                v.grad -= o_grad / (1.0 - self.value * self.value).sqrt();
            }),
        );
        output
    }

    fn atan(self) -> Self {
        let output = Rev64::new(self.value.atan());
        Graph::instance().add_operation(
            output.id,
            vec![self.id],
            Arc::new(move |o_grad, v: &mut Rev64| {
                v.grad += o_grad / (1.0 + self.value * self.value);
            }),
        );
        output
    }

    fn atan2(self, other: Self) -> Self {
        let output = Rev64::new(self.value.atan2(other.value));
        Graph::instance().add_operation(
            output.id,
            vec![self.id, other.id],
            Arc::new(move |o_grad, v: &mut Rev64| {
                if v.id == self.id {
                    v.grad += o_grad * other.value
                        / (self.value * self.value + other.value * other.value);
                } else if v.id == other.id {
                    v.grad -=
                        o_grad * self.value / (self.value * self.value + other.value * other.value);
                }
            }),
        );
        output
    }

    fn sin_cos(self) -> (Self, Self) {
        let (sin_value, cos_value) = self.value.sin_cos();
        let sin_output = Rev64::new(sin_value);
        let cos_output = Rev64::new(cos_value);
        Graph::instance().add_operation(
            sin_output.id,
            vec![self.id],
            Arc::new(move |o_grad, v: &mut Rev64| {
                v.grad += o_grad * cos_value;
            }),
        );
        Graph::instance().add_operation(
            cos_output.id,
            vec![self.id],
            Arc::new(move |o_grad, v: &mut Rev64| {
                v.grad -= o_grad * sin_value;
            }),
        );
        (sin_output, cos_output)
    }

    fn exp_m1(self) -> Self {
        let output = Rev64::new(self.value.exp_m1());
        Graph::instance().add_operation(
            output.id,
            vec![self.id],
            Arc::new(move |o_grad, v: &mut Rev64| {
                v.grad += o_grad * (v.value.exp());
            }),
        );
        output
    }

    fn ln_1p(self) -> Self {
        let output = Rev64::new(self.value.ln_1p());
        Graph::instance().add_operation(
            output.id,
            vec![self.id],
            Arc::new(move |o_grad, v: &mut Rev64| {
                v.grad += o_grad / (1.0 + self.value);
            }),
        );
        output
    }

    fn sinh(self) -> Self {
        let output = Rev64::new(self.value.sinh());
        Graph::instance().add_operation(
            output.id,
            vec![self.id],
            Arc::new(move |o_grad, v: &mut Rev64| {
                v.grad += o_grad * self.value.cosh();
            }),
        );
        output
    }

    fn cosh(self) -> Self {
        let output = Rev64::new(self.value.cosh());
        Graph::instance().add_operation(
            output.id,
            vec![self.id],
            Arc::new(move |o_grad, v: &mut Rev64| {
                v.grad += o_grad * self.value.sinh();
            }),
        );
        output
    }

    fn tanh(self) -> Self {
        let output = Rev64::new(self.value.tanh());
        Graph::instance().add_operation(
            output.id,
            vec![self.id],
            Arc::new(move |o_grad, v: &mut Rev64| {
                v.grad += o_grad * (1.0 - self.value.tanh() * self.value.tanh());
            }),
        );
        output
    }

    fn asinh(self) -> Self {
        let output = Rev64::new(self.value.asinh());
        Graph::instance().add_operation(
            output.id,
            vec![self.id],
            Arc::new(move |o_grad, v: &mut Rev64| {
                v.grad += o_grad / (self.value * self.value + 1.0).sqrt();
            }),
        );
        output
    }

    fn acosh(self) -> Self {
        let output = Rev64::new(self.value.acosh());
        Graph::instance().add_operation(
            output.id,
            vec![self.id],
            Arc::new(move |o_grad, v: &mut Rev64| {
                v.grad += o_grad / (self.value * self.value - 1.0).sqrt();
            }),
        );
        output
    }

    fn atanh(self) -> Self {
        let output = Rev64::new(self.value.atanh());
        Graph::instance().add_operation(
            output.id,
            vec![self.id],
            Arc::new(move |o_grad, v: &mut Rev64| {
                v.grad += o_grad / (1.0 - self.value * self.value);
            }),
        );
        output
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        self.value.integer_decode()
    }

    fn to_degrees(self) -> Self {
        let output = Rev64::new(self.value.to_degrees());
        Graph::instance().add_operation(
            output.id,
            vec![self.id],
            Arc::new(|_o_grad, v: &mut Rev64| {
                // v.grad += 0.0; // this is incorrect
                v.grad += 180.0 / std::f64::consts::PI;
            }),
        );
        output
    }

    fn to_radians(self) -> Self {
        let output = Rev64::new(self.value.to_radians());
        Graph::instance().add_operation(
            output.id,
            vec![self.id],
            Arc::new(|_o_grad, v: &mut Rev64| {
                v.grad += std::f64::consts::PI / 180.0;
            }),
        );
        output
    }

    fn max_value() -> Self {
        Rev64::new(f64::max_value())
    }

    fn min_value() -> Self {
        Rev64::new(f64::min_value())
    }

    fn epsilon() -> Self {
        Rev64::new(f64::epsilon())
    }

    fn min_positive_value() -> Self {
        Rev64::new(f64::min_positive_value())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use proptest::{prelude::*, test_runner::Config};

    const NUM_CASES: u32 = 10000;
    const EPSILON: f64 = 1e-6; // finite difference step size

    // automatic differentiation tolerance
    const AUTO_ERROR: f64 = 1e-13; // absolute error

    // Finite difference tolerances
    // Fails with a REL_ERROR of 1e-5 and ABS_ERROR of 1e-9 (automatic differentiation is so much more accurate than finite difference!)
    const REL_ERROR: f64 = 1e-4; // relative error (worst offenders: div, asin, acos)
    const ABS_ERROR: f64 = 1e-8; // absolute error (worst offenders: hypot, exp_m1, atan2, tanh)

    const TEST_RANGE: f64 = 100.0;

    fn test_operation<F, G, H>(x: f64, y: f64, op: F, op_f64: G, grad_fn: H)
    where
        F: Fn(Rev64, Rev64) -> Rev64,
        G: Fn(f64, f64) -> f64,
        H: Fn(f64, f64) -> (f64, f64),
    {
        // exact
        let (grad_x_exact, grad_y_exact) = grad_fn(x, y);

        // automatic differentiation
        let mut var_x = Rev64::new(x);
        let mut var_y = Rev64::new(y);
        let mut result = op(var_x, var_y);

        result.backward();
        var_x.update();
        var_y.update();
        let grad_x = var_x.grad;
        let grad_y = var_y.grad;

        assert_relative_eq!(grad_x, grad_x_exact, epsilon = AUTO_ERROR);
        assert_relative_eq!(grad_y, grad_y_exact, epsilon = AUTO_ERROR);

        // finite difference
        let grad_x_fd = (op_f64(x + EPSILON, y) - op_f64(x - EPSILON, y)) / (2.0 * EPSILON);
        let grad_y_fd = (op_f64(x, y + EPSILON) - op_f64(x, y - EPSILON)) / (2.0 * EPSILON);

        let tolerance_x = ABS_ERROR + REL_ERROR * grad_x.abs();
        let tolerance_y = ABS_ERROR + REL_ERROR * grad_y.abs();

        assert_relative_eq!(grad_x_fd, grad_x_exact, epsilon = tolerance_x);
        assert_relative_eq!(grad_y_fd, grad_y_exact, epsilon = tolerance_y);
    }

    fn test_single_operation<F, G, H>(x: f64, op: F, op_f64: G, grad_fn: H)
    where
        F: Fn(Rev64) -> Rev64,
        G: Fn(f64) -> f64,
        H: Fn(f64) -> f64,
    {
        // exact
        let grad_x_exact = grad_fn(x);

        // automatic differentiation
        let mut var_x = Rev64::new(x);
        let mut result = op(var_x);

        result.backward();
        var_x.update();
        let grad_x = var_x.grad;

        assert_relative_eq!(grad_x, grad_x_exact, epsilon = AUTO_ERROR);

        // finite difference
        let grad_x_fd = (op_f64(x + EPSILON) - op_f64(x - EPSILON)) / (2.0 * EPSILON);

        let tolerance = ABS_ERROR + REL_ERROR * grad_x.abs();
        assert_relative_eq!(grad_x_fd, grad_x_exact, epsilon = tolerance);
    }

    proptest! {
        #![proptest_config(Config {
            cases: NUM_CASES,
            ..Config::default()
        })]

        #[test]
        fn test_add(x in -TEST_RANGE..TEST_RANGE, y in -TEST_RANGE..TEST_RANGE) {
            test_operation(x, y, |a, b| a + b,|a, b| a + b, |_, _| (1.0, 1.0));
        }

        #[test]
        fn test_sub(x in -TEST_RANGE..TEST_RANGE, y in -TEST_RANGE..TEST_RANGE) {
            test_operation(x, y, |a, b| a - b,|a, b| a - b, |_, _| (1.0, -1.0));
        }

        #[test]
        fn test_mul(x in -TEST_RANGE..TEST_RANGE, y in -TEST_RANGE..TEST_RANGE) {
            test_operation(x, y, |a, b| a * b,|a, b| a * b, |x,y| (y, x));
        }

        #[test]
        fn test_div(x in -TEST_RANGE..TEST_RANGE, y in -TEST_RANGE..TEST_RANGE) {
            prop_assume!(y.abs() > 100.0*EPSILON ); // finite difference fails near zero
            test_operation(x, y, |a, b| a / b,|a, b| a / b, div_derivative);
        }

        #[test]
        fn test_rem(x in -TEST_RANGE..TEST_RANGE, y in -TEST_RANGE..TEST_RANGE) {
            prop_assume!(y.abs() > 10.0*EPSILON );
            prop_assume!((x / (y+EPSILON)).trunc() == (x / (y-EPSILON)).trunc());
            test_operation(x, y, |a, b| a % b,|a, b| a % b, rem_derivative);
        }

        #[test]
        fn test_to_degrees(x in -TEST_RANGE..TEST_RANGE) {
            test_single_operation(x, |a| a.to_degrees(),|a| a.to_degrees(), |_| 180.0 / std::f64::consts::PI);
        }

        #[test]
        fn test_to_radians(x in -TEST_RANGE..TEST_RANGE) {
            test_single_operation(x, |a| a.to_radians(), |a| a.to_radians(), |_| std::f64::consts::PI / 180.0);
        }

        #[test]
        fn test_sin(x in -TEST_RANGE..TEST_RANGE) {
            test_single_operation(x, |a| a.sin(),|a| a.sin(), |x| x.cos());
        }

        #[test]
        fn test_cos(x in -TEST_RANGE..TEST_RANGE) {
            test_single_operation(x, |a| a.cos(),|a| a.cos(), |x| -x.sin());
        }

        #[test]
        fn test_tan(x in (-1.0+EPSILON)..(1.0-EPSILON)) {
            test_single_operation(x, |a| a.tan(),|a| a.tan(), |x| 1.0 / (x.cos() * x.cos()));
        }

        #[test]
        fn test_asin(x in (-1.0+100.*EPSILON)..(1.0-100.*EPSILON)) {
            // badly behaved at the edges [1.0, -1.0] finite difference fails
            test_single_operation(x, |a| a.asin(),|a| a.asin(), |x| 1.0 / (1.0 - x * x).sqrt());
        }

        #[test]
        fn test_acos(x in (100.0*EPSILON)..(1.0-100.0*EPSILON)) {
            // badly behaved at the edges [1.0, -1.0] finite difference fails
            test_single_operation(x, |a| a.acos(),|a| a.acos(), |x| -1.0 / (1.0 - x * x).sqrt());
        }

        #[test]
        fn test_atan(x in -TEST_RANGE..TEST_RANGE) {
            test_single_operation(x, |a| a.atan(),|a| a.atan(), |x| 1.0 / (1.0 + x * x));
        }

        #[test]
        fn test_sinh(x in -TEST_RANGE..TEST_RANGE) {
            test_single_operation(x, |a| a.sinh(),|a| a.sinh(), |x| x.cosh());
        }

        #[test]
        fn test_cosh(x in -TEST_RANGE..TEST_RANGE) {
            test_single_operation(x, |a| a.cosh(), |a| a.cosh(), |x| x.sinh());
        }

        #[test]
        fn test_tanh(x in -TEST_RANGE..TEST_RANGE) {
            test_single_operation(x, |a| a.tanh(), |a| a.tanh(), |x| 1.0 - x.tanh() * x.tanh());
        }

        #[test]
        fn test_asinh(x in -TEST_RANGE..TEST_RANGE) {
            test_single_operation(x, |a| a.asinh(),|a| a.asinh(), |x| 1.0 / (x * x + 1.0).sqrt());
        }

        #[test]
        fn test_acosh(x in 1.0..TEST_RANGE) {
            test_single_operation(x, |a| a.acosh(),|a| a.acosh(), |x| 1.0 / (x * x - 1.0).sqrt());
        }

        #[test]
        fn test_atanh(x in -0.99..0.99) {
            test_single_operation(x, |a| a.atanh(),|a| a.atanh(), |x| 1.0 / (1.0 - x * x));
        }

        #[test]
        fn test_exp(x in -TEST_RANGE..TEST_RANGE) {
            test_single_operation(x, |a| a.exp(), |a| a.exp(), |x| x.exp());
        }

        #[test]
        fn test_ln(x in 0.1..TEST_RANGE) {
            test_single_operation(x, |a| a.ln(),|a| a.ln(), |x| 1.0 / x);
        }

        #[test]
        fn test_log2(x in 0.1..TEST_RANGE) {
            test_single_operation(x, |a| a.log2(),|a| a.log2(), |x| 1.0 / (x * std::f64::consts::LN_2));
        }

        #[test]
        fn test_log10(x in 0.1..TEST_RANGE) {
            test_single_operation(x, |a| a.log10(),|a| a.log10(), |x| 1.0 / (x * std::f64::consts::LN_10));
        }

        #[test]
        fn test_exp2(x in -TEST_RANGE..TEST_RANGE) {
            test_single_operation(x, |a| a.exp2(), |a| a.exp2(), |x| x.exp2() * std::f64::consts::LN_2);
        }

        #[test]
        fn test_sqrt(x in (1.0/TEST_RANGE)..TEST_RANGE) {
            test_single_operation(x, |a| a.sqrt(),|a| a.sqrt(), |x| 0.5 / x.sqrt());
        }

        #[test]
        fn test_cbrt(x in EPSILON..TEST_RANGE) {
            prop_assume!(x > 0.0);
            test_single_operation(x, |a| a.cbrt(),|a| a.cbrt(), |x| 1.0 / (3.0 * x.powf(2. / 3.)));
        }

        #[test]
        fn test_hypot(x in -TEST_RANGE..TEST_RANGE, y in -TEST_RANGE..TEST_RANGE) {
            test_operation(x, y, |a, b| a.hypot(b),|a, b| a.hypot(b), |x, y| (x / (x * x + y * y).sqrt(), y / (x * x + y * y).sqrt()));
        }

        #[test]
        fn test_atan2(x in -TEST_RANGE..TEST_RANGE, y in -TEST_RANGE..TEST_RANGE) {
            prop_assume!(x.abs() > 10.0*EPSILON );
            test_operation(x, y, |a, b| a.atan2(b),|a, b| a.atan2(b), atan2_derivative);
        }

        #[test]
        fn test_recip(x in 0.1..TEST_RANGE) {
            test_single_operation(x, |a| a.recip(), |a| a.recip(), |x| -1.0 / (x * x));
        }

        #[test]
        fn test_neg(x in -TEST_RANGE..TEST_RANGE) {
            test_single_operation(x, |a| -a, |a| -a, |_x| -1.0);
        }

        #[test]
        fn test_abs(x in -TEST_RANGE..TEST_RANGE) {
            test_single_operation(x, |a| a.abs(),|a| a.abs(), |x| if x >= 0.0 { 1.0 } else { -1.0 });
        }

        #[test]
        fn test_fract(x in -TEST_RANGE..TEST_RANGE) {
            // this may fail for values near whole numbers
            test_single_operation(x, |a| a.fract(),|a| a.fract(), |_x| 1.0);
        }

        #[test]
        fn test_signum(x in -TEST_RANGE..TEST_RANGE) {
            prop_assume!((x+EPSILON).signum() == (x-EPSILON).signum());
            test_single_operation(x, |a| a.signum(), |a| a.signum(), |_x| 0.0);
        }

        #[test]
        fn test_floor(x in -TEST_RANGE..TEST_RANGE) {
            prop_assume!((x+EPSILON).floor() == (x-EPSILON).floor());
            test_single_operation(x, |a| a.floor(),|a| a.floor(), |_x| 0.0);
        }

        #[test]
        fn test_ceil(x in -TEST_RANGE..TEST_RANGE) {
            prop_assume!((x+EPSILON).ceil() == (x-EPSILON).ceil());
            test_single_operation(x, |a| a.ceil(),|a| a.ceil(), |_x| 0.0);
        }

        #[test]
        fn test_round(x in -TEST_RANGE..TEST_RANGE) {
            prop_assume!((x+EPSILON).round() == (x-EPSILON).round());
            test_single_operation(x, |a| a.round(),|a| a.round(), |_x| 0.0);
        }

        #[test]
        fn test_trunc(x in -TEST_RANGE..TEST_RANGE) {
            prop_assume!((x+EPSILON).trunc() == (x-EPSILON).trunc());
            test_single_operation(x, |a| a.trunc(),|a| a.trunc(), |_x| 0.0);
        }

        #[test]
        fn test_exp_m1(x in -TEST_RANGE..TEST_RANGE) {
            test_single_operation(x, |a| a.exp_m1(),|a| a.exp_m1(), |x| x.exp());
        }

        #[test]
        fn test_ln_1p(x in 0.0..TEST_RANGE) {
            test_single_operation(x, |a| a.ln_1p(),|a| a.ln_1p(), |x| 1.0 / (1.0 + x));
        }

        #[test]
        fn test_mul_add(x in -TEST_RANGE..TEST_RANGE, y in -TEST_RANGE..TEST_RANGE, z in -TEST_RANGE..TEST_RANGE) {
            let mut var_x = Rev64::new(x);
            let mut var_y = Rev64::new(y);
            let mut var_z = Rev64::new(z);

            let mut result = var_x.mul_add(var_y, var_z);

            result.backward();
            var_x.update();
            var_y.update();
            var_z.update();
            let grad_x = var_x.grad;
            let grad_y = var_y.grad;
            let grad_z = var_z.grad;

            assert_relative_eq!(grad_x, y, epsilon = 10. * EPSILON);
            assert_relative_eq!(grad_y, x, epsilon = 10. * EPSILON);
            assert_relative_eq!(grad_z, 1.0, epsilon = 10. * EPSILON);
        }
    }

    fn div_derivative(x: f64, y: f64) -> (f64, f64) {
        (1.0 / y, -x / (y * y))
    }

    fn rem_derivative(x: f64, y: f64) -> (f64, f64) {
        (1.0, -(x / y).trunc())
    }

    fn atan2_derivative(x: f64, y: f64) -> (f64, f64) {
        let denom = x * x + y * y;
        (y / denom, -x / denom)
    }
}
