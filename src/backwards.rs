use num_traits::{Float, Num, NumCast, One, ToPrimitive, Zero};
use std::{
    collections::HashMap,
    ops::{Add, Div, Mul, Neg, Rem, Sub},
    sync::{Arc, Mutex},
};

#[derive(Clone, Copy, Debug)]
pub struct Rev64 {
    pub id: usize,
}

#[derive(Clone, Copy, Debug)]
struct Rev64Data {
    id: usize,
    value: f64,
    grad: f64,
}

impl Rev64 {
    pub fn new(value: f64) -> Rev64 {
        let graph = Graph::instance();
        let id = graph.add_variable(value);
        Rev64 { id }
    }

    /// Performs the backward pass from this variable.
    pub fn backward(&mut self) {
        let graph = Graph::instance();
        graph.reset_gradients();

        // Seed the gradient of the output variable with 1
        graph.set_grad(*self, 1.0);

        // Use a queue to manage the nodes that need their gradients updated
        let mut agenda: Vec<usize> = vec![self.id];

        // Visit each variable in the agenda to perform the backward pass
        while let Some(var_id) = agenda.pop() {
            let (var_grad, predecessors) = graph.get_grad_and_predecessors(var_id);
            for (pred_id, backward_fn) in predecessors {
                let mut pred_var = graph.get_variable(pred_id);
                backward_fn(var_grad, &mut pred_var);
                agenda.push(pred_id);
                graph.update_variable(pred_id, pred_var);
            }
        }
    }

    pub fn get_value(&self) -> f64 {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        self_data.value
    }

    pub fn get_grad(&self) -> f64 {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        self_data.grad
    }

    pub fn get_value_grad(&self) -> (f64, f64) {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        (self_data.value, self_data.grad)
    }

    pub fn set_value(&self, value: f64) {
        let graph = Graph::instance();
        let mut vars = graph.variables.lock().unwrap();
        let var = vars.get_mut(&self.id).unwrap();
        var.value = value;
    }
}

impl Add for Rev64 {
    type Output = Rev64;

    fn add(self, rhs: Rev64) -> Rev64 {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let rhs_data = graph.get_variable(rhs.id);
        let output = Rev64::new(self_data.value + rhs_data.value);
        graph.add_operation(
            output.id,
            vec![self.id, rhs.id],
            Arc::new(|o_grad, a: &mut Rev64Data| {
                a.grad += o_grad;
            }),
        );
        output
    }
}

impl Mul for Rev64 {
    type Output = Rev64;

    fn mul(self, rhs: Rev64) -> Rev64 {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let rhs_data = graph.get_variable(rhs.id);
        let output = Rev64::new(self_data.value * rhs_data.value);
        graph.add_operation(
            output.id,
            vec![self.id, rhs.id],
            Arc::new(move |o_grad, v: &mut Rev64Data| {
                if v.id == self.id {
                    v.grad += o_grad * rhs_data.value;
                } else {
                    v.grad += o_grad * self_data.value;
                }
            }),
        );
        output
    }
}

impl Rem for Rev64 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let rhs_data = graph.get_variable(rhs.id);
        let output = Rev64::new(self_data.value % rhs_data.value);
        graph.add_operation(
            output.id,
            vec![self.id, rhs.id],
            Arc::new(move |o_grad, v: &mut Rev64Data| {
                if v.id == self.id {
                    v.grad += o_grad;
                } else {
                    v.grad -= o_grad * (self_data.value / rhs_data.value).trunc();
                }
            }),
        );
        output
    }
}
// everything above has already been updated for Rev64Data

// everything below needs to be update for Rev64Data
impl Div for Rev64 {
    type Output = Rev64;

    fn div(self, other: Rev64) -> Rev64 {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let other_data = graph.get_variable(other.id);
        let output = Rev64::new(self_data.value / other_data.value);
        graph.add_operation(
            output.id,
            vec![self.id, other.id],
            Arc::new(move |o_grad, v: &mut Rev64Data| {
                if v.id == self.id {
                    v.grad += o_grad / other_data.value;
                } else {
                    v.grad -= o_grad * self_data.value / (other_data.value * other_data.value);
                }
            }),
        );
        output
    }
}

impl Sub for Rev64 {
    type Output = Rev64;

    fn sub(self, other: Rev64) -> Rev64 {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let other_data = graph.get_variable(other.id);
        let output = Rev64::new(self_data.value - other_data.value);
        graph.add_operation(
            output.id,
            vec![self.id, other.id],
            Arc::new(move |o_grad, v: &mut Rev64Data| {
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
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let output = Rev64::new(-self_data.value);
        graph.add_operation(
            output.id,
            vec![self.id],
            Arc::new(|o_grad, v: &mut Rev64Data| {
                v.grad -= o_grad;
            }),
        );
        output
    }
}

pub struct Graph {
    variables: Mutex<HashMap<usize, Rev64Data>>,
    operations: Mutex<HashMap<usize, Vec<(usize, Arc<dyn Fn(f64, &mut Rev64Data) + Send + Sync>)>>>,
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
            Rev64Data {
                id,
                value,
                grad: 0.0,
            },
        );
        id
    }

    pub fn set_grad(&self, var: Rev64, grad: f64) {
        let mut vars = self.variables.lock().unwrap();
        let var = vars.get_mut(&var.id).unwrap();
        var.grad = grad;
    }

    fn get_variable(&self, id: usize) -> Rev64Data {
        let vars = self.variables.lock().unwrap();
        *vars.get(&id).unwrap()
    }

    fn update_variable(&self, id: usize, var: Rev64Data) {
        let mut vars = self.variables.lock().unwrap();
        vars.insert(id, var);
    }

    fn add_operation<F>(&self, var_id: usize, predecessors: Vec<usize>, backward_fn: Arc<F>)
    where
        F: 'static + Fn(f64, &mut Rev64Data) + Send + Sync,
    {
        let mut ops = self.operations.lock().unwrap();
        for pred_id in predecessors {
            ops.entry(var_id).or_insert_with(Vec::new).push((
                pred_id,
                Arc::clone(&backward_fn) as Arc<dyn Fn(f64, &mut Rev64Data) + Send + Sync>,
            ));
        }
    }

    fn get_grad_and_predecessors(
        &self,
        var_id: usize,
    ) -> (
        f64,
        Vec<(usize, Arc<dyn Fn(f64, &mut Rev64Data) + Send + Sync>)>,
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
        let graph = Graph::instance();
        let rhs_data = graph.get_variable(rhs.id);
        let output = Rev64::new(self * rhs_data.value);
        let scalar = self;
        graph.add_operation(
            output.id,
            vec![rhs.id],
            Arc::new(move |o_grad, b: &mut Rev64Data| {
                b.grad += o_grad * scalar;
            }),
        );
        output
    }
}

impl PartialOrd for Rev64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let other_data = graph.get_variable(other.id);
        self_data.value.partial_cmp(&other_data.value)
    }
}

impl PartialEq for Rev64 {
    fn eq(&self, other: &Self) -> bool {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let other_data = graph.get_variable(other.id);
        self_data.value == other_data.value
    }
}

impl NumCast for Rev64 {
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        n.to_f64().map(Rev64::new)
    }
}

impl ToPrimitive for Rev64 {
    fn to_isize(&self) -> Option<isize> {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        self_data.value.to_isize()
    }

    fn to_i8(&self) -> Option<i8> {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        self_data.value.to_i8()
    }

    fn to_i16(&self) -> Option<i16> {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        self_data.value.to_i16()
    }

    fn to_i32(&self) -> Option<i32> {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        self_data.value.to_i32()
    }

    fn to_i64(&self) -> Option<i64> {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        self_data.value.to_i64()
    }

    fn to_i128(&self) -> Option<i128> {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        self_data.value.to_i128()
    }

    fn to_usize(&self) -> Option<usize> {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        self_data.value.to_usize()
    }

    fn to_u8(&self) -> Option<u8> {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        self_data.value.to_u8()
    }

    fn to_u16(&self) -> Option<u16> {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        self_data.value.to_u16()
    }

    fn to_u32(&self) -> Option<u32> {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        self_data.value.to_u32()
    }

    fn to_u64(&self) -> Option<u64> {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        self_data.value.to_u64()
    }

    fn to_u128(&self) -> Option<u128> {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        self_data.value.to_u128()
    }

    fn to_f32(&self) -> Option<f32> {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        self_data.value.to_f32()
    }

    fn to_f64(&self) -> Option<f64> {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        self_data.value.to_f64()
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
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        self_data.value == 0.0
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
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        self_data.value.is_nan()
    }

    fn is_infinite(self) -> bool {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        self_data.value.is_infinite()
    }

    fn is_finite(self) -> bool {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        self_data.value.is_finite()
    }

    fn is_normal(self) -> bool {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        self_data.value.is_normal()
    }

    fn classify(self) -> std::num::FpCategory {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        self_data.value.classify()
    }

    fn floor(self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let output = Rev64::new(self_data.value.floor());
        graph.add_operation(
            output.id,
            vec![self.id],
            Arc::new(|_o_grad, v: &mut Rev64Data| {
                v.grad += 0.0;
            }),
        );
        output
    }

    fn ceil(self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let output = Rev64::new(self_data.value.ceil());
        graph.add_operation(
            output.id,
            vec![self.id],
            Arc::new(|_o_grad, v: &mut Rev64Data| {
                v.grad = 0.0;
            }),
        );
        output
    }

    fn round(self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let output = Rev64::new(self_data.value.round());
        graph.add_operation(
            output.id,
            vec![self.id],
            Arc::new(|_o_grad, v: &mut Rev64Data| {
                v.grad = 0.0;
            }),
        );
        output
    }

    fn trunc(self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let output = Rev64::new(self_data.value.trunc());
        graph.add_operation(
            output.id,
            vec![self.id],
            Arc::new(|_o_grad, v: &mut Rev64Data| {
                v.grad = 0.0;
            }),
        );
        output
    }

    fn fract(self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let output = Rev64::new(self_data.value.fract());
        graph.add_operation(
            output.id,
            vec![self.id],
            Arc::new(|o_grad, v: &mut Rev64Data| {
                v.grad += o_grad;
            }),
        );
        output
    }

    fn abs(self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let output = Rev64::new(self_data.value.abs());
        graph.add_operation(
            output.id,
            vec![self.id],
            Arc::new(|o_grad, v: &mut Rev64Data| {
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
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let output = Rev64::new(self_data.value.signum());
        graph.add_operation(
            output.id,
            vec![self.id],
            Arc::new(|_o_grad, v: &mut Rev64Data| {
                v.grad = 0.0;
            }),
        );
        output
    }

    fn is_sign_positive(self) -> bool {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        self_data.value.is_sign_positive()
    }

    fn is_sign_negative(self) -> bool {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        self_data.value.is_sign_negative()
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let a_data = graph.get_variable(a.id);
        let b_data = graph.get_variable(b.id);
        let output = Rev64::new(self_data.value.mul_add(a_data.value, b_data.value));
        graph.add_operation(
            output.id,
            vec![self.id, a.id, b.id],
            Arc::new(move |o_grad, v: &mut Rev64Data| {
                if v.id == self.id {
                    v.grad += o_grad * a_data.value;
                } else if v.id == a.id {
                    v.grad += o_grad * self_data.value;
                } else if v.id == b.id {
                    v.grad += o_grad;
                }
            }),
        );
        output
    }

    fn recip(self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let output = Rev64::new(self_data.value.recip());
        graph.add_operation(
            output.id,
            vec![self.id],
            Arc::new(|o_grad, v: &mut Rev64Data| {
                v.grad -= o_grad / (v.value * v.value);
            }),
        );
        output
    }

    fn powi(self, n: i32) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let output = Rev64::new(self_data.value.powi(n));
        graph.add_operation(
            output.id,
            vec![self.id],
            Arc::new(move |o_grad, v: &mut Rev64Data| {
                v.grad += o_grad * (n as f64) * self_data.value.powi(n - 1);
            }),
        );
        output
    }

    fn powf(self, n: Self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let n_data = graph.get_variable(n.id);
        let output = Rev64::new(self_data.value.powf(n_data.value));
        graph.add_operation(
            output.id,
            vec![self.id, n.id],
            Arc::new(move |o_grad, v: &mut Rev64Data| {
                if v.id == self.id {
                    v.grad += o_grad * n_data.value * self_data.value.powf(n_data.value - 1.0);
                } else if v.id == n.id {
                    v.grad += o_grad * self_data.value.powf(n_data.value) * self_data.value.ln();
                }
            }),
        );
        output
    }

    fn sqrt(self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let output = Rev64::new(self_data.value.sqrt());
        graph.add_operation(
            output.id,
            vec![self.id],
            Arc::new(|o_grad, v: &mut Rev64Data| {
                v.grad += o_grad / (2.0 * v.value.sqrt());
            }),
        );
        output
    }

    fn exp(self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let output = Rev64::new(self_data.value.exp());
        graph.add_operation(
            output.id,
            vec![self.id],
            Arc::new(|o_grad, v: &mut Rev64Data| {
                v.grad += o_grad * v.value.exp();
            }),
        );
        output
    }

    fn exp2(self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let output = Rev64::new(self_data.value.exp2());
        graph.add_operation(
            output.id,
            vec![self.id],
            Arc::new(|o_grad, v: &mut Rev64Data| {
                v.grad += o_grad * v.value.exp2() * std::f64::consts::LN_2;
            }),
        );
        output
    }

    fn ln(self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let output = Rev64::new(self_data.value.ln());
        graph.add_operation(
            output.id,
            vec![self.id],
            Arc::new(|o_grad, v: &mut Rev64Data| {
                v.grad += o_grad / v.value;
            }),
        );
        output
    }

    fn log(self, base: Self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let base_data = graph.get_variable(base.id);
        let output = Rev64::new(self_data.value.log(base_data.value));
        graph.add_operation(
            output.id,
            vec![self.id, base.id],
            Arc::new(move |o_grad, v: &mut Rev64Data| {
                if v.id == self.id {
                    v.grad += o_grad / (self_data.value * base_data.value.ln());
                } else if v.id == base.id {
                    v.grad -=
                        o_grad * self_data.value.ln() / (base_data.value * base_data.value.ln());
                }
            }),
        );
        output
    }

    fn log2(self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let output = Rev64::new(self_data.value.log2());
        graph.add_operation(
            output.id,
            vec![self.id],
            Arc::new(|o_grad, v: &mut Rev64Data| {
                v.grad += o_grad / (v.value * std::f64::consts::LN_2);
            }),
        );
        output
    }

    fn log10(self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let output = Rev64::new(self_data.value.log10());
        graph.add_operation(
            output.id,
            vec![self.id],
            Arc::new(|o_grad, v: &mut Rev64Data| {
                v.grad += o_grad / (v.value * std::f64::consts::LN_10);
            }),
        );
        output
    }

    fn max(self, other: Self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let other_data = graph.get_variable(other.id);
        if self_data.value > other_data.value {
            self
        } else {
            other
        }
    }

    fn min(self, other: Self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let other_data = graph.get_variable(other.id);
        if self_data.value < other_data.value {
            self
        } else {
            other
        }
    }

    fn abs_sub(self, other: Self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let other_data = graph.get_variable(other.id);
        Rev64::new((self_data.value - other_data.value).abs_sub(other_data.value))
    }

    fn cbrt(self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let output = Rev64::new(self_data.value.cbrt());
        graph.add_operation(
            output.id,
            vec![self.id],
            Arc::new(|o_grad, v: &mut Rev64Data| {
                v.grad += o_grad / (3.0 * v.value.powf(2. / 3.));
            }),
        );
        output
    }

    fn hypot(self, other: Self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let other_data = graph.get_variable(other.id);
        let output = Rev64::new(self_data.value.hypot(other_data.value));
        graph.add_operation(
            output.id,
            vec![self.id, other.id],
            Arc::new(move |o_grad, v: &mut Rev64Data| {
                if v.id == self.id {
                    v.grad += o_grad * self_data.value / self_data.value.hypot(other_data.value);
                } else if v.id == other.id {
                    v.grad += o_grad * other_data.value / self_data.value.hypot(other_data.value);
                }
            }),
        );
        output
    }

    fn sin(self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let output = Rev64::new(self_data.value.sin());

        graph.add_operation(
            output.id,
            vec![self.id],
            Arc::new(move |o_grad, v: &mut Rev64Data| {
                v.grad += o_grad * self_data.value.cos();
            }),
        );
        output
    }

    fn cos(self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let output = Rev64::new(self_data.value.cos());
        graph.add_operation(
            output.id,
            vec![self.id],
            Arc::new(move |o_grad, v: &mut Rev64Data| {
                v.grad -= o_grad * self_data.value.sin();
            }),
        );
        output
    }

    fn tan(self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let output = Rev64::new(self_data.value.tan());
        graph.add_operation(
            output.id,
            vec![self.id],
            Arc::new(move |o_grad, v: &mut Rev64Data| {
                v.grad += o_grad / (self_data.value.cos() * self_data.value.cos());
            }),
        );
        output
    }

    fn asin(self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let output = Rev64::new(self_data.value.asin());
        graph.add_operation(
            output.id,
            vec![self.id],
            Arc::new(move |o_grad, v: &mut Rev64Data| {
                v.grad += o_grad / (1.0 - self_data.value * self_data.value).sqrt();
            }),
        );
        output
    }

    fn acos(self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let output = Rev64::new(self_data.value.acos());
        graph.add_operation(
            output.id,
            vec![self.id],
            Arc::new(move |o_grad, v: &mut Rev64Data| {
                v.grad -= o_grad / (1.0 - self_data.value * self_data.value).sqrt();
            }),
        );
        output
    }

    fn atan(self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let output = Rev64::new(self_data.value.atan());
        graph.add_operation(
            output.id,
            vec![self.id],
            Arc::new(move |o_grad, v: &mut Rev64Data| {
                v.grad += o_grad / (1.0 + self_data.value * self_data.value);
            }),
        );
        output
    }

    fn atan2(self, other: Self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let other_data = graph.get_variable(other.id);
        let output = Rev64::new(self_data.value.atan2(other_data.value));
        graph.add_operation(
            output.id,
            vec![self.id, other.id],
            Arc::new(move |o_grad, v: &mut Rev64Data| {
                if v.id == self.id {
                    v.grad += o_grad * other_data.value
                        / (self_data.value * self_data.value + other_data.value * other_data.value);
                } else if v.id == other.id {
                    v.grad -= o_grad * self_data.value
                        / (self_data.value * self_data.value + other_data.value * other_data.value);
                }
            }),
        );
        output
    }

    fn sin_cos(self) -> (Self, Self) {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let (sin_value, cos_value) = self_data.value.sin_cos();
        let sin_output = Rev64::new(sin_value);
        let cos_output = Rev64::new(cos_value);
        graph.add_operation(
            sin_output.id,
            vec![self.id],
            Arc::new(move |o_grad, v: &mut Rev64Data| {
                v.grad += o_grad * cos_value;
            }),
        );
        graph.add_operation(
            cos_output.id,
            vec![self.id],
            Arc::new(move |o_grad, v: &mut Rev64Data| {
                v.grad -= o_grad * sin_value;
            }),
        );
        (sin_output, cos_output)
    }

    fn exp_m1(self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let output = Rev64::new(self_data.value.exp_m1());
        graph.add_operation(
            output.id,
            vec![self.id],
            Arc::new(move |o_grad, v: &mut Rev64Data| {
                v.grad += o_grad * (v.value.exp());
            }),
        );
        output
    }

    fn ln_1p(self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let output = Rev64::new(self_data.value.ln_1p());
        graph.add_operation(
            output.id,
            vec![self.id],
            Arc::new(move |o_grad, v: &mut Rev64Data| {
                v.grad += o_grad / (1.0 + self_data.value);
            }),
        );
        output
    }

    fn sinh(self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let output = Rev64::new(self_data.value.sinh());
        graph.add_operation(
            output.id,
            vec![self.id],
            Arc::new(move |o_grad, v: &mut Rev64Data| {
                v.grad += o_grad * self_data.value.cosh();
            }),
        );
        output
    }

    fn cosh(self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let output = Rev64::new(self_data.value.cosh());
        graph.add_operation(
            output.id,
            vec![self.id],
            Arc::new(move |o_grad, v: &mut Rev64Data| {
                v.grad += o_grad * self_data.value.sinh();
            }),
        );
        output
    }

    fn tanh(self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let output = Rev64::new(self_data.value.tanh());
        graph.add_operation(
            output.id,
            vec![self.id],
            Arc::new(move |o_grad, v: &mut Rev64Data| {
                v.grad += o_grad * (1.0 - self_data.value.tanh() * self_data.value.tanh());
            }),
        );
        output
    }

    fn asinh(self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let output = Rev64::new(self_data.value.asinh());
        graph.add_operation(
            output.id,
            vec![self.id],
            Arc::new(move |o_grad, v: &mut Rev64Data| {
                v.grad += o_grad / (self_data.value * self_data.value + 1.0).sqrt();
            }),
        );
        output
    }

    fn acosh(self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let output = Rev64::new(self_data.value.acosh());
        graph.add_operation(
            output.id,
            vec![self.id],
            Arc::new(move |o_grad, v: &mut Rev64Data| {
                v.grad += o_grad / (self_data.value * self_data.value - 1.0).sqrt();
            }),
        );
        output
    }

    fn atanh(self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let output = Rev64::new(self_data.value.atanh());
        graph.add_operation(
            output.id,
            vec![self.id],
            Arc::new(move |o_grad, v: &mut Rev64Data| {
                v.grad += o_grad / (1.0 - self_data.value * self_data.value);
            }),
        );
        output
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        self_data.value.integer_decode()
    }

    fn to_degrees(self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let output = Rev64::new(self_data.value.to_degrees());
        graph.add_operation(
            output.id,
            vec![self.id],
            Arc::new(|_o_grad, v: &mut Rev64Data| {
                // v.grad += 0.0; // this is incorrect
                v.grad += 180.0 / std::f64::consts::PI;
            }),
        );
        output
    }

    fn to_radians(self) -> Self {
        let graph = Graph::instance();
        let self_data = graph.get_variable(self.id);
        let output = Rev64::new(self_data.value.to_radians());
        graph.add_operation(
            output.id,
            vec![self.id],
            Arc::new(|_o_grad, v: &mut Rev64Data| {
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

    const NUM_CASES: u32 = 1000;
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
        let var_x = Rev64::new(x);
        let var_y = Rev64::new(y);
        let mut result = op(var_x, var_y);

        result.backward();
        let grad_x = var_x.get_grad();
        let grad_y = var_y.get_grad();

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
        let var_x = Rev64::new(x);
        let mut result = op(var_x);

        result.backward();
        let grad_x = var_x.get_grad();

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
            let var_x = Rev64::new(x);
            let var_y = Rev64::new(y);
            let var_z = Rev64::new(z);

            let mut result = var_x.mul_add(var_y, var_z);

            result.backward();
            let grad_x = var_x.get_grad();
            let grad_y = var_y.get_grad();
            let grad_z = var_z.get_grad();

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
