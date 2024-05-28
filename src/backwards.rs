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
}

impl Add for Rev64 {
    type Output = Rev64;

    fn add(self, rhs: Rev64) -> Rev64 {
        let output = Rev64::new(self.value + rhs.value);
        Graph::instance().add_operation(
            output.id,
            vec![self.id, rhs.id],
            Arc::new(|o_grad, a: &mut Rev64| {
                // println!("Backward pass - add: updating gradient of {}", a.id);
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
                // println!("Backward pass - mul: updating gradient of {}", v.id);
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
                // println!("Backward pass - rem: updating gradient of {}", v.id);
                if v.id == self.id {
                    v.grad += o_grad % other.value;
                } else {
                    v.grad += o_grad % self.value;
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
                // println!("Backward pass - div: updating gradient of {}", v.id);
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
                // println!("Backward pass - sub: updating gradient of {}", v.id);
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
                // println!("Backward pass - neg: updating gradient of {}", v.id);
                v.grad -= o_grad;
            }),
        );
        output
    }
}

// Graph structure to manage the computational graph
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
                println!("Backward pass - scalar mul: updating gradient of {}", b.id);
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

// Implementing Float for RcVar by delegating to the inner value
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
        Rev64::new(self.value.floor())
    }

    fn ceil(self) -> Self {
        Rev64::new(self.value.ceil())
    }

    fn round(self) -> Self {
        Rev64::new(self.value.round())
    }

    fn trunc(self) -> Self {
        Rev64::new(self.value.trunc())
    }

    fn fract(self) -> Self {
        Rev64::new(self.value.fract())
    }

    fn abs(self) -> Self {
        Rev64::new(self.value.abs())
    }

    fn signum(self) -> Self {
        Rev64::new(self.value.signum())
    }

    fn is_sign_positive(self) -> bool {
        self.value.is_sign_positive()
    }

    fn is_sign_negative(self) -> bool {
        self.value.is_sign_negative()
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        Rev64::new(self.value.mul_add(a.value, b.value))
    }

    fn recip(self) -> Self {
        Rev64::new(self.value.recip())
    }

    fn powi(self, n: i32) -> Self {
        Rev64::new(self.value.powi(n))
    }

    fn powf(self, n: Self) -> Self {
        Rev64::new(self.value.powf(n.value))
    }

    fn sqrt(self) -> Self {
        Rev64::new(self.value.sqrt())
    }

    fn exp(self) -> Self {
        Rev64::new(self.value.exp())
    }

    fn exp2(self) -> Self {
        Rev64::new(self.value.exp2())
    }

    fn ln(self) -> Self {
        Rev64::new(self.value.ln())
    }

    fn log(self, base: Self) -> Self {
        Rev64::new(self.value.log(base.value))
    }

    fn log2(self) -> Self {
        Rev64::new(self.value.log2())
    }

    fn log10(self) -> Self {
        Rev64::new(self.value.log10())
    }

    fn max(self, other: Self) -> Self {
        Rev64::new(self.value.max(other.value))
    }

    fn min(self, other: Self) -> Self {
        Rev64::new(self.value.min(other.value))
    }

    fn abs_sub(self, other: Self) -> Self {
        Rev64::new(self.value.abs_sub(other.value))
    }

    fn cbrt(self) -> Self {
        Rev64::new(self.value.cbrt())
    }

    fn hypot(self, other: Self) -> Self {
        Rev64::new(self.value.hypot(other.value))
    }

    fn sin(self) -> Self {
        Rev64::new(self.value.sin())
    }

    fn cos(self) -> Self {
        Rev64::new(self.value.cos())
    }

    fn tan(self) -> Self {
        Rev64::new(self.value.tan())
    }

    fn asin(self) -> Self {
        Rev64::new(self.value.asin())
    }

    fn acos(self) -> Self {
        Rev64::new(self.value.acos())
    }

    fn atan(self) -> Self {
        Rev64::new(self.value.atan())
    }

    fn atan2(self, other: Self) -> Self {
        Rev64::new(self.value.atan2(other.value))
    }

    fn sin_cos(self) -> (Self, Self) {
        let (sin, cos) = self.value.sin_cos();
        (Rev64::new(sin), Rev64::new(cos))
    }

    fn exp_m1(self) -> Self {
        Rev64::new(self.value.exp_m1())
    }

    fn ln_1p(self) -> Self {
        Rev64::new(self.value.ln_1p())
    }

    fn sinh(self) -> Self {
        Rev64::new(self.value.sinh())
    }

    fn cosh(self) -> Self {
        Rev64::new(self.value.cosh())
    }

    fn tanh(self) -> Self {
        Rev64::new(self.value.tanh())
    }

    fn asinh(self) -> Self {
        Rev64::new(self.value.asinh())
    }

    fn acosh(self) -> Self {
        Rev64::new(self.value.acosh())
    }

    fn atanh(self) -> Self {
        Rev64::new(self.value.atanh())
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        self.value.integer_decode()
    }

    fn to_degrees(self) -> Self {
        Rev64::new(self.value.to_degrees())
    }

    fn to_radians(self) -> Self {
        Rev64::new(self.value.to_radians())
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
        todo!()
    }
}
