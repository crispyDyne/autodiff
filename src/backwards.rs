use std::{
    cell::RefCell,
    fmt,
    ops::{Add, Deref, Mul},
    rc::Rc,
};

pub struct RcVar(Rc<Var>);

impl Deref for RcVar {
    type Target = Var;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Clone for RcVar {
    fn clone(&self) -> Self {
        RcVar(Rc::clone(&self.0))
    }
}

/// A node in the computational graph.
pub struct Var {
    value: f64,
    pub grad: RefCell<f64>,                      // Gradient of the variable
    backward: RefCell<Option<Box<dyn Fn(f64)>>>, // Function to be called in the backward pass
    predecessors: RefCell<Vec<RcVar>>,           // Predecessor variables in the computation graph
}

impl fmt::Debug for RcVar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Var")
            .field("value", &self.value)
            .field("grad", &self.grad.borrow())
            .field(
                "backward",
                &format_args!(
                    "{}",
                    if self.backward.borrow().is_some() {
                        "Some(...)"
                    } else {
                        "None"
                    }
                ),
            )
            // Print the predecessors
            .field(
                "predecessors",
                &format_args!(
                    "{:?}",
                    self.predecessors
                        .borrow()
                        .iter()
                        .map(|v| v.value)
                        .collect::<Vec<_>>()
                ),
            )
            .finish()
    }
}

impl RcVar {
    pub fn new(value: f64) -> RcVar {
        RcVar(Rc::new(Var {
            value,
            grad: RefCell::new(0.0),
            backward: RefCell::new(None),
            predecessors: RefCell::new(vec![]),
        }))
    }

    /// Performs the backward pass from this variable.
    pub fn backward(self: RcVar) {
        *self.grad.borrow_mut() = 1.0; // Seed the gradient of the output variable with 1

        // Use a queue to manage the nodes that need their gradients updated
        let mut agenda: Vec<RcVar> = vec![self.clone()];

        // Visit each variable in the agenda to perform the backward pass
        while let Some(var) = agenda.pop() {
            // Execute the backward function for the current var
            println!("Visiting {:?}", var);
            if let Some(backward_fn) = var.backward.borrow_mut().take() {
                let var_grad = *var.grad.borrow();
                backward_fn(var_grad);

                // Push predecessors to the agenda
                for pred in var.predecessors.borrow().iter() {
                    agenda.push(pred.clone());
                }
            }
        }
    }
}

impl Add for &RcVar {
    type Output = RcVar;

    fn add(self, other: &RcVar) -> RcVar {
        let output = RcVar::new(self.value + other.value);
        {
            let mut predecessors = output.predecessors.borrow_mut();

            predecessors.push(self.clone());
            predecessors.push(other.clone());
        }
        let back = {
            let a = self.clone();
            let b = other.clone();
            Box::new(move |o_grad: f64| {
                println!("Backward pass through add: updating gradients of a and b");
                *a.grad.borrow_mut() += o_grad;
                *b.grad.borrow_mut() += o_grad;
            })
        };
        *output.backward.borrow_mut() = Some(back);
        output
    }
}

impl Mul for &RcVar {
    type Output = RcVar;

    fn mul(self, other: &RcVar) -> RcVar {
        let output = RcVar::new(self.value * other.value);
        {
            let mut predecessors = output.predecessors.borrow_mut();
            predecessors.push(self.clone());
            predecessors.push(other.clone());
        }
        let back = {
            let a = self.clone();
            let b = other.clone();
            let a_value = self.value;
            let b_value = other.value;
            Box::new(move |o_grad: f64| {
                println!("Backward pass through mul: updating gradients of a and b");
                *a.grad.borrow_mut() += o_grad * b_value;
                *b.grad.borrow_mut() += o_grad * a_value;
            })
        };
        *output.backward.borrow_mut() = Some(back);
        output
    }
}

// implement multiplication for f64 and RcVar
impl Mul<&RcVar> for f64 {
    type Output = RcVar;

    fn mul(self, other: &RcVar) -> RcVar {
        let output = RcVar::new(self * other.value);
        {
            let mut predecessors = output.predecessors.borrow_mut();
            predecessors.push(other.clone());
        }
        let back = {
            let b = other.clone();
            Box::new(move |o_grad: f64| {
                println!("Backward pass through mul: updating gradient of b");
                *b.grad.borrow_mut() += o_grad * self;
            })
        };
        *output.backward.borrow_mut() = Some(back);
        output
    }
}
