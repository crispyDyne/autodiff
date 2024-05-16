use std::{cell::RefCell, fmt, rc::Rc};

/// A node in the computational graph.
pub struct Var {
    value: f64,
    pub grad: RefCell<f64>,                      // Gradient of the variable
    backward: RefCell<Option<Box<dyn Fn(f64)>>>, // Function to be called in the backward pass
    predecessors: RefCell<Vec<Rc<Var>>>,         // Predecessor variables in the computation graph
}

impl fmt::Debug for Var {
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

impl Var {
    pub fn new(value: f64) -> Rc<Self> {
        Rc::new(Var {
            value,
            grad: RefCell::new(0.0),
            backward: RefCell::new(None),
            predecessors: RefCell::new(vec![]),
        })
    }

    /// Performs the backward pass from this variable.
    pub fn backward(self: Rc<Self>) {
        *self.grad.borrow_mut() = 1.0; // Seed the gradient of the output variable with 1

        // Use a queue to manage the nodes that need their gradients updated
        let mut agenda = vec![self.clone()];

        // Visit each variable in the agenda to perform the backward pass
        while let Some(var) = agenda.pop() {
            // Execute the backward function for the current var
            println!("Visiting {:?}", var);
            if let Some(backward_fn) = var.backward.borrow_mut().take() {
                let var_grad = *var.grad.borrow();
                backward_fn(var_grad);

                // Push predecessors to the agenda
                for pred in var.predecessors.borrow().iter() {
                    agenda.push(Rc::clone(pred));
                }
            }
        }
    }
}

/// Adds two `Var` references and returns a new `Var` reference.
pub fn add(a: Rc<Var>, b: Rc<Var>) -> Rc<Var> {
    let output = Var::new(a.value + b.value);
    {
        let mut predecessors = output.predecessors.borrow_mut();
        predecessors.push(Rc::clone(&a));
        predecessors.push(Rc::clone(&b));
    }
    let back = {
        let a_c = Rc::clone(&a);
        let b_c = Rc::clone(&b);
        Box::new(move |o_grad: f64| {
            println!("Backward pass through add: updating gradients of a and b");
            // Print the values of the gradients
            // println!("a.grad: {:?}", a_c.borrow());
            // println!("b.grad: {:?}", b_grad.borrow());
            // println!("o.grad: {:?}", o_grad);
            *a_c.grad.borrow_mut() += o_grad;
            *b_c.grad.borrow_mut() += o_grad;

            // *a_grad.borrow_mut() += o_grad;
            // *b_grad.borrow_mut() += o_grad;
        })
    };
    *output.backward.borrow_mut() = Some(back);
    output
}

/// Multiplies two `Var` references and returns a new `Var` reference.
pub fn mul(a: Rc<Var>, b: Rc<Var>) -> Rc<Var> {
    let output = Var::new(a.value * b.value);
    {
        let mut predecessors = output.predecessors.borrow_mut();
        predecessors.push(Rc::clone(&a));
        predecessors.push(Rc::clone(&b));
    }
    let back = {
        let a_c = Rc::clone(&a);
        let b_c = Rc::clone(&b);
        let b_value = b.value;
        let a_value = a.value;
        Box::new(move |o_grad: f64| {
            println!("Backward pass through mul: updating gradients of a and b");
            // Print the values of the gradients
            // println!("a.grad: {:?}", a_c.grad.borrow());
            // println!("b.grad: {:?}", b_c.grad.borrow());
            // println!("o.grad: {:?}", o_grad);
            *a_c.grad.borrow_mut() += o_grad * b_value;
            *b_c.grad.borrow_mut() += o_grad * a_value;
        })
    };
    *output.backward.borrow_mut() = Some(back);
    output
}
