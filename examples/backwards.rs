use std::rc::Rc;

use autodiff::backwards::{add, mul, Var};

fn main() {
    let x = Var::new(2.0);
    let y = Var::new(3.0);
    let z = add(Rc::clone(&x), Rc::clone(&y));
    let w = mul(Rc::clone(&x), Rc::clone(&y));

    // // Perform the backward pass wrt `w`
    Rc::clone(&z).backward(); // Trigger the backward pass
    println!("x.grad: {:?}", x.grad.borrow());
    println!("y.grad: {:?}", y.grad.borrow());

    // reset the gradients
    *x.grad.borrow_mut() = 0.0;
    *y.grad.borrow_mut() = 0.0;

    // Perform the backward pass wrt `z`
    w.backward(); // Trigger the backward pass
    println!("x.grad: {:?}", x.grad.borrow()); // 12
    println!("y.grad: {:?}", y.grad.borrow()); // 8
}
