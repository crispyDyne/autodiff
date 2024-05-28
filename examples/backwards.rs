use autodiff::backwards::{Graph, Rev64};

fn main() {
    let x = Rev64::new(2.0); // id = 0
    let y = Rev64::new(3.0); // id = 1
    let a = 10.0; // id = 2
    let b = 10.0; // id = 3
    let mut z = (a * x) + y; // (a * x) id = 4
    let mut w = (b * x) * y; // (b * x) id = 5

    // Perform the backward pass wrt `z` and print the gradients
    z.backward();
    println!(
        "Gradient of z w.r.t x: {}",
        Graph::instance().get_variable(x.id).grad
    );
    println!(
        "Gradient of z w.r.t y: {}",
        Graph::instance().get_variable(y.id).grad
    );

    // Reset the gradients
    Graph::instance().reset_gradients();

    // Perform the backward pass wrt `w` and print the gradients
    w.backward();
    println!(
        "Gradient of w w.r.t x: {}",
        Graph::instance().get_variable(x.id).grad
    );
    println!(
        "Gradient of w w.r.t y: {}",
        Graph::instance().get_variable(y.id).grad
    );
}
