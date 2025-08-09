use nn::graph;

fn main() {
    // Test single input graph (backward compatibility)
    let mut single_graph = graph! {
        input -> sin -> cos -> output
    };

    let (result, derivative) = single_graph.compute(1.0);
    println!(
        "Single input - f(1.0) = {:.6}, f'(1.0) = {:.6}",
        result, derivative
    );

    // Test multi-input graph with type-level arity
    let mut multi_graph = graph! {
        inputs: [x, y, z]
        x -> Pow(2) -> @x_sq
        z -> Cos -> @z_cos
        (@x_sq, @z_cos) -> Add -> @result
        output @result
    };

    let results = multi_graph.compute(&[2.0, 1.0]);
    if let Some((result, derivative)) = results.first() {
        println!(
            "Multi input - f(2.0, 1.0) = {:.6}, f'(2.0, 1.0) = {:.6}",
            result, derivative
        );
    }

    // // Test mixed graph with type-level arity
    // let mut mixed_graph = graph! {
    //     inputs: [x, y]
    //     x -> pow(2) -> sin -> @temp1
    //     y -> cos -> scale(2.0) -> @temp2
    //     (@temp1, @temp2) -> mul -> @result
    //     output @result
    // };

    // let results = mixed_graph.compute(&[1.0, 0.5]);
    // if let Some((result, derivative)) = results.first() {
    //     println!(
    //         "Mixed graph - f(1.0, 0.5) = {:.6}, f'(1.0, 0.5) = {:.6}",
    //         result, derivative
    //     );
    // }

    // Test that type-level arity is enforced at compile time
    // This should cause a compilation error if we try to use wrong arity:
    // let mut invalid_graph = graph! {
    //     inputs: [x, y]
    //     x -> add -> @result  // This should fail - add needs 2 inputs
    //     output @result
    // };
}
