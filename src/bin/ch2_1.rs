use tch::{Device, IndexOp, Kind, Tensor};

fn main() {
    // Flatten the data into a single-dimensional slice
    let data = [2, 1, 4, 3, 1, 2, 3, 4, 4, 3, 2, 1];

    // Create the tensor and reshape it to (3, 4)
    let mut tensor = Tensor::arange(12, (Kind::Float, Device::Cuda(0))).view((3, 4));
    let t = tensor.i(1..3);

    // Indexing and Slicing
    tensor.i((..2, ..)).copy_(&Tensor::from(12.0));

    tensor.print();
    t.print();

    //Operations
    tensor = Tensor::exp(&tensor);
    tensor.print();
    let x = Tensor::from_slice(&[1, 2, 4, 8]);
    let y = Tensor::from_slice(&[2, 2, 2, 2]);

    let exp = &x.pow(&y);
    let add = &x + &y;
    let subtract = &x - &y;
    let multiply = &x * &y;
    let divide = &x / &y;

    add.print();
    subtract.print();
    multiply.print();
    divide.print();
    exp.print();

    let X = Tensor::arange(12, (Kind::Float, Device::Cpu)).view((3, 4));
    let Y = Tensor::from_slice(&[2, 1, 4, 3, 1, 2, 3, 4, 4, 3, 2, 1]).view((3, 4));

    let concat_zero = Tensor::cat(&[&X, &Y], 0);
    let concat_one = Tensor::cat(&[&X, &Y], 1);

    concat_zero.print();
    concat_one.print();

    let equality = &X.eq_tensor(&Y);
    equality.print();

    let sum = &X.sum(Kind::Float);

    sum.print();
    //  2. Move it to CUDA
    // Using .to(Device::Cuda(0)) for a specific GPU
    // let cuda_tensor = cpu_tensor.to(Device::Cuda(0));

    //Saving memory
    // performing in-place operations is easy.
    let mut Z = Tensor::zeros_like(&Y);
    println!("id(Z): {}", Z.data_ptr() as usize);
    Z.copy_(&(&X + &Y));
    println!("id(Z): {}", Z.data_ptr() as usize);

    //  Conversion to Other Rust Objects
    let a = Tensor::from(3.5);
    println!("float: {}", a.double_value(&[]));
}
