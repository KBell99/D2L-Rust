use tch::{Device, IndexOp, Kind, Scalar, Tensor};

fn main() {
    //2.3.1 Scalars
    let x = Tensor::from(3.0);
    let y = Tensor::from(2.0);

    println!("{}", &x + &y);
    println!("{}", &x * &y);
    println!("{}", &x / &y);
    println!("{}", &x.pow(&y));

    //2.3.2 Vectors
    let x = Tensor::arange(3, (Kind::Float, Device::cuda_if_available()));
    x.print();

    x.i(2).print();
    println!("{:?}", x.size());

    //2.3.3 Matrices
    let A = Tensor::arange(6, (Kind::Float, Device::cuda_if_available())).view((3, 2));
    A.print();

    A.tr().print();

    let A = Tensor::from_slice(&[1, 2, 3, 2, 0, 4, 3, 4, 5]).view((3, 3));
    A.eq_tensor(&A.tr()).print();

    //2.3.4 Tensors
    Tensor::arange(24, (Kind::Float, Device::cuda_if_available()))
        .view((2, 3, 4))
        .print();

    //2.3.5 Basic Properties of Tensor Arithmetic
    let A = Tensor::arange(6, (Kind::Float, Device::cuda_if_available())).view((2, 3));
    let B = A.copy();
    A.print();
    (&A + &B).print();
    // Hadamard Product
    (&A * &B).print();

    let a = Tensor::from(2.0);
    let X = Tensor::arange(24, (Kind::Float, Device::cuda_if_available())).view((2, 3, 4));
    (&a + &X).print();
    (&a * &X).print();

    //2.3.6 Reduction
    let x = Tensor::arange(3, (Kind::Float, Device::cuda_if_available()));
    x.print();
    x.sum(Kind::Float).print();

    println!("A tensor shape: {:?}", A.size());
    A.sum(Kind::Float).print();

    println!("A tensor shape: {:?}", A.size());
    println!(
        "A tensor sum on axis 0 {:?}",
        A.sum_dim_intlist(0, false, Kind::Float).size()
    );
    println!(
        "A tensor sum on axis 1 {:?}",
        A.sum_dim_intlist(1, false, Kind::Float).size()
    );

    A.sum_dim_intlist(&vec![0, 1], false, Kind::Float)
        .eq_tensor(&A.sum(Kind::Float))
        .print();

    A.mean(Kind::Float).print();
    (A.sum(Kind::Float) / Tensor::from(A.numel() as f64)).print();

    A.mean_dim(0, false, Kind::Float).print();
    (A.sum_dim_intlist(0, false, Kind::Float) / A.size()[0] as f64).print();

    //2.3.7 Non-Reduction Sum
    let sum_A = A.sum_dim_intlist(1, true, Kind::Float);
    sum_A.print();
    println!("sum_A size: {:?}", sum_A.size());

    (&A / sum_A).print();

    A.cumsum(0, Kind::Float).print();

    //2.3.8 Dot Products
    let y = Tensor::ones(3, (Kind::Float, Device::cuda_if_available()));
    x.print();
    y.print();
    x.dot(&y).print();

    //2.3.9 Matrix-Vector Products
    println!("Shape A: {:?}", A.size());
    println!("Shape x: {:?}", x.size());
    A.mv(&x).print();

    //2.3.10 Matrix-Matrix Multiplication
    let B = Tensor::ones(&vec![3, 4], (Kind::Float, Device::cuda_if_available()));
    A.mm(&B).print();

    //2.3.11 Norms
    let u = Tensor::from_slice(&[3.0, -4.0]);
    u.norm().print();
    u.abs().sum(Kind::Float).print();
    Tensor::ones(&vec![4, 9], (Kind::Float, Device::cuda_if_available()))
        .norm()
        .print();
}
