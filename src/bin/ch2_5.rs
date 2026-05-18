use tch::{Device, Kind, Tensor};

fn main() {
    //2.5.1 A Simple Function
    let mut x = Tensor::arange(4.0, (Kind::Float, Device::cuda_if_available()));
    x.print();

    x.requires_grad_(true);
    x.grad().print();

    let y = Tensor::from(2) * x.dot(&x);
    y.print();

    y.backward();
    x.grad().print();

    //x.grad == 4 * x
    x.grad().eq_tensor(&(Tensor::from(4) * &x)).print();

    //Reset gradient buffer
    x.zero_grad();
    let y = x.sum(Kind::Float);
    y.backward();
    x.grad().print();

    //2.5.2
    x.zero_grad();
    let y = (&x * &x).sum(Kind::Float);
    y.backward();
    x.grad().print();

    //2.5.3 Detaching Computation
    x.zero_grad();
    let y = &x * &x;
    let u = y.detach();
    let z = (&u * &x);

    z.sum(Kind::Float).backward();
    x.grad().eq_tensor(&u).print();

    x.zero_grad();
    y.sum(Kind::Float).backward();
    x.grad().eq_tensor(&(Tensor::from(2.0) * x)).print();
}
