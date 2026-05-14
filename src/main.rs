use tch::{Device, Kind, Tensor};

fn main() {
    // Flatten the data into a single-dimensional slice
    let data = [2, 1, 4, 3, 1, 2, 3, 4, 4, 3, 2, 1];

    // Create the tensor and reshape it to (3, 4)
    let t = Tensor::arange(12, (Kind::Float, Device::Cuda(0))).resize([3, 4]);

    t.print();
    println!("{}", t.get(-1))
}
