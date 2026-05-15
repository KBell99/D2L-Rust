use ndarray::{Array, ArrayD, Dimension, IxDyn};
use tch::Tensor;

pub fn ndarray_to_tensor<D>(arr: Array<f64, D>) -> tch::Tensor
where
    D: Dimension,
{
    let tn = Tensor::from_slice(arr.as_slice().unwrap());
    let shape: Vec<i64> = arr.shape().iter().map(|s| *s as i64).collect();
    tn.reshape(&shape)
}

pub fn tensor_to_ndarray(tensor: &Tensor) -> ArrayD<f64> {
    let shape: Vec<usize> = tensor.size().iter().map(|s| *s as usize).collect();
    let numel: usize = shape.iter().product();
    let mut v = vec![0.0f64; numel];
    tensor
        .to_kind(tch::Kind::Double)
        .reshape(&[-1])
        .copy_data(&mut v, numel);
    ArrayD::from_shape_vec(IxDyn(&shape), v).unwrap()
}
