use ndarray::{Array, ArrayD, Dimension, IxDyn};
use rand::distr::{Distribution, weighted::WeightedIndex};
use tch::Tensor;

// Code attribution: https://github.com/LaurentMazare/tch-rs/issues/52
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

pub trait IntoSize {
    fn into_size(self) -> Option<Vec<usize>>;
}

impl IntoSize for () {
    fn into_size(self) -> Option<Vec<usize>> { None }
}

impl IntoSize for usize {
    fn into_size(self) -> Option<Vec<usize>> { Some(vec![self]) }
}

impl IntoSize for &usize {
    fn into_size(self) -> Option<Vec<usize>> { Some(vec![*self]) }
}

impl IntoSize for (usize, usize) {
    fn into_size(self) -> Option<Vec<usize>> { Some(vec![self.0, self.1]) }
}

impl IntoSize for (usize, usize, usize) {
    fn into_size(self) -> Option<Vec<usize>> { Some(vec![self.0, self.1, self.2]) }
}

pub fn multinomial<S: IntoSize>(probs: &[f64], num_samples: usize, size: S) -> ArrayD<f64> {
    let dist = WeightedIndex::new(probs).unwrap();
    let mut rng = rand::rng();
    let k = probs.len();

    let (num_draws, out_shape) = match size.into_size() {
        None => (1, vec![k]),
        Some(s) => {
            let mut shape = s.clone();
            shape.push(k);
            (s.iter().product(), shape)
        }
    };

    let mut data = Vec::with_capacity(num_draws * k);
    for _ in 0..num_draws {
        let mut counts = vec![0u32; k];
        for _ in 0..num_samples {
            counts[dist.sample(&mut rng)] += 1;
        }
        data.extend(counts.iter().map(|&x| x as f64));
    }

    ArrayD::from_shape_vec(IxDyn(&out_shape), data).unwrap()
}
