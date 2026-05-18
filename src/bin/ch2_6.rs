use d2l_rust::utils::{multinomial, ndarray_to_tensor, tensor_to_ndarray};
use plotters::{prelude::*, style::full_palette::ORANGE};
use rand::RngExt;
use tch::Kind;

fn main() {
    //2.6.1 A simple example: tossing coins
    let num_tosses = 100;
    let heads = (0..num_tosses)
        .filter(|_| rand::rng().random_bool(0.5))
        .count();
    let tails = num_tosses - heads;

    println!("Heads, Tails: {heads} {tails}");

    // tch-rs does not have multinomial distribution implemented
    let fair_probs = vec![0.5_f64, 0.5];
    let counts = multinomial(&fair_probs, 100, ());
    println!("counts: {:?}", counts);
    let proportions = &counts / 100.0;
    println!("proportions: {:?}", proportions);
    let counts = multinomial(&fair_probs, 10000, ());
    println!("{:?}", &counts / 10000.0);

    let counts = multinomial(&fair_probs, 1, &100000_usize);
    let count_tensor = ndarray_to_tensor(counts);
    let cum_counts = count_tensor.cumsum(0, Kind::Float);
    let estimates = &cum_counts / &cum_counts.sum_dim_intlist(&vec![1], true, Kind::Float);
    let estimates = tensor_to_ndarray(&estimates);

    let n = estimates.shape()[0];
    let heads_est: Vec<(f32, f32)> = (0..n)
        .map(|i| (i as f32, estimates[[i, 0]] as f32))
        .collect();
    let tails_est: Vec<(f32, f32)> = (0..n)
        .map(|i| (i as f32, estimates[[i, 1]] as f32))
        .collect();

    let root = BitMapBackend::new("plots/ch2_6_coin_estimates.png", (900, 700)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0f32..n as f32, 0f32..1f32)
        .unwrap();

    chart
        .configure_mesh()
        .x_desc("Samples")
        .y_desc("Estimated probability")
        .draw()
        .unwrap();

    chart
        .draw_series(LineSeries::new(heads_est, &BLUE))
        .unwrap()
        .label("P(coin=heads)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .draw_series(LineSeries::new(tails_est, &ORANGE))
        .unwrap()
        .label("P(coin=tails)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &ORANGE));

    // Dashed horizontal line at y=0.5
    let dash = 2000usize;
    let gap = 2000usize;
    let mut pos = 0;
    while pos < n {
        let end = (pos + dash).min(n);
        chart
            .draw_series(LineSeries::new(
                vec![(pos as f32, 0.5f32), (end as f32, 0.5f32)],
                &BLACK,
            ))
            .unwrap();
        pos += dash + gap;
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()
        .unwrap();

    root.present().unwrap();
    println!("Saved to plots/ch2_6_coin_estimates.png");
}
