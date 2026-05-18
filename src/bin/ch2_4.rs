use ndarray::*;
use plotters::prelude::*;

fn main() {
    //2.4.1 Derivatives and Differentiation
    let hs: Array1<f64> = Array::range(-1.0, -6.0, -1.0).mapv(|e| 10.0f64.powf(e));
    for &h in hs.iter() {
        println!(
            "h={:.5}, numerical limit={:.5}",
            h,
            (f(1.0 + h) - f(1.0)) / h
        );
    }

    //2.4.2 Visualization utilities
    plot().unwrap();
}

fn f(x: f64) -> f64 {
    3.0 * x.powf(2.0) - 4.0 * x
}

fn tangent(x: f64) -> f64 {
    2.0 * x - 3.0
}

fn plot() -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("plots/ch2_4.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("f(x) and tangent line at x=1", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0.0f64..3.0f64, -3.0f64..15.0f64)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(
            (0..=300).map(|i| {
                let x = i as f64 / 100.0;
                (x, f(x))
            }),
            &BLUE,
        ))?
        .label("f(x) = 3x² - 4x")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    chart
        .draw_series(LineSeries::new(
            (0..=300).map(|i| {
                let x = i as f64 / 100.0;
                (x, tangent(x))
            }),
            &RED,
        ))?
        .label("Tangent: 2x - 3")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    chart.configure_series_labels().border_style(BLACK).draw()?;

    root.present()?;
    println!("Plot saved to ch2_4.png");

    Ok(())
}
