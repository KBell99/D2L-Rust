use d2l_rust::utils::ndarray_to_tensor;
use polars::prelude::*;
use std::fs;

fn main() {
    // 2.2.1 Reading the Dataset
    match fs::exists("data") {
        Ok(false) => fs::create_dir("data").unwrap(),
        Ok(true) => {}
        Err(e) => panic!("{e}"),
    }
    // assert!(!fs::exists("data").expect("Can't check exists of directory"));

    let data_file_path = "data/house_tiny.csv";

    let _ = fs::write(
        data_file_path,
        "NumRooms,RoofType,Price
NA,NA,127500
2,NA,106000
4,Slate,178100
NA,NA,140000",
    )
    .expect("Can't write to csv.");

    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(data_file_path.into()))
        .expect("Failed to open CSV file")
        .finish()
        .expect("Failed to parse CSV");

    println!("{}", df);

    // 2.2.2 Data preparation: (https://docs.pola.rs/user-guide/migration/pandas/#key-syntax-differences)
    // Split inputs (columns 0-1) and targets (column 2)
    // i.e. inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
    let inputs = df
        .select(["NumRooms", "RoofType"])
        .expect("Failed to select columns");
    let targets = df.select(["Price"]).expect("Failed to select target");

    println!("{}", inputs);

    //inputs = pd.get_dummies(inputs, dummy_na=True)
    let inputs = inputs
        .columns_to_dummies(vec!["RoofType"], None, false, false)
        .expect("Failed to create dummies");

    println!("{}", inputs);

    // pandas
    // inputs = inputs.fillna(inputs.mean())
    // print(inputs)
    let inputs = inputs
        .lazy()
        .with_column(
            col("NumRooms")
                .cast(DataType::Float64)
                .fill_null(col("NumRooms").cast(DataType::Float64).mean()),
        )
        .collect()
        .expect("Failed to fill nulls");

    println!("{}", inputs);

    //2.2.3 Conversion to the Tensor Format
    let inputs_arr = inputs.to_ndarray::<Float64Type>(IndexOrder::C).unwrap();
    let inputs_tensor = ndarray_to_tensor(inputs_arr);

    let targets_arr = targets.to_ndarray::<Float64Type>(IndexOrder::C).unwrap();
    let targets_tensor = ndarray_to_tensor(targets_arr);

    inputs_tensor.print();
    targets_tensor.print();
}
