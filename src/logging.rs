pub fn init_logger() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .with_file(true)
        .with_line_number(true)
        .init();
}
