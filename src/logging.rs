use std::sync::Once;

static LOGGER_INIT: Once = Once::new();

pub fn init_logger() {
    LOGGER_INIT.call_once(|| {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_target(false)
            .with_file(true)
            .with_line_number(true)
            .init();
    });
}
