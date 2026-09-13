fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file_descriptors = protox::compile(["proto/query_engine.proto"], ["proto"])?;
    tonic_build::configure()
        .compile_fds(file_descriptors)?;
    Ok(())
}
