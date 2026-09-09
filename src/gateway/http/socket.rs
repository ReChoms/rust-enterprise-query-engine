use socket2::{Domain, Protocol, Socket, Type};
use std::net::SocketAddr;

/// Creates a Linux-tuned TCP listener for low-latency, high-throughput query delivery.
///
/// Features:
/// - SO_REUSEADDR and SO_REUSEPORT for zero-cooldown zero-downtime restarts.
/// - TCP_NODELAY: disables Nagle's algorithm to eliminate 40ms delayed-ACK stalls.
/// - 128KB kernel socket buffers to prevent window throttling on large Arrow result sets.
/// - 1024 backlog queue for burst connection absorbs.
pub fn create_tuned_tcp_listener(addr: SocketAddr) -> std::io::Result<tokio::net::TcpListener> {
    let domain = match addr {
        SocketAddr::V4(_) => Domain::IPV4,
        SocketAddr::V6(_) => Domain::IPV6,
    };

    let socket = Socket::new(domain, Type::STREAM, Some(Protocol::TCP))?;

    socket.set_reuse_address(true)?;
    #[cfg(all(unix, not(target_os = "solaris"), not(target_os = "illumos")))]
    socket.set_reuse_port(true)?;

    socket.set_nodelay(true)?;
    socket.set_recv_buffer_size(128 * 1024)?;
    socket.set_send_buffer_size(128 * 1024)?;
    socket.set_nonblocking(true)?;

    socket.bind(&socket2::SockAddr::from(addr))?;
    socket.listen(1024)?;

    let std_listener: std::net::TcpListener = socket.into();
    tokio::net::TcpListener::from_std(std_listener)
}
