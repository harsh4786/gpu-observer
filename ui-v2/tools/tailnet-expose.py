#!/usr/bin/env python3
"""Re-expose the loopback-bound demo services on another local address.

The UI page and the three observer sockets bind to 127.0.0.1, which makes them
reachable only through an SSH tunnel. When that tunnel drops the page simply
stops loading, with nothing wrong on the server. This forwards each port from a
second address -- in practice the host's Tailscale address -- so a browser on
another device in the same tailnet can reach them directly.

It is purely additive: nothing it forwards to is restarted, reconfigured, or
stopped, and killing this process restores the loopback-only state exactly.

The browser decides where to send the WebSocket and control requests from
location.hostname (see ui-v2/trace.js), so loading the page from this address
makes every other socket follow it automatically. vLLM already listens on
0.0.0.0 and needs no forwarding.

    python3 ui-v2/tools/tailnet-expose.py --bind 100.103.230.46

Exposes the page to every device on that network, so bind to a private
interface, never to 0.0.0.0 on an untrusted network.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib

DEFAULT_PORTS = [8088, 8089, 8090, 8091]


async def pump(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                break
            writer.write(chunk)
            await writer.drain()
    except (ConnectionResetError, BrokenPipeError, TimeoutError):
        pass
    finally:
        # Half-close so the far side sees EOF rather than hanging on a socket
        # that will never produce more bytes.
        with contextlib.suppress(Exception):
            writer.close()


def make_handler(port: int, target: str):
    async def handle(client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter) -> None:
        try:
            server_reader, server_writer = await asyncio.open_connection(target, port)
        except OSError as error:
            print(f"  {port}: upstream refused ({error})", flush=True)
            with contextlib.suppress(Exception):
                client_writer.close()
            return
        # Both directions run concurrently; the first EOF closes its own half and
        # the other finishes on its own, which is what a WebSocket upgrade needs.
        await asyncio.gather(
            pump(client_reader, server_writer),
            pump(server_reader, client_writer),
        )

    return handle


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", required=True, help="address to listen on, e.g. the Tailscale IP")
    parser.add_argument("--target", default="127.0.0.1", help="address to forward to (default 127.0.0.1)")
    parser.add_argument("--ports", type=int, nargs="+", default=DEFAULT_PORTS)
    args = parser.parse_args()

    servers = []
    for port in args.ports:
        try:
            server = await asyncio.start_server(make_handler(port, args.target), args.bind, port)
        except OSError as error:
            print(f"  {port}: cannot bind ({error})", flush=True)
            continue
        servers.append(server)
        print(f"  {args.bind}:{port} -> {args.target}:{port}", flush=True)

    if not servers:
        raise SystemExit("nothing bound")

    print(f"\nopen http://{args.bind}:8088/ui-v2/trace.html", flush=True)
    await asyncio.gather(*(server.serve_forever() for server in servers))


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
