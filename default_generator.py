import sys
import csv
import socket
import argparse
import signal
import time
from pathlib import Path
from datetime import datetime, timezone

try:
    from scapy.all import PcapReader, sniff, IP, TCP, UDP, DNS, Raw
    from scapy.layers.http import HTTP, HTTPRequest, HTTPResponse
    from scapy.layers.tls.all import TLS, TLSClientHello
except ImportError:
    print("[ERROR] Scapy no está instalado. Ejecuta: pip install scapy", file=sys.stderr)
    sys.exit(1)


# -------------------------------------------------------------------
# Columnas del CSV — compatibles con preprocess.py
# -------------------------------------------------------------------
CSV_COLS = [
    "timestamp",
    "frame_number",
    "frame_len",
    "frame_protocols",
    "ip_src",
    "ip_dst",
    "tcp_srcport",
    "tcp_dstport",
    "tcp_flags",
    "udp_srcport",
    "udp_dstport",
    "dns_qry_name",
    "tls_handshake_extensions_server_name",
    "http_request_method",
    "http_request_uri",
    "http_response_code",
]


def get_local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def format_tcp_flags(pkt) -> str | None:
    if TCP not in pkt:
        return None
    return hex(int(pkt[TCP].flags))


def get_frame_protocols(pkt) -> str:
    name_map = {
        "ether":        "eth",
        "ip":           "ip",
        "tcp":          "tcp",
        "udp":          "udp",
        "dns":          "dns",
        "httprequest":  "http",
        "httpresponse": "http",
        "tls":          "tls",
        "raw":          "data",
    }
    layers = []
    layer = pkt
    while layer:
        name = layer.__class__.__name__.lower()
        layers.append(name_map.get(name, name))
        layer = layer.payload if layer.payload.__class__.__name__ != "NoPayload" else None
    return ":".join(layers)


def extract_sni(pkt) -> str | None:
    try:
        if TLSClientHello not in pkt:
            return None
        for ext in (pkt[TLSClientHello].ext or []):
            if hasattr(ext, "type") and ext.type == 0:
                if hasattr(ext, "servernames"):
                    for sn in ext.servernames:
                        if hasattr(sn, "servername"):
                            return sn.servername.decode("utf-8", errors="replace")
    except Exception:
        pass
    return None


def extract_dns_query(pkt) -> str | None:
    try:
        if DNS not in pkt:
            return None
        dns = pkt[DNS]
        if dns.qd and hasattr(dns.qd, "qname"):
            return dns.qd.qname.decode("utf-8", errors="replace").rstrip(".")
    except Exception:
        pass
    return None


def process_packet(pkt, frame_number: int) -> tuple[dict, int]:
    """
    Extrae todos los campos de un paquete Scapy.

    Returns
    -------
    (row_dict, pkt_len) — row_dict vacío si el paquete no tiene capa IP.
    """
    row = {}

    # Timestamp
    if hasattr(pkt, "time"):
        ts = datetime.fromtimestamp(float(pkt.time), tz=timezone.utc)
        row["timestamp"] = ts.isoformat()
    else:
        row["timestamp"] = datetime.now(tz=timezone.utc).isoformat()

    pkt_len = len(pkt)

    row["frame_number"]    = str(frame_number)
    row["frame_len"]       = str(pkt_len)
    row["frame_protocols"] = get_frame_protocols(pkt)

    # Sin capa IP no sirve para el modelo
    if IP not in pkt:
        return {}, pkt_len

    row["ip_src"] = pkt[IP].src
    row["ip_dst"] = pkt[IP].dst

    if TCP in pkt:
        row["tcp_srcport"] = str(pkt[TCP].sport)
        row["tcp_dstport"] = str(pkt[TCP].dport)
        row["tcp_flags"]   = format_tcp_flags(pkt)

    if UDP in pkt:
        row["udp_srcport"] = str(pkt[UDP].sport)
        row["udp_dstport"] = str(pkt[UDP].dport)

    dns_name = extract_dns_query(pkt)
    if dns_name:
        row["dns_qry_name"] = dns_name

    sni = extract_sni(pkt)
    if sni:
        row["tls_handshake_extensions_server_name"] = sni

    try:
        if HTTPRequest in pkt:
            req    = pkt[HTTPRequest]
            method = getattr(req, "Method", None)
            path   = getattr(req, "Path",   None)
            if method:
                row["http_request_method"] = method.decode("utf-8", errors="replace")
            if path:
                row["http_request_uri"] = path.decode("utf-8", errors="replace")
        if HTTPResponse in pkt:
            code = getattr(pkt[HTTPResponse], "Status_Code", None)
            if code:
                row["http_response_code"] = code.decode("utf-8", errors="replace")
    except Exception:
        pass

    return row, pkt_len


# -------------------------------------------------------------------
# Procesamiento — comparte la misma lógica para PCAP y live
# -------------------------------------------------------------------

def process_packets(packet_iter, server_ip: str) -> tuple[list[dict], int, int]:
    """
    Itera sobre cualquier fuente de paquetes (PcapReader o lista de sniff).

    Returns
    -------
    (rows, input_bytes, output_bytes)
    """
    rows         = []
    input_bytes  = 0
    output_bytes = 0

    for i, pkt in enumerate(packet_iter, 1):
        row, pkt_len = process_packet(pkt, i)

        if not row:
            continue

        ip_src = row.get("ip_src")
        ip_dst = row.get("ip_dst")

        if ip_dst == server_ip:
            input_bytes += pkt_len
        elif ip_src == server_ip:
            output_bytes += pkt_len

        rows.append(row)

        if i % 500 == 0:
            print(f"[INFO] Procesados {i} paquetes...")

    return rows, input_bytes, output_bytes


def write_csv(rows: list[dict], out_base: Path, input_mb: float, output_mb: float) -> Path:
    csv_name = f"{out_base.stem}_({input_mb}_input)_({output_mb}_output).csv"
    csv_path = out_base.parent / csv_name

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    return csv_path


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Captura única — lógica reutilizable desde el loop
# -------------------------------------------------------------------

def capture_once(args, server_ip: str, out_dir: Path) -> None:
    """
    Realiza una captura (PCAP o live) y escribe el CSV resultante.

    Parameters
    ----------
    args      : namespace de argparse
    server_ip : IP local detectada
    out_dir   : directorio de salida
    """

    if args.pcap:
        pcap_path = Path(args.pcap)
        if not pcap_path.exists():
            print(f"[ERROR] No existe el archivo: {pcap_path}", file=sys.stderr)
            sys.exit(1)

        print(f"[INFO] Leyendo PCAP: {pcap_path}")
        with PcapReader(str(pcap_path)) as reader:
            rows, input_bytes, output_bytes = process_packets(reader, server_ip)

        out_base = out_dir / pcap_path.stem

    else:
        iface_info = args.iface or "todas las interfaces"
        print(f"[INFO] Capturando {args.duration}s en {iface_info}...")

        packets  = sniff(iface=args.iface, timeout=args.duration, store=True)
        rows, input_bytes, output_bytes = process_packets(packets, server_ip)

        ts       = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_base = out_dir / f"traffic_{ts}"

    input_mb  = round(input_bytes  / (1024 * 1024), 2)
    output_mb = round(output_bytes / (1024 * 1024), 2)
    csv_path  = write_csv(rows, out_base, input_mb, output_mb)

    print(f"[OK] CSV generado : {csv_path}")
    print(f"[OK] Entrada      : {input_mb} MB")
    print(f"[OK] Salida       : {output_mb} MB")
    print(f"[OK] Filas        : {len(rows)}")


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Captura/convierte tráfico de red a CSV compatible con preprocess.py"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pcap",      metavar="FILE",      help="Leer un archivo PCAP existente")
    group.add_argument("--live",      action="store_true", help="Captura en vivo (requiere permisos de Admin/root)")
    parser.add_argument("--duration", type=int, default=10,
                        help="Duración de cada captura en vivo en segundos (default: 10)")
    parser.add_argument("--iface",    metavar="IFACE",     default=None,
                        help="Interfaz de red para captura en vivo (default: auto)")
    parser.add_argument("--out",      metavar="DIR",       default=".",
                        help="Directorio de salida (default: directorio actual)")

    #  Modo loop 
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Repetir la captura indefinidamente cada --interval minutos"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        metavar="MINUTOS",
        help="Minutos entre capturas en modo --loop (default: 5)"
    )

    args      = parser.parse_args()
    server_ip = get_local_ip()
    out_dir   = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] IP local detectada: {server_ip}")

    #  Modo loop 
    if args.loop:
        if args.pcap:
            print("[ERROR] --loop solo funciona con --live, no con --pcap.", file=sys.stderr)
            sys.exit(1)

        interval_sec = args.interval * 60
        capture_num  = 0

        # Captura limpia con Ctrl+C / SIGTERM
        stop = {"flag": False}

        def _handle_signal(sig, frame):
            print("\n[INFO] Señal recibida — deteniendo el loop tras la captura actual...")
            stop["flag"] = True

        signal.signal(signal.SIGINT,  _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        print(f"[INFO] Modo loop activado — captura cada {args.interval} min "
              f"(duración por captura: {args.duration}s)")
        print("[INFO] Presiona Ctrl+C para detener.\n")

        while not stop["flag"]:
            capture_num += 1
            print(f"[LOOP]  Captura #{capture_num}  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ")

            try:
                capture_once(args, server_ip, out_dir)
            except Exception as e:
                print(f"[ERROR] Falló la captura #{capture_num}: {e}", file=sys.stderr)

            if stop["flag"]:
                break

            next_ts = datetime.now().strftime("%H:%M:%S")
            wait_until = datetime.now().timestamp() + interval_sec
            print(f"[LOOP] Próxima captura en {args.interval} min "
                  f"(~{datetime.fromtimestamp(wait_until).strftime('%H:%M:%S')})\n")

            # Espera en trozos pequeños para responder rápido al Ctrl+C
            while not stop["flag"] and time.time() < wait_until:
                time.sleep(1)

        print(f"[INFO] Loop detenido. Total capturas realizadas: {capture_num}")

    #  Modo single (comportamiento original) 
    else:
        print("[INFO] Ejecuta como root/Admin si ves errores de permisos.")
        capture_once(args, server_ip, out_dir)


if __name__ == "__main__":
    main()