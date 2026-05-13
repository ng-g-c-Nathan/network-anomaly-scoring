import sys
import csv
import socket
import argparse
from pathlib import Path
from datetime import datetime, timezone

try:
    from scapy.all import rdpcap, sniff, IP, TCP, UDP, DNS, Raw
    from scapy.layers.http import HTTP, HTTPRequest, HTTPResponse
    from scapy.layers.tls.all import TLS, TLSClientHello
except ImportError:
    print("[ERROR] Scapy no está instalado. Ejecuta: pip install scapy", file=sys.stderr)
    sys.exit(1)


# -------------------------------------------------------------------
# Columnas del CSV — idénticas a pruebas.py (Linux)
# para compatibilidad con preprocess.py
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
    """
    Convierte los flags TCP de Scapy al formato hex que usa Wireshark.
    Scapy los expone como entero en pkt[TCP].flags.
    """
    if TCP not in pkt:
        return None
    flags_int = int(pkt[TCP].flags)
    return hex(flags_int)


def get_frame_protocols(pkt) -> str:
    """
    Construye una cadena de protocolos presente en el paquete,
    equivalente al campo frame.protocols de Wireshark.
    Ejemplo: "eth:ip:tcp:http"
    """
    layers = []
    layer = pkt
    while layer:
        name = layer.__class__.__name__.lower()
        # Nombres más limpios para los layers más comunes
        name_map = {
            "ether": "eth",
            "ip": "ip",
            "tcp": "tcp",
            "udp": "udp",
            "dns": "dns",
            "httprequest": "http",
            "httpresponse": "http",
            "tls": "tls",
            "raw": "data",
        }
        layers.append(name_map.get(name, name))
        layer = layer.payload if layer.payload.__class__.__name__ != "NoPayload" else None
    return ":".join(layers)


def extract_sni(pkt) -> str | None:
    """
    Extrae el Server Name Indication (SNI) del TLS ClientHello.
    Scapy expone las extensiones TLS en pkt[TLSClientHello].ext
    """
    try:
        if TLSClientHello not in pkt:
            return None
        for ext in (pkt[TLSClientHello].ext or []):
            # Tipo 0 = server_name
            if hasattr(ext, "type") and ext.type == 0:
                if hasattr(ext, "servernames"):
                    for sn in ext.servernames:
                        if hasattr(sn, "servername"):
                            return sn.servername.decode("utf-8", errors="replace")
    except Exception:
        pass
    return None


def extract_dns_query(pkt) -> str | None:
    """Extrae el primer nombre de consulta DNS del paquete."""
    try:
        if DNS not in pkt:
            return None
        dns = pkt[DNS]
        if dns.qd and hasattr(dns.qd, "qname"):
            return dns.qd.qname.decode("utf-8", errors="replace").rstrip(".")
    except Exception:
        pass
    return None


def process_packet(pkt, frame_number: int, server_ip: str) -> tuple[dict, int]:
    """
    Extrae todos los campos de un paquete Scapy y retorna
    (row_dict, pkt_len) compatibles con el schema de CSV_COLS.

    Returns
    -------
    (row, pkt_len) donde row puede estar vacío si el paquete no tiene IP.
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

    # IP — si no tiene capa IP, no es útil para el modelo
    if IP not in pkt:
        return {}, pkt_len

    row["ip_src"] = pkt[IP].src
    row["ip_dst"] = pkt[IP].dst

    # TCP
    if TCP in pkt:
        row["tcp_srcport"] = str(pkt[TCP].sport)
        row["tcp_dstport"] = str(pkt[TCP].dport)
        row["tcp_flags"]   = format_tcp_flags(pkt)

    # UDP
    if UDP in pkt:
        row["udp_srcport"] = str(pkt[UDP].sport)
        row["udp_dstport"] = str(pkt[UDP].dport)

    # DNS
    dns_name = extract_dns_query(pkt)
    if dns_name:
        row["dns_qry_name"] = dns_name

    # TLS SNI
    sni = extract_sni(pkt)
    if sni:
        row["tls_handshake_extensions_server_name"] = sni

    # HTTP (Scapy necesita scapy.layers.http importado)
    try:
        if HTTPRequest in pkt:
            req = pkt[HTTPRequest]
            method = getattr(req, "Method", None)
            path   = getattr(req, "Path",   None)
            if method:
                row["http_request_method"] = method.decode("utf-8", errors="replace")
            if path:
                row["http_request_uri"] = path.decode("utf-8", errors="replace")
        if HTTPResponse in pkt:
            resp = pkt[HTTPResponse]
            code = getattr(resp, "Status_Code", None)
            if code:
                row["http_response_code"] = code.decode("utf-8", errors="replace")
    except Exception:
        pass

    return row, pkt_len


# -------------------------------------------------------------------
# Modos: PCAP o captura en vivo
# -------------------------------------------------------------------

def process_packets(packets, server_ip: str) -> tuple[list[dict], int, int]:
    """
    Itera sobre una lista/generador de paquetes Scapy y extrae las filas.

    Returns
    -------
    (rows, input_bytes, output_bytes)
    """
    rows         = []
    input_bytes  = 0
    output_bytes = 0

    for i, pkt in enumerate(packets, 1):
        row, pkt_len = process_packet(pkt, i, server_ip)

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


def write_csv(rows: list[dict], out_path: Path, input_mb: float, output_mb: float) -> Path:
    """Escribe el CSV con el schema fijo de CSV_COLS."""
    csv_name = f"{out_path.stem}_({input_mb}_input)_({output_mb}_output).csv"
    csv_path = out_path.parent / csv_name

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    return csv_path


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Captura/convierte tráfico de red a CSV compatible con preprocess.py"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pcap",     metavar="FILE",    help="Leer un archivo PCAP existente")
    group.add_argument("--live",     action="store_true", help="Captura en vivo (requiere Admin)")
    parser.add_argument("--duration", type=int, default=10,
                        help="Duración de captura en vivo en segundos (default: 10)")
    parser.add_argument("--iface",   metavar="IFACE",   default=None,
                        help="Interfaz de red para captura en vivo (default: auto)")
    parser.add_argument("--out",     metavar="DIR",     default=".",
                        help="Directorio de salida (default: directorio actual)")

    args = parser.parse_args()

    server_ip = get_local_ip()
    print(f"[INFO] IP local detectada: {server_ip}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pcap:
        # --- Modo PCAP ---
        pcap_path = Path(args.pcap)
        if not pcap_path.exists():
            print(f"[ERROR] No existe el archivo: {pcap_path}", file=sys.stderr)
            sys.exit(1)

        print(f"[INFO] Leyendo PCAP: {pcap_path}")
        packets = rdpcap(str(pcap_path))
        rows, input_bytes, output_bytes = process_packets(packets, server_ip)

        base     = pcap_path.stem
        out_base = out_dir / base

    else:
        # --- Modo captura en vivo ---
        print(f"[INFO] Capturando {args.duration}s en {'todas las interfaces' if not args.iface else args.iface}...")
        print("[INFO] Ejecuta como Administrador si ves errores de permisos.")

        packets = sniff(
            iface=args.iface,
            timeout=args.duration,
            store=True,
        )
        rows, input_bytes, output_bytes = process_packets(packets, server_ip)

        ts       = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base     = f"traffic_{ts}"
        out_base = out_dir / base

    input_mb  = round(input_bytes  / (1024 * 1024), 2)
    output_mb = round(output_bytes / (1024 * 1024), 2)

    csv_path = write_csv(rows, out_base, input_mb, output_mb)

    print(f"[OK] CSV generado : {csv_path}")
    print(f"[OK] Entrada      : {input_mb} MB")
    print(f"[OK] Salida       : {output_mb} MB")
    print(f"[OK] Filas        : {len(rows)}")


if __name__ == "__main__":
    main()
