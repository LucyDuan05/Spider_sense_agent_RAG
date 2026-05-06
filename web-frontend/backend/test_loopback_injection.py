#!/usr/bin/env python3
"""
Test script to verify if scapy send() to loopback is captured by sniff()
"""
import threading
import time
from scapy.all import IP, TCP, send, sniff

captured_packets = []

def packet_callback(pkt):
    if pkt.haslayer(IP):
        captured_packets.append(pkt)
        print(f"[CAPTURED] {pkt[IP].src} -> {pkt[IP].dst} | {pkt.summary()}")

def start_sniffer():
    print("[SNIFFER] Starting sniff on lo...")
    sniff(iface='lo', prn=packet_callback, store=False, timeout=5)
    print(f"[SNIFFER] Stopped. Captured {len(captured_packets)} packets")

def inject_packets():
    time.sleep(1)  # Wait for sniffer to start
    print("[INJECT] Sending 5 test packets to 127.0.0.1...")
    for i in range(5):
        pkt = IP(src='127.0.0.1', dst='127.0.0.1') / TCP(dport=80, sport=40000+i, flags='S')
        print(f"[INJECT] Sending packet {i+1}...")
        send(pkt, verbose=False)
        time.sleep(0.2)
    print("[INJECT] Done sending packets")

if __name__ == '__main__':
    print("=" * 60)
    print("Testing loopback injection and capture")
    print("=" * 60)

    sniffer_thread = threading.Thread(target=start_sniffer)
    injector_thread = threading.Thread(target=inject_packets)

    sniffer_thread.start()
    injector_thread.start()

    sniffer_thread.join()
    injector_thread.join()

    print("\n" + "=" * 60)
    print(f"RESULT: {len(captured_packets)} packets captured out of 5 sent")
    if len(captured_packets) == 0:
        print("❌ PROBLEM: send() to loopback is NOT captured by sniff()")
        print("   This explains why attack injection doesn't work.")
    else:
        print("✅ SUCCESS: Packets are being captured")
    print("=" * 60)
