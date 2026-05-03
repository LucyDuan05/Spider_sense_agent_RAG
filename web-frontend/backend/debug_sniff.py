# -*- coding: utf-8 -*-
from scapy.all import sniff, IP, TCP, UDP
import sys

def packet_callback(packet):
    """处理每个捕获到的数据包"""
    if packet.haslayer(IP):
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        proto = "TCP" if packet.haslayer(TCP) else "UDP" if packet.haslayer(UDP) else "Other"
        
        print(f"[Captured] {proto} | {src_ip} -> {dst_ip} | Length: {len(packet)}")

def main():
    # 根据你之前的 ip addr 结果，网卡名为 eth0
    interface = "eth0"
    
    print(f"--- 启动实时抓包测试 (网卡: {interface}) ---")
    print("提示：如果没有任何输出，请在另一个窗口执行 'ping 8.8.8.8' 或访问网页")
    
    try:
        # sniff 是 Scapy 的核心抓包函数
        # count=10 表示抓够 10 个包就自动停止
        # iface 指定网卡
        # prn 是回调函数，每抓到一个包就执行一次
        sniff(iface=interface, prn=packet_callback, count=10, timeout=30)
        
        print("\n--- 测试完成，成功捕获到 10 个数据包 ---")
    except PermissionError:
        print("\n[错误] 权限不足！请使用 sudo 运行此脚本。")
    except Exception as e:
        print(f"\n[错误] 发生异常: {e}")

if __name__ == "__main__":
    main()