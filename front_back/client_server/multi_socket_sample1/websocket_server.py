import socket
import threading

def handle_client(conn, addr):
    with conn:
        print('Connected by', addr)
        while True:
            data = conn.recv(1024)
            if not data:
                break
            conn.sendall(data)

# 创建一个TCP/IP套接字，AF_INET表示使用IPv4地址，SOCK_STREAM表示使用TCP协议
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('0.0.0.0', 1234)) # 主机上的任意网课都可以使用这个socket进行通信，8888端口
    s.listen()  # 开始监听连接请求
    while True:
        conn, addr = s.accept() # 接受来自任意客户端的连接请求
        t=threading.Thread(target=handle_client, args=(conn, addr)) # 创建一个线程来处理客户端请求
        t.start()