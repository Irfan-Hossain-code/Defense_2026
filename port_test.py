import serial.tools.list_ports

print("Available ports right now:")

ports = list(serial.tools.list_ports.comports())

for port in ports:
    print(f"{port.device}: {port.description} | hwid={port.hwid}")

if not ports:
    print("No ports found.")

if __name__ == "__main__":
    print("Starting serial port test...")