import serial
import serial.tools.list_ports
import time
import pandas as pd
import argparse

def find_esp32_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "CH340" in p.description or "CP210" in p.description or "UART" in p.description:
            return p.device
    if len(ports) > 0:
        return ports[0].device
    return None

def main():
    parser = argparse.ArgumentParser(description="IV Infiltration Serial Driver")
    parser.add_argument("--port", help="Serial port of the ESP32", default=None)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--trace", default="frontend/public/replay_trace.csv", help="Path to canonical trace CSV")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between sending rows")
    args = parser.parse_args()

    port = args.port or find_esp32_port()
    if not port:
        print("Error: Could not find any serial ports.")
        return

    print(f"Connecting to ESP32 on {port} at {args.baud} baud...")
    
    try:
        ser = serial.Serial(port, args.baud, timeout=1)
    except Exception as e:
        print(f"Failed to open port {port}: {e}")
        return
        
    time.sleep(2) # Wait for ESP32 to reset upon serial connection
    
    try:
        df = pd.read_csv(args.trace)
    except Exception as e:
        print(f"Failed to read trace file: {e}")
        return

    print(f"Loaded trace with {len(df)} rows. Starting streaming...")
    
    for _, row in df.iterrows():
        t = "NaN" if pd.isna(row['temperature_c']) else f"{row['temperature_c']:.3f}"
        p = "NaN" if pd.isna(row['pressure_adc']) else f"{row['pressure_adc']:.1f}"
        c = row['condition']
        
        # Send data
        payload = f"{t},{p},{c}\n"
        ser.write(payload.encode('utf-8'))
        
        # Wait and read response
        time.sleep(args.interval)
        
        while ser.in_waiting > 0:
            resp = ser.readline().decode('utf-8', errors='replace').strip()
            if resp:
                print(f"ESP32 -> {resp}")
                
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
