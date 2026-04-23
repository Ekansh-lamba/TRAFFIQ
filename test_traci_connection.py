import traci
import sys

sumo_binary = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe"
sumo_config = r"sumo_files\intersection.sumocfg"

print("Attempting to start SUMO...")
print(f"Binary: {sumo_binary}")
print(f"Config: {sumo_config}")

try:
    traci.start([sumo_binary, "-c", sumo_config, "--start", "--quit-on-end"])
    print("✓ TraCI connected successfully!")
    
    tls_ids = traci.trafficlight.getIDList()
    print(f"✓ Traffic lights found: {tls_ids}")
    
    traci.close()
    print("✓ Test complete!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()