import os
import subprocess
import time
import signal
import atexit
import psutil
import socket
import sys
import platform

class CarlaServerManager:
    """
    Utility class to manage CARLA server instances.
    Handles launching, checking, and terminating CARLA server processes.
    """
    
    def __init__(self):
        self.carla_process = None
        self.carla_path = self._find_carla_path()
    
    def _find_carla_path(self):
        """Try to find the CARLA installation path based on common locations"""
        # Default paths to check based on OS
        if platform.system() == "Windows":
            paths = [
                os.environ.get("CARLA_ROOT"),
                "C:\\CARLA\\WindowsNoEditor",
                "C:\\carla-0.9.15"
            ]
            executable = "CarlaUE4.exe"
        else:  # Linux/Mac
            paths = [
                os.environ.get("CARLA_ROOT"),
                os.path.expanduser("~/CARLA_0.9.15"),
                os.path.expanduser("~/carla"),
                "/opt/carla-simulator"
            ]
            executable = "CarlaUE4.sh"
        
        # Check if CARLA_ROOT environment variable is set
        if "CARLA_ROOT" in os.environ and os.environ["CARLA_ROOT"]:
            if os.path.exists(os.path.join(os.environ["CARLA_ROOT"], executable)):
                return os.environ["CARLA_ROOT"]
        
        # Check each path
        for path in paths:
            if path and os.path.exists(os.path.join(path, executable)):
                return path
        
        # If not found, return None and let the user specify
        return None
    
    def is_port_in_use(self, port):
        """Check if a port is already in use"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    
    def is_server_running(self, port=2000):
        """Check if a CARLA server is already running on the specified port"""
        return self.is_port_in_use(port)
    
    def find_carla_processes(self):
        """Find running CARLA server processes"""
        carla_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                # Check if the process name contains 'carla' or 'CarlaUE4'
                if proc.info['name'] and ('carla' in proc.info['name'].lower() or 'carlaue4' in proc.info['name'].lower()):
                    carla_processes.append(proc)
                # Also check command line for carla
                elif proc.info['cmdline'] and any('carla' in cmd.lower() for cmd in proc.info['cmdline'] if isinstance(cmd, str)):
                    carla_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return carla_processes
    
    def kill_running_servers(self):
        """Kill any running CARLA server processes"""
        carla_processes = self.find_carla_processes()
        for proc in carla_processes:
            try:
                print(f"Terminating CARLA process with PID {proc.pid}")
                proc.terminate()
                proc.wait(timeout=5)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                try:
                    proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
    
    def start_server(self, port=2000, town="Town01", quality="Epic", offscreen=False, carla_path=None):
        """
        Start a CARLA server with the specified parameters
        
        Args:
            port (int): Port to run the server on
            town (str): Town/map to load
            quality (str): Quality level (Low, Epic)
            offscreen (bool): Whether to run in headless mode
            carla_path (str): Path to CARLA installation (overrides auto-detection)
            
        Returns:
            bool: True if server started successfully, False otherwise
        """
        # Use provided path or try to find it
        if carla_path:
            self.carla_path = carla_path
        
        if not self.carla_path:
            print("Error: Could not find CARLA installation path.")
            print("Please set the CARLA_ROOT environment variable or provide the path explicitly.")
            return False
        
        # Check if a server is already running on this port
        if self.is_server_running(port):
            print(f"A server is already running on port {port}.")
            return True
        
        # Determine the executable based on platform
        if platform.system() == "Windows":
            executable = os.path.join(self.carla_path, "CarlaUE4.exe")
        else:
            executable = os.path.join(self.carla_path, "CarlaUE4.sh")
        
        if not os.path.exists(executable):
            print(f"Error: CARLA executable not found at {executable}")
            return False
        
        # Build command with parameters
        cmd = [executable, f"-carla-port={port}"]
        
        # Add town parameter if specified
        if town:
            cmd.append(town)
        
        # Add quality setting
        if quality.lower() == "low":
            cmd.append("-quality-level=Low")
        
        # Add offscreen mode if requested
        if offscreen:
            cmd.append("-RenderOffScreen")
        
        print(f"Starting CARLA server with command: {' '.join(cmd)}")
        
        try:
            # Start the server process
            if platform.system() == "Windows":
                # On Windows, we need to use CREATE_NEW_CONSOLE to avoid blocking
                self.carla_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:
                # On Linux/Mac, we can just use regular Popen
                self.carla_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            
            # Register cleanup function to ensure server is terminated on exit
            atexit.register(self.stop_server)
            
            # Wait for server to start (check if port becomes available)
            max_wait = 60  # Maximum wait time in seconds
            wait_interval = 1  # Check interval in seconds
            
            print(f"Waiting for CARLA server to start on port {port}...")
            for _ in range(max_wait):
                if self.is_server_running(port):
                    print(f"CARLA server started successfully on port {port}!")
                    return True
                time.sleep(wait_interval)
            
            print(f"Timed out waiting for CARLA server to start on port {port}.")
            self.stop_server()
            return False
            
        except Exception as e:
            print(f"Error starting CARLA server: {e}")
            return False
    
    def stop_server(self):
        """Stop the CARLA server if it's running"""
        if self.carla_process:
            try:
                print("Stopping CARLA server...")
                if platform.system() == "Windows":
                    # On Windows, we need to use taskkill to ensure all child processes are terminated
                    subprocess.call(['taskkill', '/F', '/T', '/PID', str(self.carla_process.pid)])
                else:
                    # On Linux/Mac, we can use process group termination
                    os.killpg(os.getpgid(self.carla_process.pid), signal.SIGTERM)
                    self.carla_process.wait(timeout=10)
            except (subprocess.SubprocessError, psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                print("Could not gracefully terminate CARLA. Trying to force kill...")
                try:
                    self.carla_process.kill()
                except:
                    pass
            finally:
                self.carla_process = None
                # Unregister the atexit handler to avoid calling it twice
                try:
                    atexit.unregister(self.stop_server)
                except:
                    pass 