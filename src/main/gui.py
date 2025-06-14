import tkinter as tk
from tkinter import ttk
import subprocess
import os
import signal
import time
import psutil

class DatasetGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Dataset Script Runner")
        
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        
        # Store the process ID of the running script
        self.current_process = None
        self.terminal_pid = None
        self.timer_running = False
        self.start_time = None

        main_frame = ttk.Frame(root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create task selection
        task_label = ttk.Label(main_frame, text="Select Task:")
        task_label.grid(row=0, column=0, pady=(0, 10), sticky=tk.W)
        
        self.task_var = tk.StringVar()
        self.task_combo = ttk.Combobox(
            main_frame,
            textvariable=self.task_var,
            values=["pick mint candle", "pick red candle", "pick yellow candle", "pick purple candle"],
            state="readonly"
        )
        self.task_combo.grid(row=1, column=0, pady=(0, 20), sticky=tk.W)
        self.task_combo.set("pick mint candle")  # Default value
        
        # Create resume checkbox
        self.resume_var = tk.BooleanVar(value=False)
        self.resume_check = ttk.Checkbutton(
            main_frame,
            text="Resume previous recording",
            variable=self.resume_var
        )
        self.resume_check.grid(row=2, column=0, pady=(0, 20), sticky=tk.W)
        
        # Create button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, pady=20)
        
        # Create and configure run button
        self.run_button = ttk.Button(
            button_frame,
            text="Run Dataset Script",
            command=self.run_dataset_script,
            style="Accent.TButton"
        )
        self.run_button.grid(row=0, column=0, padx=5)
        
        # Create stop recording button
        self.stop_button = ttk.Button(
            button_frame,
            text="Stop Recording",
            command=self.stop_recording,
            style="Accent.TButton"
        )
        self.stop_button.grid(row=0, column=1, padx=5)
        self.stop_button.state(['disabled'])  # Initially disabled
        
        # Create start timer button
        self.timer_button = ttk.Button(
            button_frame,
            text="Start Timer",
            command=self.start_timer_only,
            style="Accent.TButton"
        )
        self.timer_button.grid(row=0, column=2, padx=5)
        
        # Create status label
        self.status_label = ttk.Label(main_frame, text="")
        self.status_label.grid(row=4, column=0, pady=10)

        # Create timer label
        self.timer_label = ttk.Label(main_frame, text="Time: 0.0s", font=("Arial", 14, "bold"))
        self.timer_label.grid(row=5, column=0, pady=10)

        # Configure style
        style = ttk.Style()
        style.configure("Accent.TButton", font=("Arial", 12))
        
    def start_timer_only(self):
        """Start only the timer without running the dataset script"""
        self.timer_running = True
        self.start_time = time.time()
        self.update_timer()
        self.timer_button.state(['disabled'])  # Disable timer button while running
        
    def update_timer(self):
        if self.timer_running:
            elapsed_time = time.time() - self.start_time
            if elapsed_time <= 8.0:  # Only show timer up to 8 seconds
                self.timer_label.config(text=f"Time: {elapsed_time:.1f}s")
                self.root.after(100, self.update_timer)  # Update every 100ms for smoother display
            else:
                self.timer_label.config(text="Time: 8.0s")
                self.timer_running = False
                self.timer_button.state(['!disabled'])  # Re-enable timer button when done
        
    def run_dataset_script(self):
        selected_task = self.task_var.get()
        resume = self.resume_var.get()
        
        try:
            # Update status
            self.status_label.config(text="Opening terminal and running script...")
            self.root.update()
            
            # Build command with all necessary parameters
            command = (
                f"/home/navaneet/miniconda3/envs/lerobot/bin/python -m lerobot.record "
                f"--dataset.fps=30 "   
                f"--dataset.num_image_writer_processes=8 "
                f"--robot.type=so101_follower "
                f"--robot.port=/dev/tty_follower_arm "
                f'--robot.cameras="{{"left_cam": {{"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}, "right_cam": {{"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}}, "down_cam": {{"type": "opencv", "index_or_path": 8, "width": 640, "height": 480, "fps": 30}}}}" '
                f"--teleop.type=so101_leader "
                f"--teleop.port=/dev/tty_leader_arm "
                f'--dataset.single_task="{selected_task}" '
                f"--dataset.push_to_hub=false "
                f'--dataset.repo_id="lerobot/example_dataset" '
                f"--dataset.root=/home/navaneet/lerobotvla/datasets/test"
            )
            
            if resume:
                command += " --resume True"
            
            # Open new terminal and run the command
            self.current_process = subprocess.Popen([
                "gnome-terminal",
                "--",
                "bash",
                "-c",
                f"cd /home/navaneet/lerobotvla && {command}; exec bash"
            ])
            
            # Get the terminal window PID
            time.sleep(1)  # Give terminal time to open
            self.terminal_pid = self.current_process.pid
            
            self.status_label.config(text="Script is running in new terminal window")
            self.run_button.state(['disabled'])  # Disable run button
            self.stop_button.state(['!disabled'])  # Enable stop button
            
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")
            
    def stop_recording(self):
        try:
            # Stop the timer
            self.timer_running = False
            self.timer_label.config(text="Time: 0.0s")
            self.timer_button.state(['!disabled'])  # Re-enable timer button
            
            # Send ESC key to the terminal
            subprocess.run(['xdotool', 'key', 'Escape'])
            self.status_label.config(text="Stop command sent. Waiting for encoding to complete...")
            
            # Start monitoring the process
            self.root.after(1000, self.monitor_process)
            
        except Exception as e:
            self.status_label.config(text=f"Error stopping recording: {str(e)}")
            
    def monitor_process(self):
        try:
            if self.terminal_pid:
                # Check if the process is still running
                process = psutil.Process(self.terminal_pid)
                if process.is_running():
                    # Check if any child processes are still running
                    children = process.children(recursive=True)
                    if any(child.is_running() for child in children):
                        # Continue monitoring
                        self.root.after(1000, self.monitor_process)
                        return
                    
                    # If no children are running, just update the UI
                    self.status_label.config(text="Recording stopped")
                    self.run_button.state(['!disabled'])  # Enable run button
                    self.stop_button.state(['disabled'])  # Disable stop button
                    self.terminal_pid = None
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # Process has already terminated
            self.status_label.config(text="Recording stopped")
            self.run_button.state(['!disabled'])  # Enable run button
            self.stop_button.state(['disabled'])  # Disable stop button
            self.terminal_pid = None

def main():
    root = tk.Tk()
    app = DatasetGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
