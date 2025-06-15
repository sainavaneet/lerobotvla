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
        self.root.title("LeRobot Data Collection Dashboard")
        self.root.geometry("800x600") # Slightly larger window
        self.root.resizable(False, False) # Keep fixed for consistent design

        self.current_process = None
        self.terminal_pid = None
        self.timer_running = False
        self.start_time = None

        # --- Styling Setup (More sophisticated) ---
        self.setup_styles()

        # --- Main Container Frame ---
        # This frame holds everything and sets a uniform background
        main_container = ttk.Frame(root, style="Background.TFrame")
        main_container.pack(fill=tk.BOTH, expand=True)

        # --- Header Section ---
        header_frame = ttk.Frame(main_container, style="Header.TFrame")
        header_frame.pack(fill=tk.X, pady=(0, 20))
        ttk.Label(header_frame, text="🤖 LeRobot Data Collection", style="DashboardHeader.TLabel").pack(pady=10)
        ttk.Separator(header_frame, orient='horizontal').pack(fill='x', padx=20)


        # --- Main Content Area (Divided into two columns) ---
        content_frame = ttk.Frame(main_container, style="Background.TFrame")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)
        content_frame.grid_columnconfigure(0, weight=1) # Left column for controls
        content_frame.grid_columnconfigure(1, weight=1) # Right column for status/timer


        # --- Left Column: Controls ---
        controls_frame = ttk.Frame(content_frame, style="Controls.TFrame", padding=20)
        controls_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W), padx=(0, 15), pady=10)
        controls_frame.grid_rowconfigure(0, weight=1) # Allow row for task frame to expand
        controls_frame.grid_columnconfigure(0, weight=1)

        # Task Selection
        task_labelframe = ttk.LabelFrame(controls_frame, text="Select Task", style="Modern.TLabelframe", padding=(20,10))
        task_labelframe.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        task_labelframe.grid_columnconfigure(0, weight=1) # Allow combobox to expand

        self.task_var = tk.StringVar()
        self.task_combo = ttk.Combobox(
            task_labelframe,
            textvariable=self.task_var,
            values=["pick mint candle", "pick red candle", "pick yellow candle", "pick purple candle"],
            state="readonly",
            font=("Arial", 11)
        )
        self.task_combo.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=10, pady=10)
        self.task_combo.set("pick mint candle")
        self.create_tooltip(self.task_combo, "Choose the specific task for data recording.")

        # Resume Checkbox
        self.resume_var = tk.BooleanVar(value=False)
        self.resume_check = ttk.Checkbutton(
            controls_frame,
            text="Resume previous session",
            variable=self.resume_var,
            style="Modern.TCheckbutton",
            command=self.toggle_resume_info
        )
        self.resume_check.grid(row=1, column=0, sticky=tk.W, pady=(0, 5), padx=10)
        self.create_tooltip(self.resume_check, "Check to append data to the last recording for this task.")

        self.resume_info_label = ttk.Label(controls_frame, text="", foreground="#FF5722", font=("Arial", 9, "italic"), wraplength=300)
        self.resume_info_label.grid(row=2, column=0, sticky=tk.W, pady=(0, 20), padx=10)

        # Action Buttons
        action_button_frame = ttk.Frame(controls_frame, style="Background.TFrame")
        action_button_frame.grid(row=3, column=0, pady=(10, 0), sticky=(tk.W, tk.E))
        action_button_frame.grid_columnconfigure(0, weight=1) # Center buttons
        action_button_frame.grid_columnconfigure(1, weight=1) # Center buttons


        self.run_button = ttk.Button(
            action_button_frame,
            text="🚀 Start Recording",
            command=self.run_dataset_script,
            style="Run.TButton"
        )
        self.run_button.grid(row=0, column=0, padx=5, pady=10, sticky=(tk.W, tk.E))
        self.create_tooltip(self.run_button, "Starts the LeRobot recording script in a new terminal.")

        self.stop_button = ttk.Button(
            action_button_frame,
            text="🛑 Stop Recording",
            command=self.stop_recording,
            style="Stop.TButton"
        )
        self.stop_button.grid(row=0, column=1, padx=5, pady=10, sticky=(tk.W, tk.E))
        self.stop_button.state(['disabled'])
        self.create_tooltip(self.stop_button, "Sends a signal to stop the recording and encode the data.")

        self.timer_button = ttk.Button(
            controls_frame,
            text="⏱️ Start 8s Timer",
            command=self.start_timer_only,
            style="Timer.TButton"
        )
        self.timer_button.grid(row=4, column=0, pady=(20, 10), padx=10, sticky=(tk.W, tk.E))
        self.create_tooltip(self.timer_button, "Starts a standalone 8-second visual timer.")

        # --- Right Column: Status & Output ---
        status_output_frame = ttk.Frame(content_frame, style="Controls.TFrame", padding=20)
        status_output_frame.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.E, tk.W), padx=(15, 0), pady=10)
        status_output_frame.grid_columnconfigure(0, weight=1)
        status_output_frame.grid_rowconfigure(2, weight=1) # Allow log display to expand

        # Current Status
        ttk.Label(status_output_frame, text="Current Status:", style="SectionHeader.TLabel").grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        self.status_label = ttk.Label(status_output_frame, text="Ready to record.", style="MainStatus.TLabel")
        self.status_label.grid(row=1, column=0, sticky=tk.W, pady=(0, 20))

        # Timer Display
        ttk.Label(status_output_frame, text="Activity Timer:", style="SectionHeader.TLabel").grid(row=2, column=0, sticky=tk.W, pady=(20, 10))
        self.timer_label = ttk.Label(status_output_frame, text="0.0s", style="BigTimer.TLabel")
        self.timer_label.grid(row=3, column=0, sticky=tk.W, pady=(0, 20))

        # Placeholder for a simple log/output area
        ttk.Label(status_output_frame, text="Recent Activity Log:", style="SectionHeader.TLabel").grid(row=4, column=0, sticky=tk.W, pady=(20, 10))
        self.log_text = tk.Text(status_output_frame, height=8, state='disabled', wrap='word',
                                font=("Consolas", 9), background="#E0E0E0", foreground="#333333",
                                relief="flat", borderwidth=0)
        self.log_text.grid(row=5, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        self.log_text.tag_configure("info", foreground="#34495E")
        self.log_text.tag_configure("success", foreground="#28A745")
        self.log_text.tag_configure("warning", foreground="#FFC107")
        self.log_text.tag_configure("error", foreground="#DC3545")

        self.append_log("Application started.", "info")


    def setup_styles(self):
        style = ttk.Style()
        # Modern theme (e.g., 'clam', 'alt', 'vista', 'xpnative' - 'clam' is cross-platform)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass # Fallback to default if not available

        # --- Colors Palette ---
        PRIMARY_COLOR = "#3498DB" # Blue
        ACCENT_COLOR = "#2ECC71"  # Green
        DANGER_COLOR = "#E74C3C"  # Red
        WARNING_COLOR = "#F39C12" # Orange
        TEXT_COLOR = "#2C3E50"    # Dark Blue/Grey
        LIGHT_BG = "#ECF0F1"      # Light Grey
        MID_BG = "#F9F9F9"        # Off-white
        DARK_BG = "#BDC3C7"       # Medium Grey

        # --- Global Style Configurations ---
        style.configure("TFrame", background=LIGHT_BG)
        style.configure("Background.TFrame", background=LIGHT_BG)

        # Header Frame
        style.configure("Header.TFrame", background=TEXT_COLOR)
        style.configure("DashboardHeader.TLabel",
                        font=("Arial", 20, "bold"),
                        foreground="white",
                        background=TEXT_COLOR,
                        padding=(10, 15))

        # Main Content Frames
        style.configure("Controls.TFrame", background=MID_BG, relief="flat", borderwidth=1, bordercolor=DARK_BG)
        style.configure("Modern.TLabelframe", background=MID_BG, relief="groove", borderwidth=1, bordercolor=DARK_BG)
        style.configure("Modern.TLabelframe.Label", font=("Arial", 11, "bold"), foreground=TEXT_COLOR, background=MID_BG)

        # Labels
        style.configure("TLabel", background=MID_BG, foreground=TEXT_COLOR, font=("Arial", 10))
        style.configure("SectionHeader.TLabel", font=("Arial", 12, "bold"), foreground=PRIMARY_COLOR, background=MID_BG)
        style.configure("MainStatus.TLabel", font=("Arial", 11, "bold"), foreground=TEXT_COLOR, background=MID_BG)
        style.configure("BigTimer.TLabel", font=("Arial", 28, "bold"), foreground=PRIMARY_COLOR, background=MID_BG)


        # Buttons
        style.configure("TButton",
            font=("Arial", 11, "bold"),
            padding=8,
            borderwidth=0,
            relief="flat",
            foreground="white",
            background=PRIMARY_COLOR # Default button color
        )
        style.map("TButton",
            background=[('active', '#5DADE2'), ('!disabled', PRIMARY_COLOR)], # Lighter blue on active
            foreground=[('disabled', '#BBBBBB')] # Lighter grey for disabled text
        )

        style.configure("Run.TButton", background=ACCENT_COLOR) # Green
        style.map("Run.TButton", background=[('active', '#58D68D')])

        style.configure("Stop.TButton", background=DANGER_COLOR) # Red
        style.map("Stop.TButton", background=[('active', '#EB746C')])

        style.configure("Timer.TButton", background=WARNING_COLOR) # Orange
        style.map("Timer.TButton", background=[('active', '#F5B041')])

        # Checkbutton
        style.configure("Modern.TCheckbutton", font=("Arial", 10), background=MID_BG, foreground=TEXT_COLOR)

        # Combobox
        style.configure("TCombobox",
                        fieldbackground="white",
                        background="white",
                        foreground=TEXT_COLOR,
                        selectbackground=PRIMARY_COLOR,
                        selectforeground="white",
                        arrowcolor=TEXT_COLOR) # Set arrow color
        style.map("TCombobox", fieldbackground=[('readonly', 'white')])


    def create_tooltip(self, widget, text):
        toolTip = ToolTip(widget, text)
        def enter(event):
            toolTip.show()
        def leave(event):
            toolTip.hide()
        widget.bind('<Enter>', enter)
        widget.bind('<Leave>', leave)

    def toggle_resume_info(self):
        if self.resume_var.get():
            self.resume_info_label.config(text="Warning: Resuming appends new data to the last recording for this task. Ensure you intend to continue an existing session.")
        else:
            self.resume_info_label.config(text="")

    def append_log(self, message, tag="info"):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {message}\n", tag)
        self.log_text.see(tk.END) # Scroll to the end
        self.log_text.config(state='disabled')

    def start_timer_only(self):
        if not self.timer_running:
            self.timer_running = True
            self.start_time = time.time()
            self.update_timer()
            self.timer_button.state(['disabled'])
            self.status_label.config(text="Timer active for 8 seconds.", foreground="#3498DB")
            self.append_log("8-second timer started.", "info")

    def update_timer(self):
        if self.timer_running:
            elapsed_time = time.time() - self.start_time
            if elapsed_time <= 8.0:
                self.timer_label.config(text=f"{elapsed_time:.1f}s")
                self.root.after(100, self.update_timer)
            else:
                self.timer_label.config(text="8.0s (Done)")
                self.timer_running = False
                self.timer_button.state(['!disabled'])
                self.status_label.config(text="Timer finished. Ready to record.", foreground="#2C3E50")
                self.append_log("8-second timer finished.", "info")

    def run_dataset_script(self):
        selected_task = self.task_var.get()
        resume = self.resume_var.get()

        if self.current_process and self.current_process.poll() is None:
            self.status_label.config(text="A script is already running. Please stop it first.", foreground="#F39C12")
            self.append_log("Cannot start: A recording is already active.", "warning")
            return

        self.status_label.config(text="Opening terminal and launching script...", foreground="#F39C12")
        self.append_log(f"Attempting to start recording for '{selected_task}'...", "info")
        self.root.update_idletasks() # Force GUI update

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
            f"--dataset.root=/home/navaneet/lerobotvla/datasets/special_dataset"
        )

        if resume:
            command += " --resume True"
            self.append_log("Resume option enabled.", "info")

        try:
            # Using Popen to launch gnome-terminal with the script
            self.current_process = subprocess.Popen([
                "gnome-terminal",
                "--",
                "bash",
                "-c",
                f"cd /home/navaneet/lerobotvla && {command}; exec bash"
            ])

            # Give a moment for the terminal to appear and the process to start
            time.sleep(2) # Increased sleep for robustness

            # Attempt to find the actual lerobot script PID within the terminal's children
            self.terminal_pid = None
            try:
                parent_process = psutil.Process(self.current_process.pid)
                children = parent_process.children(recursive=True)
                for child in children:
                    # Look for the Python process running 'lerobot.record'
                    if "python" in child.name() and "-m lerobot.record" in " ".join(child.cmdline()):
                        self.terminal_pid = child.pid
                        self.append_log(f"Found LeRobot script PID: {self.terminal_pid}", "info")
                        break
                if not self.terminal_pid:
                    # Fallback to the gnome-terminal PID if the script isn't immediately found
                    self.terminal_pid = self.current_process.pid
                    self.append_log("Could not find specific script PID, using terminal PID as fallback.", "warning")

            except (psutil.NoSuchProcess, psutil.AccessDenied) as proc_error:
                self.append_log(f"Process lookup issue: {proc_error}. Falling back to terminal PID.", "warning")
                self.terminal_pid = self.current_process.pid # Fallback if psutil fails

            self.status_label.config(text="Recording in progress...", foreground="#2ECC71")
            self.append_log("LeRobot script running in new terminal.", "success")
            self.run_button.state(['disabled'])
            self.stop_button.state(['!disabled'])
            self.timer_button.state(['disabled']) # Disable timer during recording

        except FileNotFoundError:
            self.status_label.config(text="Error: 'gnome-terminal' or 'xdotool' not found.", foreground="#E74C3C")
            self.append_log("Error: gnome-terminal or xdotool command not found. Please install them.", "error")
            self.run_button.state(['!disabled'])
            self.stop_button.state(['disabled'])
            self.timer_button.state(['!disabled'])
        except Exception as e:
            self.status_label.config(text=f"Script launch failed: {str(e)}", foreground="#E74C3C")
            self.append_log(f"Error launching script: {e}", "error")
            self.run_button.state(['!disabled'])
            self.stop_button.state(['disabled'])
            self.timer_button.state(['!disabled'])


    def stop_recording(self):
        if not self.terminal_pid and (not self.current_process or self.current_process.poll() is not None):
            self.status_label.config(text="No active recording process to stop.", foreground="#F39C12")
            self.append_log("Stop command ignored: No active recording.", "warning")
            return

        self.status_label.config(text="Sending stop signal...", foreground="#F39C12")
        self.append_log("Attempting to stop recording...", "info")
        self.timer_running = False
        self.timer_label.config(text="0.0s")
        self.timer_button.state(['!disabled']) # Re-enable timer button

        try:
            # Send Escape key to the terminal. This is a common way to stop lerobot.record
            subprocess.run(['xdotool', 'key', 'Escape'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.status_label.config(text="Stop signal sent. Waiting for data encoding...", foreground="#3498DB")
            self.append_log("Escape key sent to terminal. Monitoring process for exit.", "info")

            # Start monitoring to detect when the process actually exits
            self.root.after(1000, self.monitor_process)

        except FileNotFoundError:
            self.status_label.config(text="Error: 'xdotool' command not found. Cannot send stop signal.", foreground="#E74C3C")
            self.append_log("Error: 'xdotool' not found. Please install it to enable stopping.", "error")
            self.run_button.state(['!disabled'])
            self.stop_button.state(['disabled'])
            self.timer_button.state(['!disabled'])
        except subprocess.CalledProcessError as e:
            self.status_label.config(text=f"Failed to send stop signal: {e.stderr.decode().strip()}", foreground="#E74C3C")
            self.append_log(f"Error sending Escape key via xdotool: {e.stderr.decode().strip()}", "error")
            self.run_button.state(['!disabled'])
            self.stop_button.state(['disabled'])
            self.timer_button.state(['!disabled'])
        except Exception as e:
            self.status_label.config(text=f"An error occurred during stop: {str(e)}", foreground="#E74C3C")
            self.append_log(f"Unexpected error during stop operation: {e}", "error")
            self.run_button.state(['!disabled'])
            self.stop_button.state(['disabled'])
            self.timer_button.state(['!disabled'])

    def monitor_process(self):
        try:
            if self.terminal_pid:
                process_is_running = False
                try:
                    main_proc = psutil.Process(self.terminal_pid)
                    if main_proc.is_running():
                        process_is_running = True
                        # Check for descendants, specifically the Python script
                        children = main_proc.children(recursive=True)
                        if any("python -m lerobot.record" in " ".join(c.cmdline()) for c in children if c.is_running()):
                             self.status_label.config(text="Data encoding in progress...", foreground="#3498DB")
                             self.root.after(2000, self.monitor_process) # Re-check after 2 seconds
                             return
                        # If main_proc is running but the specific script child isn't, it might be the terminal itself
                        elif main_proc.name() == "gnome-terminal":
                            # Check if the terminal has relevant child processes (e.g., bash still running the script)
                            if any("bash" in c.name() for c in children if c.is_running()):
                                self.status_label.config(text="Terminal is open, waiting for script to finish...", foreground="#3498DB")
                                self.root.after(2000, self.monitor_process)
                                return

                except psutil.NoSuchProcess:
                    process_is_running = False # Process has terminated

                if not process_is_running:
                    # Process has terminated
                    self.status_label.config(text="Recording stopped. Data encoding complete!", foreground="#2ECC71")
                    self.append_log("Recording and encoding successfully completed.", "success")
                    self.run_button.state(['!disabled'])
                    self.stop_button.state(['disabled'])
                    self.terminal_pid = None
                    self.current_process = None # Clear the main process handle

                else: # Fallback for cases where process_is_running is true but relevant child isn't found
                    self.status_label.config(text="Monitoring complete. Script should be stopped.", foreground="#2ECC71")
                    self.append_log("Process monitoring finished. Script expected to be stopped.", "info")
                    self.run_button.state(['!disabled'])
                    self.stop_button.state(['disabled'])
                    self.terminal_pid = None
                    self.current_process = None


        except (psutil.AccessDenied, Exception) as e:
            self.status_label.config(text=f"Error during monitoring: {str(e)}", foreground="#E74C3C")
            self.append_log(f"Error during process monitoring: {e}", "error")
            self.run_button.state(['!disabled'])
            self.stop_button.state(['disabled'])
            self.terminal_pid = None
            self.current_process = None


# --- Tooltip Class (Remains the same as it's already functional) ---
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.id = None
        self.x = self.y = 0

    def show(self):
        if self.tip_window or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tip_window = tk.Toplevel(self.widget)
        self.tip_window.wm_overrideredirect(True)
        self.tip_window.wm_geometry(f"+{x}+{y}")

        label = ttk.Label(self.tip_window, text=self.text, background="#FFFFCC",
                          relief="solid", borderwidth=1,
                          font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hide(self):
        if self.tip_window:
            self.tip_window.destroy()
        self.tip_window = None


def main():
    root = tk.Tk()
    app = DatasetGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()