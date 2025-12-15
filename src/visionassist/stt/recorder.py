import threading
import sounddevice as sd
import soundfile as sf
import queue
import inquirer
import io
import numpy as np

from visionassist.logger import logger

def pick_audio_input():
    """
    Interactive CLI-based audio input device selection using arrow keys.
    Lists all available input devices and allows navigation with up/down arrows.
    Only shows devices that can actually be opened for recording.
    """
    devices = sd.query_devices()

    # Build list of input-capable devices and validate they can be opened
    input_devices = []
    logger.info("Scanning available audio devices...")
    
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            try:
                test_stream = sd.InputStream(
                    samplerate=int(d['default_samplerate']),
                    channels=1,
                    device=i,
                    blocksize=1024
                )
                test_stream.close()
                input_devices.append((i, d))
            except Exception:
                pass

    if not input_devices:
        logger.info("No working input devices found!")
        return None

    choices = []
    default_device_idx = None
    
    for idx, (dev_index, dev) in enumerate(input_devices):
        is_default = dev_index == sd.default.device[0]
        default_marker = " [DEFAULT]" if is_default else ""
        label = f"{dev['name']}{default_marker} (Channels: {dev['max_input_channels']}, SR: {int(dev['default_samplerate'])} Hz)"
        choices.append((label, dev_index))
        
        if is_default:
            default_device_idx = idx

    logger.info("Audio Input Device Selection")
    logger.info("Use ↑/↓ arrow keys to navigate, Enter to select\n")

    # Create inquirer list selection
    questions = [
        inquirer.List(
            'device',
            message="Select an audio input device",
            choices=choices,
            default=choices[default_device_idx][1] if default_device_idx is not None else None,
        ),
    ]

    try:
        answers = inquirer.prompt(questions)
        if answers is None:
            logger.info("Selection cancelled.")
            return None
        
        selected_index = answers['device']
        selected_device = sd.query_devices(selected_index)
        logger.info(f"✓ Selected: {selected_device['name']}")
        return selected_index
        
    except KeyboardInterrupt:
        logger.info("Selection cancelled.")
        return None


class AudioRecorder:
    
    def __init__(self, output_file="recorded_audio.wav", samplerate=44100):
        self.output_file = output_file
        self.samplerate = samplerate
        
        # Select device once during initialization
        self.device = pick_audio_input()
        if self.device is None:
            raise ValueError("No audio device selected. Cannot initialize recorder.")
        
        self.device_info = sd.query_devices(self.device)
        self.channels = min(self.device_info["max_input_channels"], 1)  # force mono
        
        # Recording state
        self.recording = False
        self.audio_queue = queue.Queue()
        self.audio_data = []  # Store audio chunks for in-memory processing
        self.writer = None
        self.stream = None
        self.write_thread = None
        
        logger.info(f"AudioRecorder initialized with: {self.device_info['name']}")
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback function for audio stream."""
        if self.recording:
            self.audio_queue.put(indata.copy())
            self.audio_data.append(indata.copy())  # Also store for in-memory access
    
    def _write_audio(self):
        """Background thread that writes audio data to file."""
        while self.recording:
            try:
                data = self.audio_queue.get(timeout=0.1)
                self.writer.write(data)
            except queue.Empty:
                pass
    
    def start_recording(self):
        """Start audio recording."""
        if self.recording:
            logger.info("Already recording!")
            return
        
        # Clear previous audio data
        self.audio_data = []
        
        self.recording = True
        self.writer = sf.SoundFile(
            self.output_file,
            mode='w',
            samplerate=self.samplerate,
            channels=self.channels
        )
        
        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            device=self.device,
            callback=self._audio_callback
        )
        self.stream.start()
        
        logger.info(f"Recording started ... ")
        
        self.write_thread = threading.Thread(target=self._write_audio, daemon=True)
        self.write_thread.start()
    
    def stop_recording(self, return_bytes=False):
        """
        Stop audio recording and optionally return audio bytes.
        
        Args:
            return_bytes: If True, returns audio as bytes. If False, just saves to file.
            
        Returns:
            bytes: Audio data as bytes if return_bytes=True, None otherwise
        """
        if not self.recording:
            logger.info("Not recording currently!")
            return None
        
        self.recording = False
        logger.info("Stopping recording...")
        
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        # Flush any remaining audio
        while not self.audio_queue.empty():
            self.writer.write(self.audio_queue.get())
        
        self.writer.close()
        self.writer = None
        
        logger.info(f"Recording saved to: {self.output_file}")
        
        if return_bytes:
            # Read the file and return as bytes
            with open(self.output_file, 'rb') as f:
                audio_bytes = f.read()
            return audio_bytes
        
        return None
    
    def get_audio_bytes_in_memory(self):
        """
        Stop recording and return audio bytes without saving to disk.
        Uses in-memory buffer for faster processing.
        
        Returns:
            bytes: Audio data as bytes in WAV format
        """
        if not self.recording:
            logger.info("Not recording currently!")
            return None
        
        self.recording = False
        logger.info("Stopping recording...")
        
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        # Wait a bit for any remaining callbacks
        import time as time_module
        time_module.sleep(0.1)
        
        # Create in-memory buffer
        audio_buffer = io.BytesIO()
        
        # Write all collected audio data to in-memory buffer
        if self.audio_data:
            all_audio = np.concatenate(self.audio_data, axis=0)
            
            with sf.SoundFile(
                audio_buffer,
                mode='w',
                samplerate=self.samplerate,
                channels=self.channels,
                format='WAV'
            ) as writer_mem:
                writer_mem.write(all_audio)
        
        # Close the file writer if it was open
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        
        # Clear the queue
        while not self.audio_queue.empty():
            self.audio_queue.get()
        
        # Get bytes from buffer
        audio_bytes = audio_buffer.getvalue()
        audio_buffer.close()
        
        logger.info(f"Recording completed. Audio size: {len(audio_bytes)} bytes")
        
        return audio_bytes
    
    def is_recording(self):
        """Check if currently recording."""
        return self.recording
    
    def cleanup(self):
        """Cleanup resources."""
        if self.recording:
            self.stop_recording()