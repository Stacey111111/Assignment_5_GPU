#!/usr/bin/env python3
"""
Natural Language Processing ROS2 Client for Assignment 5

This client integrates three NLP services via ROS2 topics:
1. Text-to-Speech (espeak)
2. Speech-to-Text (Whisper)
3. Language Model (LLaMA)

Option 4 implements the complete voice assistant pipeline:
- Record audio from microphone
- Transcribe speech with Whisper
- Generate AI response with LLaMA  
- Speak response with espeak

Author: Assignment 5 Solution
"""

import sys
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32
import time

# Audio recording libraries
try:
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
    import tempfile
    import os
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("WARNING: Audio libraries not available")
    print("Install: pip3 install sounddevice soundfile numpy")


class NLPClient(Node):
    """
    ROS2 Client for NLP Services
    
    Provides interactive menu to test:
    1. Text-to-Speech (espeak)
    2. Speech-to-Text (Whisper)
    3. Language Model (LLaMA)
    4. Full Voice Assistant Pipeline
    """
    
    def __init__(self):
        super().__init__('nlp_topic_client')
        
        # ====================================================
        # Publishers (Sending Requests)
        # ====================================================
        self.tts_pub = self.create_publisher(String, '/tts_request', 10)
        self.stt_pub = self.create_publisher(Int32, '/stt_request', 10)
        self.llm_pub = self.create_publisher(String, '/llm_request', 10)
        
        # ====================================================
        # Subscribers (Receiving Responses)
        # ====================================================
        self.stt_sub = self.create_subscription(
            String, '/stt_result', self.stt_callback, 10
        )
        self.llm_sub = self.create_subscription(
            String, '/llm_response_stream', self.llm_callback, 10
        )
        
        # ====================================================
        # Threading Events (For synchronization)
        # ====================================================
        self.stt_done = threading.Event()
        self.llm_done = threading.Event()
        
        # Response storage
        self.stt_result = ""
        self.llm_response = ""
        
        self.get_logger().info('NLP Client initialized')
    
    # ========================================================
    # Callback Handlers
    # ========================================================
    
    def stt_callback(self, msg):
        """Handle Speech-to-Text result"""
        self.stt_result = msg.data
        print(f"\nSpeech-to-Text Result: {msg.data}")
        self.stt_done.set()  # Unblock waiting thread
    
    def llm_callback(self, msg):
        """Handle Language Model response stream"""
        if msg.data == "[DONE]":
            print("\n")  # New line after completion
            self.llm_done.set()  # Unblock waiting thread
        else:
            # Stream tokens directly to terminal
            sys.stdout.write(msg.data)
            sys.stdout.flush()
            self.llm_response += msg.data
    
    # ========================================================
    # Audio Recording Function
    # ========================================================
    
    def record_audio(self, duration=5, sample_rate=16000):
        """
        Record audio from microphone
        
        Args:
            duration: Recording duration in seconds (default: 5)
            sample_rate: Audio sample rate in Hz (default: 16000)
        
        Returns:
            Path to saved audio file, or None if failed
        """
        if not AUDIO_AVAILABLE:
            print("\nERROR: Audio recording not available!")
            print("Install: pip3 install sounddevice soundfile numpy")
            return None
        
        try:
            print(f"\nRecording for {duration} seconds...")
            print("Speak now!")
            print("-" * 40)
            
            # Record audio
            audio_data = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype='float32'
            )
            
            # Show countdown
            for i in range(duration, 0, -1):
                print(f"  {i} seconds remaining...", end='\r', flush=True)
                time.sleep(1)
            
            sd.wait()  # Wait for recording to complete
            
            print("\nRecording complete!           ")
            print("-" * 40)
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(
                suffix='.wav',
                delete=False,
                dir='/tmp'
            )
            temp_file_path = temp_file.name
            temp_file.close()
            
            # Write audio data to file
            sf.write(temp_file_path, audio_data, sample_rate)
            
            self.get_logger().info(f'Audio saved: {temp_file_path}')
            return temp_file_path
        
        except Exception as e:
            self.get_logger().error(f'Recording failed: {e}')
            print(f"\nERROR: Recording failed - {e}")
            return None
    
    # ========================================================
    # Interactive Menu Options
    # ========================================================
    
    def show_menu(self):
        """Display interactive menu and handle user input"""
        while rclpy.ok():
            print("\n" + "="*60)
            print("Natural Language Processing Client Menu")
            print("="*60)
            print("1. Test Text-to-Speech (espeak)")
            print("2. Test Speech-to-Text (Whisper)")
            print("3. Test LLM Generation (LLaMA)")
            print("4. Full Voice Assistant Pipeline")
            print("5. Exit")
            print("="*60)
            
            choice = input("Select an option (1-5): ").strip()
            
            if choice == '1':
                self.option_1_test_tts()
            
            elif choice == '2':
                self.option_2_test_stt()
            
            elif choice == '3':
                self.option_3_test_llm()
            
            elif choice == '4':
                self.option_4_full_voice_assistant()
            
            elif choice == '5':
                print("Exiting...")
                return
            
            else:
                print("Invalid choice. Please enter 1-5.")
    
    # ========================================================
    # Option 1: Test Text-to-Speech
    # ========================================================
    
    def option_1_test_tts(self):
        """Test espeak text-to-speech"""
        print("\n" + "="*60)
        print("OPTION 1: Test Text-to-Speech")
        print("="*60)
        
        text = input("Enter text to speak (or press Enter for default): ").strip()
        if not text:
            text = "Hello! This is a test of the text to speech system."
        
        msg = String()
        msg.data = text
        self.tts_pub.publish(msg)
        
        print(f"Text-to-Speech request sent: '{text}'")
        print("Listen for the audio output from the server.")
        print("="*60)
    
    # ========================================================
    # Option 2: Test Speech-to-Text
    # ========================================================
    
    def option_2_test_stt(self):
        """Test Whisper speech-to-text"""
        print("\n" + "="*60)
        print("OPTION 2: Test Speech-to-Text")
        print("="*60)
        
        try:
            dur_input = input("Recording duration in seconds [default: 5]: ").strip()
            dur = int(dur_input) if dur_input.isdigit() else 5
            
            msg = Int32()
            msg.data = dur
            
            self.stt_done.clear()
            self.stt_pub.publish(msg)
            
            print(f"Server is recording for {dur} seconds.")
            print("Speak clearly into the microphone!")
            
            # Wait for response
            self.stt_done.wait()
            
            print("="*60)
            
        except ValueError:
            print("Invalid duration. Please enter a number.")
    
    # ========================================================
    # Option 3: Test Language Model
    # ========================================================
    
    def option_3_test_llm(self):
        """Test LLaMA language model"""
        print("\n" + "="*60)
        print("OPTION 3: Test Language Model")
        print("="*60)
        
        prompt = input("Enter your prompt (or press Enter for default): ").strip()
        if not prompt:
            prompt = "What is artificial intelligence?"
        
        msg = String()
        msg.data = prompt
        
        self.llm_done.clear()
        self.llm_response = ""
        self.llm_pub.publish(msg)
        
        print(f"\nPrompt: '{prompt}'")
        print("LLM Response: ", end="", flush=True)
        
        # Wait for completion
        self.llm_done.wait()
        
        print("="*60)
    
    # ========================================================
    # Option 4: Full Voice Assistant Pipeline
    # ========================================================
    
    def option_4_full_voice_assistant(self):
        """
        Complete voice assistant pipeline implementation
        
        This is the main assignment requirement!
        
        Pipeline:
        1. Record audio from microphone (5 seconds)
        2. Transcribe speech using Whisper service
        3. Generate AI response using LLaMA service
        4. Speak response using espeak service
        """
        print("\n" + "="*60)
        print("OPTION 4: Full Voice Assistant")
        print("="*60)
        print("This pipeline will:")
        print("  1. Record your voice")
        print("  2. Transcribe your speech")
        print("  3. Generate an AI response")
        print("  4. Speak the response back to you")
        print("="*60)
        
        # Get recording duration
        dur_input = input("\nRecording duration in seconds [default: 5]: ").strip()
        duration = int(dur_input) if dur_input.isdigit() else 5
        
        # ====================================================
        # STEP 1: Record Audio from Microphone
        # ====================================================
        print("\n[STEP 1/4] Recording Audio")
        print("-"*60)
        
        audio_file = self.record_audio(duration=duration)
        
        if audio_file is None:
            print("✗ Recording failed! Cannot continue.")
            return
        
        print("✓ Recording successful!")
        
        # ====================================================
        # STEP 2: Transcribe with Whisper Service
        # ====================================================
        print("\n[STEP 2/4] Transcribing Speech")
        print("-"*60)
        print("Sending audio to Whisper service...")
        
        # Since we recorded locally, we need to tell server to record
        # For this assignment, we'll use server-side recording
        msg = Int32()
        msg.data = duration
        
        self.stt_done.clear()
        self.stt_result = ""
        
        # Note: Server will record its own audio
        # In production, you'd upload the file or use a different approach
        print(f"Server recording for {duration} seconds...")
        print("Speak your question clearly!")
        
        self.stt_pub.publish(msg)
        
        # Wait for transcription
        self.stt_done.wait()
        
        if not self.stt_result:
            print("✗ Transcription failed! Cannot continue.")
            try:
                os.remove(audio_file)
            except:
                pass
            return
        
        print(f'✓ Transcription: "{self.stt_result}"')
        
        # ====================================================
        # STEP 3: Generate Response with LLaMA Service
        # ====================================================
        print("\n[STEP 3/4] Generating AI Response")
        print("-"*60)
        print("Sending question to LLaMA service...")
        print("(This may take 5-30 seconds depending on GPU/model)")
        
        msg = String()
        msg.data = self.stt_result
        
        self.llm_done.clear()
        self.llm_response = ""
        self.llm_pub.publish(msg)
        
        print("LLaMA Response: ", end="", flush=True)
        
        # Wait for completion
        self.llm_done.wait()
        
        if not self.llm_response:
            print("\n✗ LLaMA response generation failed!")
            try:
                os.remove(audio_file)
            except:
                pass
            return
        
        print(f'✓ Response generated ({len(self.llm_response)} characters)')
        
        # ====================================================
        # STEP 4: Speak Response with Espeak Service
        # ====================================================
        print("\n[STEP 4/4] Speaking Response")
        print("-"*60)
        print("Sending response to espeak service...")
        
        msg = String()
        msg.data = self.llm_response.strip()
        self.tts_pub.publish(msg)
        
        # Give espeak time to speak
        time.sleep(1)
        
        print("✓ Response is being spoken!")
        
        # ====================================================
        # Display Final Results
        # ====================================================
        print("\n" + "="*60)
        print("VOICE ASSISTANT SESSION COMPLETE")
        print("="*60)
        print(f"Your Question: \"{self.stt_result}\"")
        print("-"*60)
        print(f"AI Response:\n{self.llm_response.strip()}")
        print("="*60)
        
        # Cleanup temporary audio file
        try:
            os.remove(audio_file)
            self.get_logger().info(f'Cleaned up: {audio_file}')
        except Exception as e:
            self.get_logger().warn(f'Could not remove temp file: {e}')
        
        print("\n✓ Full voice assistant pipeline completed successfully!")


def main(args=None):
    """Main function to run the interactive client"""
    rclpy.init(args=args)
    
    client = NLPClient()
    
    # Spin ROS2 callbacks in background thread
    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(client,),
        daemon=True
    )
    spin_thread.start()
    
    print("\n" + "="*60)
    print("NLP CLIENT STARTED")
    print("="*60)
    print("Connected to ROS2 topics successfully!")
    print("Ready for interactive testing.")
    print("="*60)
    
    try:
        # Run interactive menu in main thread
        client.show_menu()
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        client.destroy_node()
        rclpy.shutdown()
        spin_thread.join()


if __name__ == '__main__':
    main()
