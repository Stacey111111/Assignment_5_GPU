# Assignment 5: Natural Language Processing Pipeline
## Video Presentation Script

---

## INTRODUCTION (30 seconds)

"Hello, this is my submission for Assignment 5: Natural Language Processing Pipeline using ROS2.

In this video, I will demonstrate:
- Part A: LLM Testing and Insights
- Part B: Code Architecture Walkthrough
- Part C: Live System Demonstration

Let's begin."

---

## PART A: LLM TESTING (3-4 minutes)

### Section 1: Model Comparison (Chat vs Instruct)

"First, let me explain the conceptual difference between chat and instruct LLM models.

**Chat Models** (like llama-2-7b-chat):
- These models are fine-tuned for conversational interactions
- They maintain context across multiple turns of dialogue
- They are optimized for natural, human-like responses
- Best for interactive applications like chatbots and voice assistants

**Instruct Models** (like llama-2-7b-32k-instruct):
- These models are fine-tuned to follow explicit instructions
- They excel at task completion and precise output formatting
- They are better at adhering to strict constraints and structured outputs
- Best for tasks requiring specific formats, like JSON generation or template filling

For this assignment, I chose the **chat model** because our voice assistant application requires natural conversational responses rather than strict instruction following."

---

### Section 2: System Prompt Effects

"Now, let me discuss the effects of using system prompts versus not using them.

**Without a System Prompt:**
- The model responds purely based on the user's query
- Responses can vary significantly in style and tone
- No consistent personality or role
- Example: Asking 'What is robotics?' gives a generic encyclopedia-style answer

**With a System Prompt:**
- We can define the model's role, behavior, and constraints
- Ensures consistent response style across conversations
- Can enforce specific formats or expertise levels
- Example: With a system prompt like 'You are a helpful robotics expert,' the same question gets a more focused, expert-level response with practical examples

In my implementation, I use a system prompt to make the assistant more helpful and concise, which is important for a voice interface where long responses can be tedious."

---

### Section 3: Prompting Techniques Comparison

"During testing, I experimented with various prompting techniques:

**1. Zero-Shot Prompting:**
- Direct questions without examples
- Example: 'What is artificial intelligence?'
- Result: The model provided accurate but sometimes verbose answers

**2. Few-Shot Prompting:**
- Providing examples in the prompt
- Example: 'Q: What is AI? A: [example]. Now, Q: What is robotics?'
- Result: More consistent formatting and style across responses

**3. Chain-of-Thought Prompting:**
- Asking the model to explain its reasoning
- Example: 'Explain step by step why robots need sensors'
- Result: More detailed, logical responses, but longer generation time

**4. Constraint-Based Prompting:**
- Specifying output requirements
- Example: 'Answer in one sentence'
- Result: Variable success - chat model sometimes ignored length constraints

For the voice assistant, I primarily use simple, direct prompting since the conversational context is provided by the chat model's training."

---

### Section 4: Task Execution Results

"Let me share some test results from Part 3 of the assignment:

**Test 1: Simple Factual Question**
- Prompt: 'What is the capital of France?'
- llama-2-7b-chat: ✓ Correctly answered 'Paris' with brief context
- llama-2-7b-32k-instruct: ✓ Also correct, slightly more formal

**Test 2: Strict Output Constraint**
- Prompt: 'List 3 robotics companies. Output ONLY a JSON array.'
- llama-2-7b-chat: ✗ Provided the list but added explanatory text
- llama-2-7b-32k-instruct: ✓ Followed the constraint precisely, output only JSON

**Test 3: Multi-Step Reasoning**
- Prompt: 'Explain how a robot localizes itself'
- llama-2-7b-chat: ✓ Provided conversational explanation with examples
- llama-2-7b-32k-instruct: ✓ More structured, step-by-step breakdown

**Test 4: Creative Task**
- Prompt: 'Write a poem about robots'
- llama-2-7b-chat: ✓ Generated creative, flowing poem
- llama-2-7b-32k-instruct: ✓ More structured, less creative

**Conclusion:**
The llama-2-7b-32k-instruct model was significantly better at following strict output constraints, especially for formatted outputs like JSON or specific length requirements. However, the llama-2-7b-chat model produced more natural, conversational responses that are better suited for our voice assistant application."

---

## PART B: CODE WALKTHROUGH (5-6 minutes)

### Opening Statement

"Now let me walk through the code architecture. I'll show the server code first, then the client code."

[Screen: Open sample_code_servers_gpu_topics.py]

---

### Section 1: Whisper Publisher Server

"First, let's look at the Whisper speech-to-text service.

[Scroll to Whisper class]

This is the WhisperPublisher class that handles speech-to-text transcription.

**Key Components:**

1. **Model Selection:**
```python
MODEL_PATH = "base"  # I chose the 'base' model
```

I selected the Whisper 'base' model for several reasons:
- It provides a good balance between accuracy and speed
- Size is only ~140MB, suitable for real-time applications
- Accuracy is sufficient for clear English speech
- Inference time is under 1 second on GPU, which is important for interactive voice assistant

Alternative models:
- 'tiny' would be faster but less accurate
- 'large' would be more accurate but too slow for real-time interaction

2. **Audio Capture:**
```python
def record_audio(self, duration, sample_rate=16000):
    audio_data = sd.rec(...)
    sd.wait()
```

This function captures audio from the microphone for the specified duration.

3. **ROS2 Integration:**
```python
self.stt_sub = self.create_subscription(
    Int32, '/stt_request', self.stt_callback, 10
)
self.stt_pub = self.create_publisher(
    String, '/stt_result', 10
)
```

- Subscribes to `/stt_request` topic - receives duration as Int32
- Publishes to `/stt_result` topic - sends transcribed text as String

4. **Transcription Pipeline:**
```python
def stt_callback(self, msg):
    duration = msg.data
    audio_file = self.record_audio(duration)
    result = self.model.transcribe(audio_file)
    transcribed_text = result['text']
    
    response_msg = String()
    response_msg.data = transcribed_text
    self.stt_pub.publish(response_msg)
```

The workflow is:
- Receive recording duration
- Record audio from microphone
- Transcribe using Whisper model
- Publish transcribed text to result topic"

---

### Section 2: LLaMA Publisher Server

"Next is the LLaMA language model service.

[Scroll to LLaMAPublisher class]

**Model Selection and Configuration:**

```python
MODEL_PATH = "llama-2-7b-chat.Q4_K_M.gguf"
```

I chose this model for several reasons:

1. **Model Type:** llama-2-7b-chat
   - Optimized for conversational interactions
   - Better for voice assistant than instruct model
   - More natural response style

2. **Quantization:** Q4_K_M (4-bit quantization)
   - Reduces model size from ~13GB to ~4GB
   - Faster inference speed
   - Minimal accuracy loss
   - Fits in GPU memory efficiently

**Inference Parameters:**

```python
def load_llama_model(self):
    self.llm = Llama(
        model_path=self.model_path,
        n_ctx=2048,        # Context window size
        n_threads=4,       # CPU threads for computation
        n_gpu_layers=-1,   # Use GPU for all layers
        verbose=False
    )
```

Parameter explanations:
- `n_ctx=2048`: Context window size - allows up to 2048 tokens in conversation history
- `n_threads=4`: Number of CPU threads for parallel processing
- `n_gpu_layers=-1`: Move all model layers to GPU for maximum speed
- `verbose=False`: Suppress debug output

**Generation Parameters:**

```python
def llm_callback(self, msg):
    prompt = msg.data
    
    output = self.llm(
        prompt,
        max_tokens=256,      # Maximum response length
        temperature=0.7,     # Randomness (0=deterministic, 1=creative)
        top_p=0.9,          # Nucleus sampling threshold
        top_k=40,           # Top-k sampling
        repeat_penalty=1.1,  # Penalize repetition
        stream=True         # Enable streaming output
    )
```

These parameters control response quality:
- `max_tokens=256`: Limit response length to ~200 words
- `temperature=0.7`: Balanced between accuracy and creativity
- `top_p=0.9`: Consider top 90% probability tokens
- `stream=True`: Enables real-time token streaming

**ROS2 Streaming Architecture:**

```python
self.llm_sub = self.create_subscription(
    String, '/llm_request', self.llm_callback, 10
)
self.llm_pub = self.create_publisher(
    String, '/llm_response_stream', 10
)
```

The streaming mechanism:
1. Receives prompt from `/llm_request` topic
2. Generates response token by token
3. Publishes each token to `/llm_response_stream`
4. Sends '[DONE]' marker when complete

This provides real-time feedback to users instead of waiting for complete response."

---

### Section 3: Espeak Publisher Server

"Finally, the espeak text-to-speech service.

[Scroll to EspeakPublisher class]

**Service Architecture:**

```python
def __init__(self):
    self.tts_sub = self.create_subscription(
        String, '/tts_request', self.tts_callback, 10
    )
```

Simple architecture:
- Subscribes to `/tts_request` topic
- Receives text as String message
- Uses espeak to generate speech audio

**TTS Callback:**

```python
def tts_callback(self, msg):
    text = msg.data
    
    # Execute espeak command
    subprocess.run(
        ['espeak', text],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
```

The process:
1. Extract text from message
2. Call espeak command-line tool
3. Audio is played directly through system speakers

Espeak is ideal for this application because:
- Fast response time
- No GPU required
- Clear pronunciation
- Available on all Linux systems"

---

### Section 4: Client Architecture

[Switch to sample_code_clients_topics.py]

"Now let's examine the client architecture.

**Publisher Setup:**

```python
self.tts_pub = self.create_publisher(String, '/tts_request', 10)
self.stt_pub = self.create_publisher(Int32, '/stt_request', 10)
self.llm_pub = self.create_publisher(String, '/llm_request', 10)
```

The client creates publishers to send requests to each service.

**Subscriber Setup:**

```python
self.stt_sub = self.create_subscription(
    String, '/stt_result', self.stt_callback, 10
)
self.llm_sub = self.create_subscription(
    String, '/llm_response_stream', self.llm_callback, 10
)
```

The client subscribes to results from Whisper and LLaMA services.

**Synchronization Mechanism:**

```python
self.stt_done = threading.Event()
self.llm_done = threading.Event()

def stt_callback(self, msg):
    self.stt_result = msg.data
    self.stt_done.set()  # Signal completion

def llm_callback(self, msg):
    if msg.data == "[DONE]":
        self.llm_done.set()  # Signal completion
```

Threading events ensure proper sequencing in the pipeline."

---

### Section 5: Option 4 Integration

"Now, the key part - Option 4 integration that implements the full voice assistant pipeline.

[Scroll to option_4_full_voice_assistant method]

**Pipeline Implementation:**

```python
def option_4_full_voice_assistant(self):
```

**Step 1: Record Audio**
```python
# Record from microphone
audio_file = self.record_audio(duration=duration)
```

Uses sounddevice library to capture audio locally.

**Step 2: Speech-to-Text**
```python
# Send duration to Whisper service
msg = Int32()
msg.data = duration
self.stt_done.clear()
self.stt_pub.publish(msg)

# Wait for transcription
self.stt_done.wait()
```

Process:
1. Clear the synchronization event
2. Publish recording duration to Whisper service
3. Block until transcription is received
4. Callback sets event when result arrives

**Step 3: Language Model Processing**
```python
# Send transcribed text to LLaMA
msg = String()
msg.data = self.stt_result
self.llm_done.clear()
self.llm_pub.publish(msg)

# Wait for response
self.llm_done.wait()
```

Process:
1. Take transcribed text from Step 2
2. Send to LLaMA service as prompt
3. Stream tokens as they arrive
4. Wait for '[DONE]' marker

**Step 4: Text-to-Speech**
```python
# Send LLaMA response to espeak
msg = String()
msg.data = self.llm_response.strip()
self.tts_pub.publish(msg)
```

Takes the complete LLaMA response and sends to espeak for audio output.

**Complete Data Flow:**

```
User Voice 
    ↓ (microphone)
Whisper Service (STT)
    ↓ (/stt_result topic)
LLaMA Service (LLM)
    ↓ (/llm_response_stream topic)
Espeak Service (TTS)
    ↓ (audio output)
User Hears Response
```

This creates a complete hands-free voice interaction loop."

---

## PART C: LIVE SYSTEM DEMONSTRATION (4-5 minutes)

### Opening

"Now I'll demonstrate the live system. Let me start all services."

[Screen: Show terminal with server running]

---

### Demo 1: Whisper (Speech-to-Text)

"First, let's test Whisper speech-to-text separately.

[Run client, select Option 2]

I'll speak clearly into the microphone."

[Speak into microphone]: "Hello, what can you tell me about robotics?"

[Show terminal output]

"As you can see:
1. The server received the recording request
2. It captured 5 seconds of audio
3. Whisper transcribed my speech
4. The result was published to /stt_result topic
5. The client received and displayed: 'Hello, what can you tell me about robotics?'

The transcription is accurate."

---

### Demo 2: LLaMA (Language Model)

"Next, let's test the LLaMA language model.

[Select Option 3]

I'll enter the transcribed question as a prompt."

[Type]: "Hello, what can you tell me about robotics?"

[Show terminal output]

"Observe the streaming output:
1. Each token appears in real-time
2. The response is conversational and natural
3. The chat model provides helpful information about robotics
4. When complete, '[DONE]' marker is sent

The full response is coherent and relevant to the question."

---

### Demo 3: Espeak (Text-to-Speech)

"Now let's test espeak text-to-speech.

[Select Option 1]

I'll use the LLaMA response from the previous test."

[Copy-paste LLaMA response]

[Audio plays through speakers]

"You can hear the espeak voice reading the response. The pronunciation is clear and understandable."

---

### Demo 4: Full Integration (Option 4)

"Finally, the complete voice assistant pipeline - Option 4.

[Select Option 4]

This will demonstrate all three services working together with no keyboard input required after starting.

Watch the four-step process:"

[Show recording starting]

[Speak into microphone]: "What are the three laws of robotics?"

"**Step 1: Recording Audio**
- 5 seconds of audio captured
- Countdown displayed
- Recording complete

**Step 2: Transcribing Speech**
- Sending to Whisper service
- Processing audio..."

[Wait for transcription]

[Terminal shows]: "Transcription: 'What are the three laws of robotics?'"

"Perfect transcription!

**Step 3: Generating AI Response**
- Sending question to LLaMA
- Watch the tokens stream in real-time..."

[Show streaming response]

"The LLaMA model is explaining Asimov's Three Laws of Robotics. Notice how the response:
- Is conversational in tone
- Provides relevant information
- Streams token by token for real-time feedback

**Step 4: Speaking Response**
- Sending to espeak service
- Audio output beginning..."

[Audio plays through speakers]

"You can hear the complete response being read aloud.

**Summary of what happened:**
1. ✓ My voice was captured and transcribed
2. ✓ The transcription was sent to LLaMA
3. ✓ LLaMA generated an intelligent response
4. ✓ The response was spoken back to me

This is a complete voice assistant interaction with no manual typing required after the initial trigger."

---

## CONCLUSION (30 seconds)

"To summarize, in this video I have demonstrated:

**Part A - LLM Testing:**
- Explained differences between chat and instruct models
- Discussed effects of system prompts
- Shared insights on prompting techniques
- Showed test results comparing both models

**Part B - Code Walkthrough:**
- Explained Whisper service with 'base' model selection
- Explained LLaMA service with llama-2-7b-chat.Q4_K_M model and parameters
- Explained espeak service architecture
- Showed client integration and Option 4 implementation

**Part C - Live Demonstration:**
- Tested Whisper speech-to-text
- Tested LLaMA text generation
- Tested espeak text-to-speech
- Demonstrated full integrated voice assistant pipeline

All components are working correctly and communicating through ROS2 topics.

Thank you for watching!"

---

## TECHNICAL NOTES FOR RECORDING

### Camera/Screen Setup:
1. **Part A**: Can be talking head or screen with notes
2. **Part B**: Screen recording with code visible, use cursor/highlighting
3. **Part C**: Screen recording showing terminal output clearly

### Audio Tips:
- Use good microphone for clear narration
- Keep consistent distance from mic
- Minimize background noise
- For Part C demos, ensure espeak output is audible

### Editing Tips:
- Add title cards for each part
- Use zoom/highlight for code sections
- Add timestamps in video description
- Keep total length around 12-15 minutes

### Terminal Setup for Part C:
```bash
# Terminal 1: Server (larger font)
python3 sample_code_servers_gpu_topics.py

# Terminal 2: Client (larger font)
python3 sample_code_clients_topics.py
```

Use `Ctrl+Shift++` to increase terminal font size for visibility.

---

## TROUBLESHOOTING NOTES

If during recording:

**Whisper fails:**
- Check microphone permissions
- Verify sounddevice is installed
- Test with: `python3 -c "import sounddevice; print(sounddevice.query_devices())"`

**LLaMA is slow:**
- Mention GPU is warming up (first inference slower)
- Acceptable to speed up video in post-processing
- Or cut to "...processing..." and resume

**Espeak not audible:**
- Increase system volume
- Test with: `espeak "test"`
- May need to adjust recording setup

**Topics not connecting:**
- Verify both server and client running
- Check: `ros2 topic list`
- Restart both if needed

---

## SUBMISSION CHECKLIST

- [ ] Video uploaded and link provided
- [ ] All three parts (A, B, C) covered
- [ ] Code visible and explained clearly
- [ ] Live demonstrations successful
- [ ] Audio quality good
- [ ] Length appropriate (12-15 minutes)
- [ ] Source code files submitted
- [ ] README with video link included

---

Good luck with your recording! 🎥🤖
