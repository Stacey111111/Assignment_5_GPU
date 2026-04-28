# Assignment 5 - ULTRA SHORT Script (4-5 Minutes STRICT)

## ⏱️ EXACT TIMING
```
00:00-00:15  Introduction (15s)
00:15-01:00  Part A: LLM Testing (45s)
01:00-02:45  Part B: Code Walkthrough (1min 45s)
02:45-04:45  Part C: Live Demo (2min)
04:45-05:00  Conclusion (15s)
```

---

## 🎬 [00:00-00:15] INTRODUCTION (15 seconds)

"This is my Assignment 5 submission. I'll demonstrate LLM testing, explain the code architecture, and show a complete voice assistant pipeline using Whisper, LLaMA, and espeak."

---

## 📊 [00:15-01:00] PART A: LLM TESTING (45 seconds)

"**Model Comparison:**
Chat models provide natural conversation - I chose llama-2-7b-chat for our voice assistant. Instruct models better follow strict constraints like JSON formatting.

**System Prompts:**
Adding a system prompt like 'You are a helpful robot assistant' keeps responses focused and concise.

**Test Results:**
The instruct model was better at strict output constraints, but the chat model gave more natural responses - perfect for voice interaction."

---

## 💻 [01:00-02:45] PART B: CODE WALKTHROUGH (1 min 45s)

### [01:00-01:45] Server (45s)

[Screen: sample_code_servers_gpu_topics.py, line 114]

"**Whisper - Line 114:**
```python
self.whisper_model_name = "base"
```
Base model balances speed and accuracy - under 1 second inference, 140MB size.

**LLaMA:**
I use llama-2-7b-chat.Q4_K_M - chat model for conversation, Q4 quantization reduces size to 4GB. Key parameters: max_tokens=256, temperature=0.7, stream=True for real-time output.

**Espeak:**
Simple text-to-speech using system command."

### [01:45-02:45] Client Integration (1 min)

[Screen: sample_code_clients_topics.py, option_4 function]

"**Option 4 Pipeline:**

Step 1: Record audio
Step 2: Publish to /stt_request, wait for Whisper transcription  
Step 3: Send text to /llm_request, stream LLaMA response
Step 4: Send to /tts_request for espeak output

Threading events ensure proper sequence - each step waits for the previous to complete."

---

## 🎮 [02:45-04:45] PART C: LIVE DEMO (2 minutes)

[Terminal visible]

"**Complete Pipeline - Option 4:**"

[Select Option 4, speak into mic]

**[Say]:** "What are the three laws of robotics?"

[As demo runs, narrate briefly:]

"Recording... transcribing... 'What are the three laws of robotics' - accurate.

LLaMA generating response... streaming tokens in real-time... explains Asimov's laws conversationally.

Espeak speaking the response...

[Let audio play]

Complete voice interaction - question asked, intelligent response received, no typing needed."

---

## 🎯 [04:45-05:00] CONCLUSION (15 seconds)

"I've explained LLM model differences, shown the ROS2 architecture with all three services, and demonstrated a working voice assistant. Thank you."

---

## 📝 RECORDING INSTRUCTIONS

### Preparation:
1. **Pre-record or speed up Part C** - LLaMA may take 10-30 seconds to generate
   - Record at normal speed, then speed up to 1.5-2x in editing
   - OR cut away during generation, cut back to show result
   
2. **Have code files open at exact lines:**
   - Line 114 for Whisper
   - LLaMA section visible
   - Option 4 function ready to scroll

3. **Terminal ready:**
   - Server running (background)
   - Client at menu (ready for Option 4)

### Speaking Pace:
- **Part A & B:** Talk at 150 words/minute (slightly fast but clear)
- **Part C:** Let demo run, narrate briefly
- **Total speaking:** ~300 words = 2 minutes
- **Demo watching:** ~2 minutes  
- **Intro/Outro:** ~30 seconds

### Editing Tips:
```
If LLaMA takes 30 seconds to generate:
→ Show it starting
→ Cut to "processing..." text overlay (5 seconds)
→ Cut back to show streaming output
→ Keeps video dynamic, saves time
```

---

## 🎯 ABSOLUTE MINIMUM VERSION (If still too long)

If you need to cut MORE, here's the 3.5 minute version:

### [0:00-0:10] Intro
"Assignment 5: NLP pipeline with Whisper, LLaMA, and espeak."

### [0:10-0:35] Part A (25s)
"Chat models for conversation, instruct models for strict constraints. I chose chat model. System prompts focus responses."

### [0:35-1:50] Part B (1min 15s)
"Whisper base model - line 114 - fast and accurate.
LLaMA chat model with Q4 quantization - conversational responses.
Option 4 chains all three: record → transcribe → generate → speak."

### [1:50-3:30] Part C (1min 40s)
[Just show Option 4 demo with minimal narration]

### [3:30-3:40] Outro
"Complete voice assistant demonstrated. Thank you."

---

## ✅ FINAL CHECKLIST

Before recording:
- [ ] Script memorized or on second screen
- [ ] Code open at line 114
- [ ] Terminal ready with server running
- [ ] Microphone working
- [ ] Timer ready

During recording:
- [ ] Speak clearly at 150 wpm
- [ ] Don't pause between parts
- [ ] Keep camera/screen transitions smooth
- [ ] Let Option 4 demo run (can edit later)

After recording:
- [ ] Check total time (should be 4:30-5:00)
- [ ] Speed up slow sections if needed
- [ ] Add timestamps if helpful

---

## 🎤 WORD COUNT

This ultra-short script:
- **Spoken words:** ~350 words
- **Speaking time:** 350 ÷ 150 wpm = 2.3 minutes
- **Demo time:** ~2 minutes (can edit to 1.5 min)
- **Transitions:** ~30 seconds
- **TOTAL:** ~5 minutes

---

**This version WILL fit in 4-5 minutes!** 🎯

Key: Talk slightly faster, keep demos short, no pauses between sections.

Good luck! 🎥✨
