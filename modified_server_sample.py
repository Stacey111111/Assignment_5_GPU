def llm_callback(self, msg):
    prompt = msg.data

    system_prompt = "You are a helpful robot assistant. Be concise and clear."

    full_prompt = f"""[INST] <<SYS>>
{system_prompt}
<</SYS>>

{prompt} [/INST]"""

    output = self.llm(
        prompt=full_prompt,
        max_tokens=150,
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.1
    )

    response = output["choices"][0]["text"]

    # =========================
    # CLEAN OUTPUT (IMPORTANT)
    # =========================
    response = response.strip()

    # remove accidental prompt leakage
    if "[INST]" in response:
        response = response.split("[/INST]")[-1].strip()

    # =========================
    # BETTER STREAMING
    # =========================
    msg_out = String()

    for token in response.split(" "):
        msg_out.data = token + " "
        self.llm_pub.publish(msg_out)

    # end signal
    self.llm_pub.publish(String(data="[DONE]"))
