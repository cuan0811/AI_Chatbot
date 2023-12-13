def send_message(event=None):
    msg = EntryBox.get("1.0", "end-1c").strip()
    EntryBox.delete("0.0", END)

    if msg != "":
        chat_log.config(state=NORMAL)
        chat_log.insert(END, "You: " + msg + "\n\n")
        chat_log.config(foreground="#442265", font=("Verdana", 12))
        
        if msg == "send image":
            send_image()
        else:
            res = chatbot_response(msg)
            chat_log.insert(END, "Bot: " + res + "\n\n")
        
        chat_log.config(state=DISABLED)
        chat_log.yview(END)