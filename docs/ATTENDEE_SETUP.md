# Before the workshop: attendee setup

**Paminaw! Teaching Machines to Listen in Cebuano and Tagalog** · PyCon Davao 2026 ·
Saturday 17 October, 1:10–3:10 PM · Ateneo de Davao University

You do **not** need to install anything. Everything runs in Google Colab, in your browser, on
Google's computers.

## Bring

- [ ] A **laptop**, fully charged (bring the charger too; outlets may be limited)
- [ ] **Earphones** or headphones: one exercise is listening to audio
- [ ] Your **Google account** password: Colab needs you to sign in

## Do this before Saturday (5 minutes, optional but recommended)

1. Open the workshop notebook from the link the facilitators shared (it is also the
   **Open in Colab** badge in the [README](../README.md)).
2. Sign in to Google if asked. If Colab says the notebook is "not authored by Google", click
   **Run anyway**: that is normal for notebooks from GitHub.
3. Click into **cell 0** and press **Shift+Enter**. Wait about a minute for the line
   **READY**.
4. Done. You can close the tab. On the day, you will open a fresh copy and run cell 0 again.

If cell 0 prints a line that starts with **Error:**, read the **Fix:** part of the message. It tells you what to do. If
you are still stuck, open an issue on GitHub or ask us at the start of the workshop.

## Good to know

- **No GPU needed.** A free CPU runtime runs every lab. If Colab offers you a free GPU
  (Runtime → Change runtime type → T4 GPU), Lab 2b will also run the MMS model live.
- **No Python experience beyond the basics.** If you can read a `for` loop and call a function,
  you are ready. We explain PyTorch, audio and speech recognition from scratch.
- **If the venue Wi-Fi is slow:** Colab downloads models and audio on Google's servers, not
  through your laptop. The notebook is also saved **with its outputs**, so you can read along
  even if your runtime is not working.
- **Data:** we use FLEURS, a public dataset of people reading Wikipedia-style sentences aloud
  (CC BY 4.0). No personal or medical recordings are involved.

See you on Saturday. *Kita-kits!*
