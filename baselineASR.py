from soniox.speech_service import set_api_key
from typing import List, Tuple
import signal
import threading
import sys
from soniox.transcribe_live import transcribe_microphone
from soniox.speech_service import SpeechClient, Result
import soniox.capture_device
soniox.capture_device.PREFERRED_FRAME_SIZE = 256
import string

def split_words(result: Result) -> Tuple[List[str], List[str]]:
    final_words = []
    non_final_words = []
    for word in result.words:
        if word.is_final:
            final_words.append(word.text)
        else:
            non_final_words.append(word.text)
    return final_words, non_final_words


def clear_line():
    sys.stdout.write("\033[K")
    sys.stdout.write("\033[F")
    sys.stdout.write("\033[K")


def render_final_words(words: List[str]) -> List[str]:
    MAX_CHAR_PER_LINE = 100

    clear_line()
    clear_line()

    line = "".join(words)
    if len(line) <= MAX_CHAR_PER_LINE:
        print(line)
        return words

    # Find the word to break the line.
    idx = line.rfind(" ", 0, MAX_CHAR_PER_LINE)
    assert idx != -1

    # Print all the words until the break.
    print(line[:idx])

    # Print and return the remaining words after the break.
    line = line[idx + 1 :]
    print(line)
    return line.split()


def render_non_final_words(words: List[str]) -> None:
    if len(words) > 0:
        line = "".join(words)
        print(f"\n: {line}", end="")
    else:
        print("")


def clean_text(text):
    # Remove punctuation, lowercase, and extra spaces
    return ''.join(c for c in text if c not in string.punctuation).lower().strip()


def wer(ref: str, hyp: str) -> float:
    ref = clean_text(ref)
    hyp = clean_text(hyp)
    r = [w for w in ref.split() if w.strip()]
    h = [w for w in hyp.split() if w.strip()]
    d = [[0] * (len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1): d[i][0] = i
    for j in range(len(h)+1): d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
    return d[len(r)][len(h)] / max(1, len(r))


def smart_join(words):
    # Joins words so that punctuation marks are not preceded by a space
    out = ""
    for w in words:
        if w in string.punctuation:
            out = out.rstrip() + w
        else:
            out += (w + " ")
    return out.strip()


def wer_test_mode(sentences_file="sample_sentences.txt", output_file="wer_results.txt"):
    import os
    import time
    sentences = []
    # Try relative to script dir if not found
    if not os.path.exists(sentences_file):
        sentences_file = os.path.join(os.path.dirname(__file__), sentences_file)
    if not os.path.exists(sentences_file):
        print(f"Sentences file not found: {sentences_file}")
        return
    with open(sentences_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line)
    results = []
    wers = []
    with SpeechClient() as client:
        for idx, ground_truth in enumerate(sentences, 1):
            print(f"\nSentence {idx}: {ground_truth}")
            input("Press Enter when ready to start speaking...")
            print("Speak now. Press Enter when finished speaking.")
            stop_event = threading.Event()
            def stop_on_enter():
                input()
                stop_event.set()
            t = threading.Thread(target=stop_on_enter)
            t.start()
            final_words_last_line = []
            for result in transcribe_microphone(
                client,
                model="en_v2_lowlatency",
                include_nonfinal=True,
                stop_event=stop_event,
            ):
                new_final_words, _ = split_words(result)
                # Remove any 'words' that are just whitespace
                new_final_words = [w for w in new_final_words if w.strip() and w not in string.whitespace]
                final_words_last_line += new_final_words
                if stop_event.is_set():
                    break
            t.join()
            # Use smart join for display, join with spaces for WER
            recognized = smart_join(final_words_last_line)
            recognized_for_wer = " ".join(final_words_last_line)
            xwer = wer(ground_truth, recognized_for_wer)
            wers.append(xwer)
            results.append((ground_truth, recognized, xwer))
            print(f"WER: {xwer:.3f}")
            print(f"Recognized: {recognized}")
    avg_wer = sum(wers)/len(wers) if wers else 0.0
    # Write output
    with open(output_file, "w") as f:
        f.write(f"Average WER: {avg_wer:.3f}\n\n")
        for i, (gt, rec, xwer) in enumerate(results, 1):
            f.write(f"{i}. WER: {xwer:.3f}\nGround truth: \"{gt}\"\nRecognized: \"{rec}\"\n\n")
    print(f"\nAverage WER: {avg_wer:.3f}\nResults saved to {output_file}")


def main():
    stop_event = threading.Event()

    def sigint_handler(sig, stack):
        print("Interrupted, finishing transcription...")
        stop_event.set()

    signal.signal(signal.SIGINT, sigint_handler)

    with SpeechClient() as client:
        print("Transcribing from your microphone ...\n\n")

        final_words_last_line = []

        for result in transcribe_microphone(
            client,
            model="en_v2_lowlatency",
            include_nonfinal=True,
            stop_event=stop_event,
        ):
            # Split words into final words and non-final words.
            new_final_words, non_final_words = split_words(result)

            # Render final words in last line.
            final_words_last_line += new_final_words
            final_words_last_line = render_final_words(final_words_last_line)

            # Render non-final words.
            render_non_final_words(non_final_words)

    print("\nTranscription finished.")


if __name__ == "__main__":
    import sys
    set_api_key("cf8883cd94a4037c3b7829945335c03d0922928984c5d5c219afb1ee168b6643")
    if len(sys.argv) > 1 and sys.argv[1] == "--wer-test":
        wer_test_mode()
    else:
        main()