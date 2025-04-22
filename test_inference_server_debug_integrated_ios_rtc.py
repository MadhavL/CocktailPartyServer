import asyncio
import websockets
import numpy as np
import torch
import yaml
import json
import re
import math
from models.models import JNF_SSF as Model
from models.exp_ssf import SSFExp as Exp
from soniox.speech_service import SpeechClient
from soniox.transcribe_live import transcribe_stream
from scipy.io.wavfile import write as write_wav
import threading
from scipy.signal import resample_poly
import signal
from soniox.speech_service import set_api_key
import queue
from collections import deque, defaultdict
import time
import grpc
from concurrent import futures
import transcribeglass_pb2
import transcribeglass_pb2_grpc
from typing import Set
from asyncio import Queue
import time
import signal
import atexit

# Globals for timing
process_chunk_times = []
process_chunk_count = 0

def print_average_process_chunk_time():
    if process_chunk_times:
        avg_time = sum(process_chunk_times) / len(process_chunk_times)
        print(f"\nAverage process_chunk() time: {avg_time*1000:.3f} ms over {len(process_chunk_times)} calls")
    else:
        print("\nNo process_chunk() calls recorded.")

# Register to print average on exit
atexit.register(print_average_process_chunk_time)

def signal_handler(sig, frame):
    print_average_process_chunk_time()
    exit(0)
signal.signal(signal.SIGINT, signal_handler)

class GRPCClient:
    def __init__(self, stream_writer: asyncio.Queue):
        self.queue = stream_writer

grpc_ui_clients: Set[GRPCClient] = set()

raw_received_chunks = []
sent_to_soniox_chunks = []
debug_downsampled_audio = []
debug_normalized_audio = []
debug_model_output_audio = []
debug_overlap_output_audio = []
smoothing_window = 500
source_history = defaultdict(lambda: deque(maxlen=smoothing_window))
next_source_id = 1
smoothed_sources = []
last_broadcast_time = 0
next_custom_id = 1
active_custom_id = None
odas_id_to_custom_id = {}
# Global source tracking results
current_sources = []
active_odas_id = None
sources_lock = threading.Lock()
all_sources_by_odas_id = {}  # key: odas_id → Source object
micstreamer_control_queue = asyncio.Queue()
SAMPLE_RATE=16000
SOURCE_ID_BYTES = 4  # send as 32-bit int

speaker_switch = False
old_source_id = -1


AMPLIFY_DB = 65  # dB gain

# ========== CONFIGURATION ========== #
N_CHANNELS = 6
SAMPLE_WIDTH = 4  # ← because we're using float32
CHUNK_FRAMES = 4096
CHUNK_SIZE = CHUNK_FRAMES * N_CHANNELS * SAMPLE_WIDTH
TARGET_ANGLE = -90
CKPT_PATH = "ckpts/latest_model_499.ckpt"
CONFIG_PATH = "config/ssf_config.yaml"
OVERLAP = 512  # overlap between chunks to enable smooth transitions
ODAS_SERVER_PORT = 9000  # Port to listen for ODAS SST data

# ========== SOURCE TRACKING ========== #

class TranscribeGlassServicer(transcribeglass_pb2_grpc.TranscribeGlassServicer):
    async def StreamInteraction(self, request_iterator, context):
        print("🖥 gRPC UI client connected")

        client_queue = asyncio.Queue()
        grpc_client = GRPCClient(client_queue)
        grpc_ui_clients.add(grpc_client)

        # Handle incoming requests in background (source selection etc.)
        async def receive_loop():
            global active_custom_id
            global speaker_switch
            try:
                async for req in request_iterator:
                    if req.HasField("select_source"):
                        selected = req.select_source.source_id
                        with sources_lock:
                            active_custom_id = selected
                        print(f"🎯 UI selected source: {selected}")
                        speaker_switch = True

                        # Notify mic streamer client (if connected)
                        await micstreamer_control_queue.put(json.dumps({
                            "type": "select_source",
                            "source_id": selected
                        }))
                    elif req.HasField("heartbeat"):
                        print(f"💓 Heartbeat from {req.heartbeat.client_id}")
            except Exception as e:
                print(f"❌ UI client stream closed: {e}")

        asyncio.create_task(receive_loop())

        # Yield messages from broadcast queue
        try:
            while True:
                message = await client_queue.get()
                yield message
        except asyncio.CancelledError:
            print("🛑 Client stream cancelled")
        finally:
            grpc_ui_clients.discard(grpc_client)


def serve_grpc():
    server = grpc.aio.server()
    transcribeglass_pb2_grpc.add_TranscribeGlassServicer_to_server(
        TranscribeGlassServicer(), server
    )
    server.add_insecure_port("[::]:50051")
    return server
#================================#

class Source:
    def __init__(self, odas_id, custom_id, tag, x, y, z, activity):
        self.odas_id = odas_id
        self.custom_id = custom_id
        self.tag = tag
        self.x = x
        self.y = y
        self.z = z
        self.activity = activity

    def direction_of_arrival(self):
        angle_radians = math.atan2(self.y, self.x)
        angle_degrees = angle_radians * (180.0 / math.pi)
        if angle_degrees < 0:
            angle_degrees += 360
        return angle_degrees



# ========== LOAD MODEL ========== #
with open(CONFIG_PATH) as config_file:
    config = yaml.safe_load(config_file)

stft_length = 512
stft_shift = 256
config['data']['stft_length_samples'] = stft_length
config['data']['stft_shift_samples'] = stft_shift
config['network']['causal'] = True

model = Model(**config['network'])
exp = Exp.load_from_checkpoint(
    CKPT_PATH,
    model=model,
    stft_length=stft_length,
    stft_shift=stft_shift,
    **config['experiment']
).eval()

def should_broadcast():
    global last_broadcast_time
    now = time.time()
    if now - last_broadcast_time >= 0.1:
        last_broadcast_time = now
        return True
    return False

# ========== WINDOW FOR OVERLAP-ADD ========== #
# Create a Hann window for smooth transitions
def create_window(chunk_size, overlap):
    fade_len = overlap
    window = np.ones(chunk_size)
    # Apply fade-in
    window[:fade_len] = np.hanning(fade_len * 2)[:fade_len]
    # Apply fade-out
    window[-fade_len:] = np.hanning(fade_len * 2)[fade_len:]
    return window

def convert_angle(x: float) -> int:
    # Calculate 270 - x
    diff = 270.0 - x

    # Get the proper modulo in [0, 360)
    mod = diff % 360.0
    if mod < 0:
        mod += 360.0

    # Subtract 180 to get to the [-180, 180) range
    result = mod - 180.0

    # Round to nearest even number
    even_rounded = int(round(result / 2.0) * 2)

    return even_rounded


def weighted_smooth(activity_deque):
    values = np.array(activity_deque)
    weights = values ** 2  # nonlinear emphasis
    if weights.sum() == 0:
        return 0.0
    return float(np.sum(values * weights) / np.sum(weights))

# ========== ENHANCE STREAM ========== #
def enhance_stream_from_websocket() -> callable:
    # Create window for overlap-add
    window = create_window(CHUNK_FRAMES, OVERLAP)
    output_buffer = np.zeros(OVERLAP, dtype=np.float32)
    output_audio_chunks = []
    
    def update_target_direction():
        global active_custom_id
        with sources_lock:
            for s in current_sources:
                if s.custom_id == active_custom_id:
                    doa = convert_angle(s.direction_of_arrival())
                    # print(f"🎯 DOA for ODAS ID {s.odas_id}: {doa}")
                    return torch.tensor([doa], dtype=torch.long, device=exp.device)
        print("⚠️ No active ODAS source found. Using default angle.")
        return torch.tensor([TARGET_ANGLE], dtype=torch.long, device=exp.device)

    def process_chunk(chunk: np.ndarray) -> bytes:
        nonlocal output_buffer
        
        # Update target direction based on latest SST data

        dynamic_target = update_target_direction()
        target_enc = exp.encode_condition(dynamic_target)
        
        #---------------------HARDCODING FOR NOW--------------#
        
        # Just amplify the full chunk and feed into model
        amplify_gain = 10 ** (AMPLIFY_DB / 20)
        chunk_amplified = chunk * amplify_gain  # shape: (6, 4096)
        
        # Save amplified input (no windowing)
        debug_normalized_audio.append(chunk_amplified.T.copy())
        
        # Feed to model
        input_tensor = torch.from_numpy(chunk_amplified).unsqueeze(0).to(dtype=torch.float32, device=exp.device)
        noisy_stft = exp.get_stft_rep(input_tensor)[0]
        stacked = torch.cat((torch.real(noisy_stft), torch.imag(noisy_stft)), dim=1)
        
        with torch.no_grad():
            mask_stacked = exp.model(stacked, target_enc, device=exp.device) # TEST HERE!
            speech_mask, *_ = exp.get_complex_masks_from_stacked(mask_stacked)
            enhanced_stft = noisy_stft[:, exp.reference_channel, ...] * speech_mask
            *_, enhanced_td = exp.get_td_rep(noisy_stft, noisy_stft, enhanced_stft)
        
        enhanced_np = enhanced_td.squeeze(0).cpu().numpy()
        if enhanced_np.ndim > 1:
            enhanced_np = enhanced_np.mean(axis=0)
        
        if len(enhanced_np) < CHUNK_FRAMES:
            print(f"Padding short chunk: {len(enhanced_np)} samples")
            enhanced_np = np.pad(enhanced_np, (0, CHUNK_FRAMES - len(enhanced_np)))
            
        # Apply window for smooth transitions
        enhanced_np = enhanced_np * window
        
        # Overlap-add for smooth transitions
        enhanced_np[:OVERLAP] += output_buffer
        output_buffer = enhanced_np[-OVERLAP:].copy()
        
        # Output the non-overlapping portion only
        output_chunk = enhanced_np
        # # Extract the part we want to send (excluding the next overlap)
        # output_chunk = enhanced_np[:CHUNK_SIZE-OVERLAP]
        
        # Normalize and store for debug
        output_chunk = output_chunk / (np.max(np.abs(output_chunk)) + 1e-9)
        debug_overlap_output_audio.append(output_chunk.copy())
        
        # Convert to int16
        pcm_int16 = (output_chunk * 32767).clip(-32768, 32767).astype("<i2")
        output_audio_chunks.append(pcm_int16)
        
        return pcm_int16.tobytes()
    
    def finalize():
        # === Save full-length debug audio ===
        def save_debug(name, data, sr=16000):
            if not data:
                return
            audio = np.concatenate(data)
            int16 = (audio / (np.max(np.abs(audio)) + 1e-9) * 32767).clip(-32768, 32767).astype("<i2")
            write_wav(f"audio/debug_{name}.wav", sr, int16)
            print(f"💾 Saved audio/debug_{name}.wav")
        
        # save_debug("A_downsampled_input", debug_downsampled_audio)
        save_debug("B_normalized_input", debug_normalized_audio)
        # save_debug("C_model_output", debug_model_output_audio)
        save_debug("D_overlap_output", debug_overlap_output_audio)
    
    return process_chunk, finalize

def downsample_to_16k(audio_48k: np.ndarray) -> np.ndarray:
    # audio_48k: (channels, samples) @ 48kHz
    return resample_poly(audio_48k, up=1, down=3, axis=1).astype(np.float32)

# ========== ODAS SST TCP HANDLER ========== #
async def start_odas_listener():
    server = await asyncio.start_server(
        handle_odas_connection, '0.0.0.0', ODAS_SERVER_PORT
    )
    print(f"🎯 ODAS SST server started on port {ODAS_SERVER_PORT}")
    async with server:
        await server.serve_forever()

async def handle_odas_connection(reader, writer):
    print(f"🔌 ODAS client connected from {writer.get_extra_info('peername')}")
    buffer = b""
    try:
        while True:
            chunk = await reader.read(4096)
            if not chunk:
                break
            buffer += chunk

            # Use regex to split on }{ boundaries
            parts = re.split(rb'}\s*\{', buffer)
            if len(parts) == 1:
                continue  # Wait for more data

            for i in range(len(parts) - 1):
                # Recombine the split pieces into valid JSON
                raw_json = parts[i] + b'}'
                if i != 0:
                    raw_json = b'{' + raw_json  # Re-add leading brace if not first part
                try:
                    sst_data = json.loads(raw_json.decode("utf-8"))
                    process_sst_data(sst_data)
                except json.JSONDecodeError as e:
                    print(f"⚠️ Invalid JSON: {e} → {raw_json}")

            # Save the last part (which might be partial JSON) for next iteration
            buffer = b'{' + parts[-1]

    except Exception as e:
        print(f"❌ ODAS connection error: {e}")
    finally:
        writer.close()
        await writer.wait_closed()
        print("🔌 ODAS client disconnected")


def process_sst_data(sst_data):
    global current_sources, smoothed_sources, next_custom_id

    if "src" not in sst_data:
        return

    received_odas_ids = set()
    
    for src in sst_data["src"]:
        try:
            odas_id = src.get("id", -10)
            # print(f"Processing source from ODAS with odas_id: {odas_id}, activity: {src.get("activity", -10)}")
            activity = float(src.get("activity", 0))
            received_odas_ids.add(odas_id)

            if odas_id not in odas_id_to_custom_id:
                odas_id_to_custom_id[odas_id] = next_custom_id
                next_custom_id += 1
            custom_id = odas_id_to_custom_id[odas_id]

            source_obj = Source(
                odas_id=odas_id,
                custom_id=custom_id,
                tag=src.get("tag", ""),
                x=float(src.get("x", 0)),
                y=float(src.get("y", 0)),
                z=float(src.get("z", 0)),
                activity=activity
            )
            
            all_sources_by_odas_id[odas_id] = source_obj
            source_history[odas_id].append(activity)

        except Exception as e:
            print(f"⚠️ Error processing source: {e}")

    # For all known sources, if they weren't in this update, append a 0
    for odas_id in all_sources_by_odas_id:
        if odas_id not in received_odas_ids:
            source_history[odas_id].append(0.0)

    with sources_lock:
        current_sources = list(all_sources_by_odas_id.values())
        smoothed_sources = []

        for odas_id, source_obj in all_sources_by_odas_id.items():
            smoothed = weighted_smooth(source_history[odas_id])
            # print(f"Smoothed for Odas ID {odas_id} = {smoothed:.3f}")
            if smoothed > 0.3:
                smoothed_sources.append({
                    "id": source_obj.custom_id,
                    "x": source_obj.x,
                    "y": source_obj.y,
                    "activity": smoothed,
                    "angle": convert_angle(source_obj.direction_of_arrival()),
                    "obj": source_obj
                })

        if smoothed_sources and should_broadcast():
            asyncio.get_event_loop().create_task(broadcast_to_grpc_clients({
                "type": "sources",
                "sources": [
                    {
                        "id": s["id"],
                        "x": s["x"],
                        "y": s["y"],
                        "activity": s["activity"],
                        "angle": s["angle"]
                    } for s in smoothed_sources
                ]
            }))


# ========== WEBSOCKET HANDLER ========== #
async def handler(websocket):
    print("📡 Audio client connected")
    buffer = b""
    process_chunk, finalize = enhance_stream_from_websocket()
    
    # Communication channel to send enhanced audio to Soniox from another thread
    audio_queue = queue.Queue()
    source_id_queue = queue.Queue()
    stop_event = threading.Event()
    
    loop = asyncio.get_event_loop()

    async def send_control_messages(ws):
        while True:
            msg = await micstreamer_control_queue.get()
            try:
                print("SENDING CONTROL MESSAGE!")
                await ws.send(msg.encode('utf-8'))
            except Exception as e:
                print(f"❌ Failed to send control msg to client: {e}")

    # === TRANSCRIPTION THREAD ===
    def transcription_worker(loop):
        global speaker_switch
        global old_source_id
        try:
            # Cumulative duration in ms of audio sent to Soniox
            cumulative_audio_time_ms = 0

            # Each entry: (start_ms, end_ms, source_id)
            audio_source_history = deque(maxlen=2000)

            def audio_generator():
                nonlocal cumulative_audio_time_ms

                while not stop_event.is_set():
                    try:
                        item = audio_queue.get(timeout=1.0)
                        if item is None:
                            break
                        audio, chunk_source_id = item

                        # Assume 4096 frames @ 16kHz = 256ms
                        chunk_duration_ms = int((CHUNK_FRAMES / SAMPLE_RATE) * 1000)

                        start_ms = cumulative_audio_time_ms
                        end_ms = start_ms + chunk_duration_ms
                        cumulative_audio_time_ms = end_ms

                        audio_source_history.append((start_ms, end_ms, chunk_source_id))

                        print(f"📦 Tagged chunk [{start_ms}ms → {end_ms}ms] with Source ID {chunk_source_id}")
                        yield audio
                    except queue.Empty:
                        continue

            def get_source_for_word(word_start_ms: int):
                for start_ms, end_ms, src_id in reversed(audio_source_history):
                    if start_ms <= (word_start_ms - 300) < end_ms:
                        return src_id
                return None

            speaker_final_tokens = defaultdict(list)
            speaker_final_tokens_current = defaultdict(list)

            with SpeechClient() as client:
                for result in transcribe_stream(
                    audio_generator(),
                    client,
                    model="en_v2_lowlatency",
                    include_nonfinal=True,
                    audio_format="pcm_s16le",
                    sample_rate_hertz=16000,
                    num_audio_channels=1,
                ):
                    words = result.words
                    if not words:
                        continue

                    # print(words)

                    
                    result_source_id = None
                    response_start = 0

                    # Use first word with a valid start time to determine speaker
                    if not speaker_switch:
                        result_source_id = old_source_id
                        
                    else:
                        first_word_id = None
                        last_word_id = None
                        first_word = words[0]
                        last_word = words[-1]

                        if first_word.text.strip():
                            first_word_id = get_source_for_word(first_word.start_ms) # Get the source for the first word
                            if first_word_id:
                                print(f"First word id: {first_word_id}, old id: {old_source_id}")
                                if first_word_id != old_source_id: # If the first word is already the new source, then we can go back to normal processing
                                    speaker_switch = False
                                    speaker_final_tokens_current[old_source_id] = [] # Reset the final tokens for the old speaker after the switch!
                                    result_source_id = first_word_id
                                    old_source_id = first_word_id

                                else: # First word is not the new source, it is the old source
                                     # Get the source of the last word
                                     if last_word.text.strip():
                                        last_word_id = get_source_for_word(last_word.start_ms)
                                        if last_word_id:
                                            print(f"Last word {last_word.text}, time: {last_word.start_ms}, id: {last_word_id}")
                                            if first_word_id != last_word_id:
                                                # SWITCH HAS OCCURED WITHIN THIS RESULT!
                                                for i, word in enumerate(words):
                                                    word_id = get_source_for_word(word.start_ms)
                                                    # Find the first word where the switch occured
                                                    if word_id != old_source_id:
                                                        response_start = i
                                                        result_source_id = word_id

                                                        # Finalize any un-finalized words for the old speaker before the switch so we don't lose that!
                                                        final_tokens = [w.text.strip() for w in words[response_start:] if w.is_final and w.text.strip()]
                                                        speaker_final_tokens[old_source_id] += final_tokens

                                                        # Send this as the last finalization of previous speaker!
                                                        break                                                            
                                            else: # Switch has not occured, old source transcript still being processed
                                                result_source_id = old_source_id
                    
                    print(f"Result source id: {result_source_id}, response_start: {response_start}")
                    if result_source_id is None:
                        continue

                    final_tokens = [w.text.strip() for w in words[response_start:] if w.is_final and w.text.strip()]
                    nonfinal_tokens = [w.text.strip() for w in words[response_start:] if not w.is_final and w.text.strip()]

                    speaker_final_tokens[result_source_id] += final_tokens
                    speaker_final_tokens_current[result_source_id] += final_tokens

                    final_str = " ".join(speaker_final_tokens_current[result_source_id]) # Swap with non-current version if you want the entire history
                    nonfinal_str = " ".join(nonfinal_tokens)
                    full_transcript = (final_str + " " + nonfinal_str).strip()

                    full_transcript = re.sub(r'\s+([.,!?;:])', r'\1', full_transcript)
                    full_transcript = re.sub(r'\s{2,}', ' ', full_transcript)

                    print(f"Sending result_source_id: {result_source_id}\ntranscript: {full_transcript}")

                    asyncio.run_coroutine_threadsafe(
                        broadcast_to_grpc_clients({
                            "type": "transcript",
                            "source_id": result_source_id,
                            "text": full_transcript
                        }),
                        loop
                    )
        except Exception as e:
            print(f"❌ Transcription error: {e}")
    
    # Start transcription thread
    transcription_thread = threading.Thread(target=transcription_worker, args=(loop,), daemon=True).start()
    
    try:
        control_task = asyncio.create_task(send_control_messages(websocket))
        async for message in websocket:
            buffer += message
            while len(buffer) >= CHUNK_SIZE + SOURCE_ID_BYTES:
                source_id_bytes = buffer[:SOURCE_ID_BYTES]
                chunk_bytes = buffer[SOURCE_ID_BYTES:SOURCE_ID_BYTES + CHUNK_SIZE]
                buffer = buffer[SOURCE_ID_BYTES + CHUNK_SIZE:]
                
                float_data = np.frombuffer(chunk_bytes, dtype=np.float32)
                # print(f"📦 Received {len(float_data)} floats → Reshaped to ({float_data.shape[0] // N_CHANNELS}, {N_CHANNELS})")
                audio = float_data.reshape(-1, N_CHANNELS).T  # already float32
                raw_received_chunks.append(audio.T.copy())  # Shape: (frames, channels)
                
                start_time = time.perf_counter()
                enhanced_audio = process_chunk(audio)
                end_time = time.perf_counter()
                process_chunk_times.append(end_time - start_time)
                
                # Save enhanced audio being sent to Soniox
                sent_to_soniox_chunks.append(np.frombuffer(enhanced_audio, dtype=np.int16).copy())
                # with sources_lock:
                #     chunk_source_id = active_custom_id  # snapshot the source at time of audio
                chunk_source_id = int.from_bytes(source_id_bytes, byteorder='little', signed=True)

                # print(f"Audio queue source id: {chunk_source_id}")
                audio_queue.put((enhanced_audio, chunk_source_id))
                
    except websockets.exceptions.ConnectionClosed as e:
        print(f"🚪 Audio client disconnected: {e}")
    finally:
        stop_event.set()
        audio_queue.put(None)  # Signal end of stream
        control_task.cancel()
        if transcription_thread:
            transcription_thread.join(timeout=5.0)
        
        if raw_received_chunks:
            raw_all = np.concatenate(raw_received_chunks, axis=0)  # shape: (frames, channels)
            # int16 = (raw_all / (np.max(np.abs(raw_all)) + 1e-9) * 32767).clip(-32768, 32767).astype("<i2")
            write_wav("audio/raw_received_from_client.wav", 16000, raw_all)
            print("💾 Saved raw client audio to audio/raw_received_from_client.wav")
            
        if sent_to_soniox_chunks:
            full_audio = np.concatenate(sent_to_soniox_chunks)
            write_wav("audio/sent_to_soniox_actual_input.wav", 16000, full_audio)
            print("💾 Saved enhanced audio sent to Soniox → audio/sent_to_soniox_actual_input.wav")
        
        finalize()

# === iOS App Clients ===
async def broadcast_to_grpc_clients(message: dict):
    msg = None
    try:
        # Convert dict to protobuf ServerMessage
        if message["type"] == "transcript":
            msg = transcribeglass_pb2.ServerMessage(
                transcript=transcribeglass_pb2.Transcript(source_id=message["source_id"], text=message["text"])
            )
        elif message["type"] == "sources":
            msg = transcribeglass_pb2.ServerMessage(
                sources=transcribeglass_pb2.SourceUpdate(
                    sources=[
                        transcribeglass_pb2.Source(
                            id=src["id"],
                            x=src["x"],
                            y=src["y"],
                            activity=src["activity"],
                            angle=src["angle"]
                        ) for src in message["sources"]
                    ]
                )
            )
    except Exception as e:
        print(f"⚠️ Failed to convert message to proto: {e}")
        return

    if msg is None:
        return

    dead_clients = set()
    for client in grpc_ui_clients:
        try:
            await client.queue.put(msg)
        except Exception as e:
            print(f"⚠️ Failed to send to client: {e}")
            dead_clients.add(client)

    grpc_ui_clients.difference_update(dead_clients)

ui_clients = set()

async def broadcast_to_ui_clients(message: dict):
    data = json.dumps(message, ensure_ascii=False) + "\n"  # guarantees UTF-8 valid + newline delimiter
    to_remove = set()
    # print(f"📤 Sending to {len(ui_clients)} UI clients: {message['type']}")

    async def try_send(client):
        try:
            print(f"📤 Sending to UI client: {repr(data)}")
            await asyncio.wait_for(client.send(data), timeout=1.0)
            await asyncio.sleep(0.01)  # ensures the frame is flushed
            # print(f"✅ Sent to UI client")
        except Exception as e:
            print(f"⚠️ Failed to send to UI client: {e}")
            to_remove.add(client)

    await asyncio.gather(*(try_send(client) for client in ui_clients.copy()))

    for client in to_remove:
        ui_clients.remove(client)

async def ui_handler(websocket):
    global active_custom_id
    print("🖥 UI client connected")
    ui_clients.add(websocket)
    try:
        async for message in websocket:
            print(f"📨 UI message received: {message}")
            try:
                data = json.loads(message)
                if data.get("type") == "select_source":
                    selected_custom_id = int(data.get("source_id"))
                    with sources_lock:
                        # Just set the active_custom_id directly
                        active_custom_id = selected_custom_id
                        print(f"🎯 Selected custom ID {selected_custom_id} as active")
            except Exception as e:
                print(f"⚠️ Error processing UI command: {e}")
    except websockets.exceptions.ConnectionClosed:
        print("🖥 UI client disconnected")
    finally:
        ui_clients.remove(websocket)




# ========== MAIN ========== #
connected_handlers = set()

async def main():
    print("🛰 Starting WebSocket server on ws://0.0.0.0:8765")
    print("🎯 Starting ODAS SST listener on TCP port 9000")
    print("🚀 Starting gRPC server on port 50051")

    # gRPC server
    grpc_server = serve_grpc()
    await grpc_server.start()

    # ODAS TCP listener
    odas_listener_task = asyncio.create_task(start_odas_listener())

    # WebSocket audio stream
    async def handler_wrapper(websocket):
        task = asyncio.current_task()
        connected_handlers.add(task)
        try:
            await handler(websocket)
        finally:
            connected_handlers.remove(task)

    websocket_server = await websockets.serve(
            handler_wrapper,
            "0.0.0.0",
            8765,
            ping_interval=10,         # send ping every 10s
            ping_timeout=30            # if no pong in 5s, disconnect
        )

    # (OPTIONAL) Legacy UI WebSocket server (if still used)
    # ui_server = await websockets.serve(ui_handler, "0.0.0.0", 8888)

    try:
        await asyncio.Future()  # run forever
    except asyncio.CancelledError:
        print("🛑 Server shutdown initiated. Cancelling tasks...")

        # Cancel ODAS + WebSocket clients
        odas_listener_task.cancel()
        for task in connected_handlers:
            task.cancel()

        await asyncio.gather(*connected_handlers, return_exceptions=True)
        await grpc_server.stop(grace=None)  # graceful shutdown

        print("✅ Clean shutdown complete.")

def run():
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.ensure_future(shutdown(loop)))
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()

async def shutdown(loop):
    print("🛑 Caught shutdown signal. Cancelling tasks...")
    tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()

if __name__ == "__main__":
    set_api_key("cf8883cd94a4037c3b7829945335c03d0922928984c5d5c219afb1ee168b6643")
    run()