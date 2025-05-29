import time
import random
import pygame
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

def focal_loss_fixed(y_true, y_pred):
    gamma = 1.0
    alpha = 0.25
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    cross_entropy = -y_true * K.log(y_pred)
    weight = alpha * K.pow(1 - y_pred, gamma)
    loss = weight * cross_entropy
    return K.mean(K.sum(loss, axis=1))

stimulus_duration = 0.2
pause_duration = 0.2
rest_duration = 10
target_words = ['맞아', '고마워', '아니', '도와줘']

BoardShim.enable_dev_board_logger()
params = BrainFlowInputParams()
params.serial_number = "Muse-1E9A"
board_id = BoardIds.MUSE_2_BOARD.value
board = BoardShim(board_id, params)
print("▶ Connecting to Muse 2...")
board.prepare_session()
board.start_stream()
sampling_rate = BoardShim.get_sampling_rate(board_id)

eeg_channels = [1, 4]

pygame.init()
infoObject = pygame.display.Info()
screen_width = max(800, infoObject.current_w)
screen_height = max(600, infoObject.current_h)
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("P300 Speller 2x2 Matrix")
font = pygame.font.Font("NanumGothic.ttf", 60)
clock = pygame.time.Clock()

cell_size = 150
margin = 170
grid_width = 2 * cell_size + margin
grid_height = 2 * cell_size + margin
start_x = (screen_width - grid_width) // 2
start_y = (screen_height - grid_height) // 2
positions = [(start_x + col * (cell_size + margin), start_y + row * (cell_size + margin)) for row in range(2) for col in range(2)]

def draw_matrix_with_highlight(highlight_index=None):
    screen.fill((0, 0, 0))
    for idx, word in enumerate(target_words):
        x, y = positions[idx]
        color = (255, 255, 0) if idx == highlight_index else (100, 100, 100)
        text = font.render(word, True, color)
        text_rect = text.get_rect(center=(x + cell_size // 2, y + cell_size // 2))
        screen.blit(text, text_rect)
    pygame.display.flip()

def blink_stimulus(word, duration=0.2):
    idx = target_words.index(word)
    draw_matrix_with_highlight(highlight_index=idx)
    time.sleep(duration)
    draw_matrix_with_highlight(highlight_index=None)
    time.sleep(pause_duration)

model_path = "eegnet_model.h5"
model = load_model(model_path, custom_objects={'focal_loss_fixed': focal_loss_fixed})
print(f"Model '{model_path}' loaded")

def preprocess_eeg(eeg_data, fs=256, window_sec=0.6):
    from scipy.signal import butter, lfilter

    def butter_bandpass(lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        return butter(order, [low, high], btype='band')

    def bandpass_filter(data, lowcut=1, highcut=30, fs=256):
        b, a = butter_bandpass(lowcut, highcut, fs)
        return lfilter(b, a, data)

    def baseline_correction(data, fs, baseline_sec=0.1):
        baseline_samples = int(baseline_sec * fs)
        baseline_mean = np.mean(data[:, :baseline_samples], axis=1, keepdims=True)
        return data - baseline_mean

    data_bc = baseline_correction(eeg_data, fs)
    data_filtered = np.array([bandpass_filter(ch, 1, 30, fs) for ch in data_bc])
    data_final = data_filtered[:, :int(window_sec * fs)]
    return data_final[np.newaxis, ..., np.newaxis]

num_blinks_per_word = 15
eeg_data_by_word = {word: [] for word in target_words}

def draw_matrix_static_gray():
    for idx, word in enumerate(target_words):
        x, y = positions[idx]
        color = (100, 100, 100)  # 회색
        text = font.render(word, True, color)
        text_rect = text.get_rect(center=(x + cell_size // 2, y + cell_size // 2))
        screen.blit(text, text_rect)

try:
    screen.fill((0, 0, 0))
    instruction_text = font.render("▶ 원하는 단어를 응시해주세요", True, (255, 255, 255))
    instruction_rect = instruction_text.get_rect(center=(screen_width // 2, start_y - 100))
    screen.blit(instruction_text, instruction_rect)
    draw_matrix_static_gray()
    pygame.display.flip()
    time.sleep(5)

    blink_counts = {word: 0 for word in target_words}

    while any(count < num_blinks_per_word for count in blink_counts.values()):
        stim_order = list(range(len(target_words)))
        random.shuffle(stim_order)

        for stim_index in stim_order:
            word = target_words[stim_index]
            if blink_counts[word] >= num_blinks_per_word:
                continue

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

            blink_stimulus(word, duration=stimulus_duration)

            window_duration = 0.6
            data = board.get_current_board_data(int(sampling_rate * window_duration))
            eeg_data = data[eeg_channels, :]
            features = preprocess_eeg(eeg_data, fs=sampling_rate, window_sec=window_duration)
            eeg_data_by_word[word].append(features)

            blink_counts[word] += 1
            print(f"'{word}' 깜빡임 {blink_counts[word]}/{num_blinks_per_word}회 완료")

            if all(count >= num_blinks_per_word for count in blink_counts.values()):
                break

    word_scores = {}
    for word, eeg_trials in eeg_data_by_word.items():
        preds = []
        for eeg_input in eeg_trials:
            pred = model.predict(eeg_input)[0]
            proba = pred[1]
            preds.append(proba)
        mean_proba = np.mean(preds)
        word_scores[word] = mean_proba

    predicted_word = max(word_scores, key=word_scores.get)
    print("▶ 예측된 단어:", predicted_word)

    font_result = pygame.font.Font("NanumGothic.ttf", 150)

    # --- Display Result ---
    screen.fill((0, 0, 0))
    result_text = font_result.render(f"{predicted_word}", True, (0, 0, 255))
    result_rect = result_text.get_rect(center=(screen_width // 2, screen_height // 2))
    screen.blit(result_text, result_rect)
    pygame.display.flip()
    time.sleep(5)

except KeyboardInterrupt:
    print("▶ 실험 중단됨")

finally:
    board.stop_stream()
    board.release_session()
    pygame.quit()