import pygame
import random
import time
import os
import pandas as pd
from datetime import datetime
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# -------------------- Experiment Parameters --------------------
num_trials = 27  # Number of repetitions per target word
stimulus_duration = 0.2
pause_duration = 0.19
rest_duration = 10
target_words = ['맞아', '고마워', '아니', '도와줘']

# -------------------- Initialize BrainFlow (Connect Muse 2) --------------------
BoardShim.enable_dev_board_logger()
params = BrainFlowInputParams()
params.serial_number = "Muse-1E9A"  # Serial number of the Muse 2 device
board_id = BoardIds.MUSE_2_BOARD.value
board = BoardShim(board_id, params)
print("▶ Connecting to Muse 2...")
board.prepare_session()
board.start_stream()
sampling_rate = BoardShim.get_sampling_rate(board_id)


# -------------------- Initialize Pygame --------------------
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

positions = []
for row in range(2):
    for col in range(2):
        x = start_x + col * (cell_size + margin)
        y = start_y + row * (cell_size + margin)
        positions.append((x, y))

stimulus_log = []
eeg_data_log = []

def show_target_guide(target_word, duration=2):
    screen.fill((0, 0, 0))

    try:
        idx = target_words.index(target_word)
    except ValueError:
        return

    x, y = positions[idx]

    pygame.draw.rect(screen, (0, 0, 255), (x, y, cell_size, cell_size), 5)

    text = font.render(target_word, True, (255, 255, 0))
    text_rect = text.get_rect(center=(x + cell_size // 2, y + cell_size // 2))
    screen.blit(text, text_rect)

    pygame.display.flip()
    time.sleep(duration)

def draw_matrix_with_highlight(highlight_index=None):
    screen.fill((0, 0, 0))
    for idx, word in enumerate(target_words):
        x, y = positions[idx]

        if idx == highlight_index:
            color = (250, 250, 0)
        else:
            color = (100, 100, 100)

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

# -------------------- Set Backup Folder --------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_folder = os.path.join("backup_logs", timestamp)
os.makedirs(backup_folder, exist_ok=True)

try:
    for target_word in random.sample(target_words, len(target_words)):
        target_index = target_words.index(target_word)

        show_target_guide(target_word, duration=2)

        for trial in range(num_trials):
            stim_order = list(range(len(target_words)))
            random.shuffle(stim_order)

            for stim_index in stim_order:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt

                blink_stimulus(target_words[stim_index], duration=stimulus_duration)

                data = board.get_current_board_data(int(sampling_rate * (stimulus_duration + pause_duration)))
                timestamps = data[BoardShim.get_timestamp_channel(board_id)]
                eeg_channels = BoardShim.get_eeg_channels(board_id)
                eeg_data = data[eeg_channels, :]

                for i in range(eeg_data.shape[1]):
                    eeg_data_log.append({
                        "target_word": target_word,
                        "timestamp": timestamps[i],
                        "stimulus_type": 1 if stim_index == target_index else 0,
                        "TP9": eeg_data[0, i],
                        "AF7": eeg_data[1, i],
                        "AF8": eeg_data[2, i],
                        "TP10": eeg_data[3, i],
                    })

                stimulus_log.append({
                    "target_word": target_word,
                    "time": time.time(),
                    "stimulus_type": 1 if stim_index == target_index else 0,
                    "stimulus_index": stim_index,
                })

except KeyboardInterrupt:
    print("▶ 사용자에 의해 종료됨")

finally:
    board.stop_stream()
    board.release_session()
    pygame.quit()

    stim_df = pd.DataFrame(stimulus_log)
    eeg_df = pd.DataFrame(eeg_data_log)

    stim_path = os.path.join(backup_folder, f"stimulus_log_{timestamp}.csv")
    eeg_path = os.path.join(backup_folder, f"eeg_data_log_{timestamp}.csv")
    stim_df.to_csv(stim_path, index=False)
    eeg_df.to_csv(eeg_path, index=False)

    print(f" 저장 완료: {stim_path} / {eeg_path}")