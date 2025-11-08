import sys
import math
import csv
import re
import random
from functools import reduce

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QComboBox, QLineEdit, QLabel, QFileDialog,
    QMessageBox, QGridLayout, QStatusBar, QTableWidget, QTableWidgetItem,
    QDialog, QTabWidget, QScrollArea, QHeaderView
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

# ================== КОНФИГУРАЦИЯ АЛФАВИТА ==================
# Только маленькие русские буквы + выбранные знаки препинания + пробел.
# Можно редактировать по необходимости (например, убрать многоточие или добавить слэш).
LOWERCASE_CYRILLIC = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"

# Набор знаков препинания, характерный для русскоязычного текста:
# Точка, запятая, двоеточие, точка с запятой, восклицательный, вопросительный,
# тире (короткий дефис), многоточие, кавычки «», обычные кавычки, скобки, круглые, длинное тире
PUNCTUATION_RU = ".,:;!?-…«»\"'()—"
# Пробел отдельно
SPACE = " "

# Итоговый алфавит формируем функцией, чтобы легко переопределять логику.
def build_alphabet() -> str:
    # Удалим дубликаты, сохраним порядок.
    seen = set()
    ordered = []
    for ch in LOWERCASE_CYRILLIC + PUNCTUATION_RU + SPACE:
        if ch not in seen:
            seen.add(ch)
            ordered.append(ch)
    return "".join(ordered)

ALPHABET = build_alphabet()
MOD = len(ALPHABET)

# Флаг строгой фильтрации: если True — все символы вне алфавита удаляются.
STRICT_MODE = True

# ================== ЭТАЛОННАЯ ЧАСТОТНОСТЬ ==================
STANDARD_RUSSIAN_FREQ = {
    ' ': 0.175, 'о': 0.090, 'е': 0.072, 'а': 0.062, 'и': 0.062, 'н': 0.053, 'т': 0.053,
    'с': 0.045, 'р': 0.040, 'в': 0.038, 'л': 0.035, 'к': 0.028, 'м': 0.026, 'д': 0.025,
    'п': 0.023, 'у': 0.021, 'я': 0.018, 'ы': 0.016, 'з': 0.016, 'ь': 0.014, 'ъ': 0.014,
    'б': 0.014, 'г': 0.013, 'ч': 0.012, 'й': 0.010, 'х': 0.009, 'ж': 0.007, 'ю': 0.006,
    'ш': 0.006, 'ц': 0.004, 'щ': 0.003, 'э': 0.003, 'ф': 0.002, 'ё': 0.002
}

# Частые слова для дополнительного скоринга (криптоанализ)
COMMON_WORDS = [
    "и","в","не","на","что","он","как","но","по","это","я","к","у","ты","мы","они",
    "его","еще","если","когда","то","со","из","ли","так","для","ведь","же","под",
    "там","чтобы","при","или","все","нет","да","мне","мой","она","есть","где"
]

# ================== ПРЕДОБРАБОТКА ==================
def preprocess_text(text: str) -> str:
    """
    1. lower()
    2. Нормализация различных тире к '-' если '-' присутствует в алфавите.
       Символы '–' (en-dash) и '—' (em-dash) заменяются на '—' или '-' в зависимости от алфавита.
    3. Нормализация кавычек: “ ” -> ", ‘ ’ -> ', если такие кавычки есть.
       Можно оставить как есть, если они не в алфавите — будут удалены при STRICT_MODE.
    4. Любые последовательности пробельных символов -> одиночный пробел.
    5. Удаление всего, что не входит в ALPHABET (если STRICT_MODE).
    """
    text = text.lower()

    # Нормализация тире: приведем короткое и длинное тире к одному символу если он есть.
    if "—" in ALPHABET and "-" in ALPHABET:
        # Оставляем оба, только заменим '–' (en-dash) на '—'
        text = text.replace("–", "—")
    elif "-" in ALPHABET:
        # Все длинные тире приводим к короткому дефису
        text = text.replace("–", "-").replace("—", "-")
    elif "—" in ALPHABET:
        text = text.replace("–", "—").replace("-", "—")

    # Нормализация кавычек (если стандартные двойные кавычки присутствуют)
    if "\"" in ALPHABET:
        text = text.replace("“", "\"").replace("”", "\"")
    if "'" in ALPHABET:
        text = text.replace("‘", "'").replace("’", "'")

    # Множественные пробелы / любые whitespace -> один пробел
    text = re.sub(r'\s+', ' ', text)

    if STRICT_MODE:
        text = "".join(ch for ch in text if ch in ALPHABET)
    else:
        # Если не strict — можно оставить прочие символы, но обычно для шифров лучше строгий режим
        pass

    return text.strip()

# ================== ШИФРЫ ==================
def caesar_cipher(text: str, key: int, encrypt=True) -> str:
    res = []
    shift = key if encrypt else -key
    for ch in text:
        idx = ALPHABET.find(ch)
        if idx != -1:
            res.append(ALPHABET[(idx + shift) % MOD])
        else:
            # В идеале сюда не попадем после строгой предобработки
            res.append(ch)
    return "".join(res)

def extended_gcd(a: int, b: int):
    if a == 0:
        return b, 0, 1
    g, y, x = extended_gcd(b % a, a)
    return g, x - (b // a) * y, y

def modinv(a: int, m: int) -> int:
    g, x, _ = extended_gcd(a, m)
    if g != 1:
        raise ValueError(f"Обратного элемента для {a} по модулю {m} не существует.")
    return x % m

def affine_cipher(text: str, key_a: int, key_b: int, encrypt=True) -> str:
    if math.gcd(key_a, MOD) != 1:
        raise ValueError(f"Ключ 'a' ({key_a}) должен быть взаимно прост с {MOD}.")
    res = []
    for ch in text:
        idx = ALPHABET.find(ch)
        if idx != -1:
            if encrypt:
                y = (key_a * idx + key_b) % MOD
            else:
                a_inv = modinv(key_a, MOD)
                y = (a_inv * (idx - key_b)) % MOD
            res.append(ALPHABET[y])
        else:
            res.append(ch)
    return "".join(res)

def vigenere_cipher(text: str, key: str, encrypt=True) -> str:
    if not key:
        raise ValueError("Ключ для шифра Виженера не может быть пустым.")
    key = preprocess_text(key)
    if not key:
        raise ValueError("Ключ после предобработки пуст или не содержит допустимых символов.")
    res = []
    ki = 0
    for ch in text:
        oi = ALPHABET.find(ch)
        if oi != -1:
            shift = ALPHABET.find(key[ki % len(key)])
            if not encrypt:
                shift = -shift
            res.append(ALPHABET[(oi + shift) % MOD])
            ki += 1
        else:
            res.append(ch)
    return "".join(res)

def gronsfeld_cipher(text: str, key: str, encrypt=True) -> str:
    if not key.isdigit():
        raise ValueError("Ключ для шифра Гронсфельда должен состоять только из цифр.")
    res = []
    ki = 0
    for ch in text:
        oi = ALPHABET.find(ch)
        if oi != -1:
            shift = int(key[ki % len(key)])
            if not encrypt:
                shift = -shift
            res.append(ALPHABET[(oi + shift) % MOD])
            ki += 1
        else:
            res.append(ch)
    return "".join(res)

# ================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ КРИПТОАНАЛИЗА ==================
def compute_letter_frequency(text: str):
    freq = {}
    total = 0
    for ch in text:
        if ch in ALPHABET:
            freq[ch] = freq.get(ch, 0) + 1
            total += 1
    if total == 0:
        return {}
    for ch in freq:
        freq[ch] /= total
    return freq

def chi_square_stat(freq_decoded: dict, reference: dict) -> float:
    stat = 0.0
    for ch, ref_p in reference.items():
        obs = freq_decoded.get(ch, 0.0)
        stat += (obs - ref_p) ** 2 / (ref_p + 1e-9)
    return stat

def word_match_score(text: str, words) -> int:
    tokens = re.split(r'[^а-яё]+', text.lower())
    tokens = [t for t in tokens if t]
    set_tokens = set(tokens)
    score = 0
    for w in words:
        if w in set_tokens:
            score += 1
    return score

def score_decryption(text: str, reference_freq: dict) -> float:
    lf = compute_letter_frequency(text)
    chi = chi_square_stat(lf, reference_freq)
    wscore = word_match_score(text, COMMON_WORDS)
    return -chi + wscore * 0.05

def apply_map(text: str, mapping: dict) -> str:
    return "".join(mapping.get(ch, ch) for ch in text)

def refine_substitution_map(encrypted_text: str,
                            initial_map: dict,
                            reference_freq: dict,
                            iterations: int = 1500,
                            stagnation_limit: int = 300) -> dict:
    if not encrypted_text or not initial_map:
        return initial_map

    working_map = initial_map.copy()
    best_map = working_map.copy()
    decoded_best = apply_map(encrypted_text, best_map)
    best_score = score_decryption(decoded_best, reference_freq)

    enc_chars = list(working_map.keys())
    stagnation = 0

    for _ in range(iterations):
        if stagnation > stagnation_limit:
            break

        c1, c2 = random.sample(enc_chars, 2)
        old1, old2 = working_map[c1], working_map[c2]
        working_map[c1], working_map[c2] = old2, old1

        decoded_try = apply_map(encrypted_text, working_map)
        try_score = score_decryption(decoded_try, reference_freq)

        if try_score > best_score:
            best_score = try_score
            best_map = working_map.copy()
            stagnation = 0
        else:
            working_map[c1], working_map[c2] = old1, old2
            stagnation += 1

    return best_map

# ================== ДИАЛОГ ТАБЛИЦЫ ЧАСТОТ ==================
class FrequencyTableDialog(QDialog):
    def __init__(self, parent=None, frequency_data=None):
        super().__init__(parent)
        self.frequency_data = frequency_data or []
        self.setWindowTitle("Таблица частотности символов")
        self.setGeometry(150, 150, 400, 600)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        self.table_widget = QTableWidget()
        self.table_widget.setRowCount(len(self.frequency_data))
        self.table_widget.setColumnCount(2)
        self.table_widget.setHorizontalHeaderLabels(["Символ", "Вероятность"])
        self.table_widget.setColumnWidth(0, 150)
        self.table_widget.setColumnWidth(1, 200)

        for row, (char, prob) in enumerate(self.frequency_data):
            self.table_widget.setItem(row, 0, QTableWidgetItem(char))
            self.table_widget.setItem(row, 1, QTableWidgetItem(f"{prob:.4f}"))

        self.table_widget.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table_widget)

        button_layout = QHBoxLayout()
        self.save_button = QPushButton("Сохранить таблицу")
        self.save_button.clicked.connect(self.save_table)
        self.close_button = QPushButton("Закрыть")
        self.close_button.clicked.connect(self.accept)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.close_button)
        layout.addLayout(button_layout)

    def save_table(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить таблицу частот", "frequency_table.csv", "CSV files (*.csv)"
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8-sig', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Символ", "Вероятность"])
                    for row in range(self.table_widget.rowCount()):
                        symbol = self.table_widget.item(row, 0).text()
                        probability = self.table_widget.item(row, 1).text()
                        writer.writerow([symbol, probability])
                QMessageBox.information(self, "Успех", f"Таблица успешно сохранена в {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл: {e}")

# ================== ВКЛАДКА ШИФРОВАНИЯ ==================
class EncryptionTab(QWidget):
    def __init__(self):
        super().__init__()
        self.current_state = 'initial'
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)

        top_layout = QHBoxLayout()
        self.open_button = QPushButton("Открыть .txt файл")
        self.open_button.clicked.connect(self.open_file)
        self.save_button = QPushButton("Сохранить результат")
        self.save_button.clicked.connect(self.save_file)
        top_layout.addWidget(self.open_button)
        top_layout.addWidget(self.save_button)
        top_layout.addStretch(1)
        main_layout.addLayout(top_layout)

        text_layout = QHBoxLayout()
        original_group = QVBoxLayout()
        original_group.addWidget(QLabel("Исходный текст:"))
        self.original_text_edit = QTextEdit()
        self.original_text_edit.setPlaceholderText("Введите или вставьте сюда текст...")
        self.original_text_edit.textChanged.connect(self.reset_state)
        original_group.addWidget(self.original_text_edit)
        text_layout.addLayout(original_group)

        result_group = QVBoxLayout()
        result_group.addWidget(QLabel("Результат:"))
        self.result_text_edit = QTextEdit()
        self.result_text_edit.setReadOnly(True)
        result_group.addWidget(self.result_text_edit)
        text_layout.addLayout(result_group)
        main_layout.addLayout(text_layout)

        control_layout = QGridLayout()
        cipher_label = QLabel("1. Выбрать шифр:")
        self.cipher_combo = QComboBox()
        self.cipher_combo.addItems(["Цезарь", "Аффинный", "Виженер", "Гронсфельд"])

        key_label = QLabel("2. Ввести ключ:")
        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText("Например: 3 или 5,2 или слово или цифры")

        self.encrypt_button = QPushButton("3. Зашифровать")
        self.encrypt_button.clicked.connect(self.encrypt_text)
        self.decrypt_button = QPushButton("4. Расшифровать")
        self.decrypt_button.clicked.connect(self.decrypt_text)

        freq_button_layout = QGridLayout()
        self.freq_original_button = QPushButton("Частота исходного")
        self.freq_original_button.clicked.connect(self.show_frequency_table_original)
        self.freq_preprocessed_button = QPushButton("Частота предобработанного")
        self.freq_preprocessed_button.clicked.connect(self.show_frequency_table_preprocessed)
        self.freq_result_button = QPushButton("Частота результата")
        self.freq_result_button.clicked.connect(self.show_frequency_table_result)

        freq_button_layout.addWidget(self.freq_original_button, 0, 0)
        freq_button_layout.addWidget(self.freq_preprocessed_button, 0, 1)
        freq_button_layout.addWidget(self.freq_result_button, 1, 0, 1, 2)

        control_layout.addWidget(cipher_label, 0, 0)
        control_layout.addWidget(self.cipher_combo, 0, 1)
        control_layout.addWidget(key_label, 1, 0)
        control_layout.addWidget(self.key_input, 1, 1)
        control_layout.addWidget(self.encrypt_button, 2, 0)
        control_layout.addWidget(self.decrypt_button, 2, 1)
        control_layout.addLayout(freq_button_layout, 3, 0, 1, 2)
        main_layout.addLayout(control_layout)

        self.status_bar = QStatusBar()
        main_layout.addWidget(self.status_bar)
        self.status_bar.showMessage("Готов к работе. Введите текст или откройте файл.")

    def reset_state(self):
        self.current_state = 'initial'
        self.result_text_edit.clear()
        self.status_bar.showMessage("Исходный текст изменен. Готов к новой операции.")

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Открыть текстовый файл", "", "Текстовые файлы (*.txt)"
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.original_text_edit.setText(content)
                self.status_bar.showMessage("Файл успешно открыт.")
                self.reset_state()
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось прочитать файл: {e}")

    def save_file(self):
        txt = self.result_text_edit.toPlainText()
        if not txt:
            QMessageBox.warning(self, "Внимание", "Нет текста для сохранения.")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить результат", "", "Текстовые файлы (*.txt)"
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(txt)
                self.status_bar.showMessage("Результат сохранен.")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл: {e}")

    def get_key(self):
        k = self.key_input.text().strip()
        if not k:
            raise ValueError("Ключ не может быть пустым.")
        return k

    def encrypt_text(self):
        try:
            if self.current_state not in ['initial', 'preprocessed', 'decrypted']:
                QMessageBox.warning(self, "Неверная операция", "Текст уже зашифрован.")
                return
            key_str = self.get_key()
            raw = self.original_text_edit.toPlainText()
            if not raw:
                QMessageBox.warning(self, "Внимание", "Нет исходного текста.")
                return
            processed = preprocess_text(raw)
            name = self.cipher_combo.currentText()
            if name == "Цезарь":
                encrypted = caesar_cipher(processed, int(key_str), True)
            elif name == "Аффинный":
                try:
                    a, b = map(int, key_str.split(','))
                except:
                    raise ValueError("Для аффинного: два числа через запятую (напр. 5,2)")
                encrypted = affine_cipher(processed, a, b, True)
            elif name == "Виженер":
                encrypted = vigenere_cipher(processed, key_str, True)
            elif name == "Гронсфельд":
                encrypted = gronsfeld_cipher(processed, key_str, True)
            else:
                encrypted = processed
            self.result_text_edit.setText(encrypted)
            self.current_state = 'encrypted'
            self.status_bar.showMessage(f"Зашифровано ({name}).")
        except ValueError as e:
            QMessageBox.critical(self, "Ошибка", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Ошибка шифрования", f"Неожиданная ошибка: {e}")

    def decrypt_text(self):
        try:
            if self.current_state != 'encrypted':
                QMessageBox.warning(self, "Неверная операция", "Нужно сначала зашифровать.")
                return
            key_str = self.get_key()
            txt = self.result_text_edit.toPlainText()
            if not txt:
                QMessageBox.warning(self, "Внимание", "Нет текста.")
                return
            name = self.cipher_combo.currentText()
            if name == "Цезарь":
                decrypted = caesar_cipher(txt, int(key_str), False)
            elif name == "Аффинный":
                try:
                    a, b = map(int, key_str.split(','))
                except:
                    raise ValueError("Для аффинного: два числа через запятую.")
                decrypted = affine_cipher(txt, a, b, False)
            elif name == "Виженер":
                decrypted = vigenere_cipher(txt, key_str, False)
            elif name == "Гронсфельд":
                decrypted = gronsfeld_cipher(txt, key_str, False)
            else:
                decrypted = txt
            self.result_text_edit.setText(decrypted)
            self.current_state = 'decrypted'
            self.status_bar.showMessage("Расшифровано.")
        except ValueError as e:
            QMessageBox.critical(self, "Ошибка", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Ошибка расшифровки", f"Неожиданная ошибка: {e}")

    def _calculate_frequencies(self, text: str):
        if not text: return []
        counts = {}
        for ch in text: counts[ch] = counts.get(ch, 0) + 1
        total = len(text)
        return sorted([(c, counts[c]/total) for c in counts], key=lambda x: x[1], reverse=True)

    def show_frequency_table_original(self):
        t = self.original_text_edit.toPlainText()
        if not t:
            QMessageBox.warning(self, "Внимание", "Исходный текст пуст.")
            return
        freq = self._calculate_frequencies(t)
        FrequencyTableDialog(self, freq).exec()

    def show_frequency_table_preprocessed(self):
        raw = self.original_text_edit.toPlainText()
        if not raw:
            QMessageBox.warning(self, "Внимание", "Исходный текст пуст.")
            return
        prep = preprocess_text(raw)
        if not prep:
            QMessageBox.warning(self, "Внимание", "После предобработки пусто.")
            return
        freq = self._calculate_frequencies(prep)
        FrequencyTableDialog(self, freq).exec()

    def show_frequency_table_result(self):
        t = self.result_text_edit.toPlainText()
        if not t:
            QMessageBox.warning(self, "Внимание", "Нет текста.")
            return
        freq = self._calculate_frequencies(t)
        FrequencyTableDialog(self, freq).exec()

# ================== ВКЛАДКА КРИПТОАНАЛИЗА (без изменений, кроме использования нового алфавита) ==================
class CryptanalysisTab(QWidget):
    def __init__(self):
        super().__init__()
        self.substitution_map = {}
        self.reference_frequencies = {}
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)

        top_layout = QHBoxLayout()
        self.open_button = QPushButton("Открыть .txt файл")
        self.open_button.clicked.connect(self.load_encrypted_file)
        self.save_button = QPushButton("Сохранить результат")
        self.save_button.clicked.connect(self.save_decrypted_result)
        self.import_freq_button = QPushButton("Импорт таблица частот")
        self.import_freq_button.clicked.connect(self.import_reference_frequency)
        top_layout.addWidget(self.open_button)
        top_layout.addWidget(self.save_button)
        top_layout.addWidget(self.import_freq_button)
        top_layout.addStretch(1)
        main_layout.addLayout(top_layout)

        text_layout = QHBoxLayout()
        encrypted_group = QVBoxLayout()
        encrypted_group.addWidget(QLabel("Зашифрованный текст:"))
        self.encrypted_input = QTextEdit()
        self.encrypted_input.setPlaceholderText("Вставьте зашифрованный текст...")
        self.encrypted_input.textChanged.connect(self.on_encrypted_text_changed)
        encrypted_group.addWidget(self.encrypted_input)
        text_layout.addLayout(encrypted_group)

        result_group = QVBoxLayout()
        result_group.addWidget(QLabel("Результат расшифровки:"))
        self.decrypted_output = QTextEdit()
        self.decrypted_output.setReadOnly(True)
        self.decrypted_output.setPlaceholderText("Расшифровка появится здесь...")
        result_group.addWidget(self.decrypted_output)
        text_layout.addLayout(result_group)
        main_layout.addLayout(text_layout)

        control_layout = QGridLayout()
        self.auto_button = QPushButton("1. Автоанализ (частоты)")
        self.auto_button.clicked.connect(self.auto_substitute)

        self.refine_button = QPushButton("2. Уточнить замену")
        self.refine_button.clicked.connect(self.refine_mapping)

        self.show_map_button = QPushButton("3. Редактировать вручную")
        self.show_map_button.clicked.connect(self.show_substitution_dialog)

        freq_button_layout = QGridLayout()
        self.freq_encrypted_button = QPushButton("Частота зашифрованного")
        self.freq_encrypted_button.clicked.connect(self.show_encrypted_frequency)
        self.freq_reference_button = QPushButton("Таблица частот")
        self.freq_reference_button.clicked.connect(self.show_reference_frequency)
        self.save_map_button = QPushButton("Сохранить карту")
        self.save_map_button.clicked.connect(self.save_substitution_map)

        freq_button_layout.addWidget(self.freq_encrypted_button, 0, 0)
        freq_button_layout.addWidget(self.freq_reference_button, 0, 1)
        freq_button_layout.addWidget(self.save_map_button, 1, 0, 1, 2)

        control_layout.addWidget(self.auto_button, 0, 0)
        control_layout.addWidget(self.refine_button, 0, 1)
        control_layout.addWidget(self.show_map_button, 0, 2)
        control_layout.addLayout(freq_button_layout, 1, 0, 1, 3)
        main_layout.addLayout(control_layout)

        self.status_bar = QStatusBar()
        main_layout.addWidget(self.status_bar)
        self.status_bar.showMessage("Готов к анализу. Введите текст или откройте файл.")

        self.load_standard_reference_table()

    def import_reference_frequency(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Импорт таблицы частот", "", "CSV files (*.csv);;All files (*.*)"
        )
        if not file_path:
            return
        try:
            with open(file_path, 'r', encoding='utf-8-sig', newline='') as f:
                sample = f.read(4096)
                f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=";, \t")
                except csv.Error:
                    dialect = csv.excel
                reader = csv.reader(f, dialect)
                first = next(reader, None)
                def try_float(s):
                    try:
                        return float(s.strip().replace(',', '.'))
                    except:
                        return None
                new_freq = {}
                if first and (try_float(first[1]) is None):
                    pass
                else:
                    if first and len(first) >= 2:
                        ch_raw = first[0]
                        p = try_float(first[1])
                        if p is not None:
                            if (len(ch_raw) >= 2 and ((ch_raw[0] == "'" and ch_raw[-1] == "'") or (ch_raw[0] == '"' and ch_raw[-1] == '"'))):
                                ch_raw = ch_raw[1:-1]
                            new_freq[ch_raw] = p
                for row in reader:
                    if not row or len(row) < 2:
                        continue
                    ch_raw = row[0]
                    p = try_float(row[1])
                    if p is None:
                        continue
                    if (len(ch_raw) >= 2 and ((ch_raw[0] == "'" and ch_raw[-1] == "'") or (ch_raw[0] == '"' and ch_raw[-1] == '"'))):
                        ch_raw = ch_raw[1:-1]
                    if ch_raw == "":
                        continue
                    new_freq[ch_raw] = p
            if new_freq:
                self.reference_frequencies = new_freq
                QMessageBox.information(self, "Успех", f"Импортировано {len(new_freq)} символов.")
                self.status_bar.showMessage(f"Импортировано {len(new_freq)} символов эталона")
            else:
                QMessageBox.warning(self, "Внимание", "Не удалось прочитать данные.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось импортировать: {e}")

    def load_encrypted_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Открыть зашифрованный текст", "", "Текстовые файлы (*.txt)"
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.encrypted_input.setText(f.read())
                self.status_bar.showMessage("Файл загружен.")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось прочитать файл: {e}")

    def save_decrypted_result(self):
        res = self.decrypted_output.toPlainText()
        if not res:
            QMessageBox.warning(self, "Внимание", "Нет текста для сохранения.")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить результат", "", "Текстовые файлы (*.txt)"
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(res)
                self.status_bar.showMessage("Результат сохранен.")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить: {e}")

    def save_substitution_map(self):
        if not self.substitution_map:
            QMessageBox.warning(self, "Внимание", "Карта замен пуста.")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить карту замен", "substitution_map.csv", "CSV files (*.csv)"
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8-sig', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(["Зашифр.", "Расшифр."])
                    for k, v in sorted(self.substitution_map.items()):
                        w.writerow([k, v])
                QMessageBox.information(self, "Успех", f"Карта сохранена в {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить: {e}")

    def on_encrypted_text_changed(self):
        t = self.encrypted_input.toPlainText()
        if not t:
            self.decrypted_output.clear()
            self.substitution_map = {}
            self.status_bar.showMessage("Ожидание ввода...")
            return
        freq_data = self._calculate_frequencies(t)
        self.status_bar.showMessage(f"Символов уникальных: {len(freq_data)}. Нажмите 'Автоанализ'.")

    def auto_substitute(self):
        text = self.encrypted_input.toPlainText()
        if not text:
            QMessageBox.warning(self, "Внимание", "Введите текст.")
            return
        encrypted_freq = self._calculate_frequencies(text)
        encrypted_chars = [char for char, _ in encrypted_freq]
        reference_chars = sorted(
            self.reference_frequencies.keys(),
            key=lambda x: self.reference_frequencies[x],
            reverse=True
        )
        self.substitution_map = {}
        for i, enc_char in enumerate(encrypted_chars):
            if i < len(reference_chars):
                self.substitution_map[enc_char] = reference_chars[i]
        self.apply_substitution()
        self.status_bar.showMessage(f"Автоанализ: {len(self.substitution_map)} замен.")
        QMessageBox.information(self, "Автоанализ", "Начальная карта сформирована.\nМожно нажать 'Уточнить замену'.")

    def refine_mapping(self):
        if not self.substitution_map:
            QMessageBox.warning(self, "Внимание", "Сначала выполните автоанализ.")
            return
        encrypted_text = self.encrypted_input.toPlainText()
        if not encrypted_text:
            QMessageBox.warning(self, "Внимание", "Нет зашифрованного текста.")
            return
        self.status_bar.showMessage("Уточнение карты... (несколько секунд)")
        QApplication.processEvents()

        new_map = refine_substitution_map(
            encrypted_text,
            self.substitution_map,
            self.reference_frequencies,
            iterations=2000,
            stagnation_limit=400
        )
        self.substitution_map = new_map
        self.apply_substitution()

        decoded = self.decrypted_output.toPlainText()
        score = score_decryption(decoded, self.reference_frequencies)
        self.status_bar.showMessage(f"Карта уточнена. Итоговый скор: {score:.3f}")
        QMessageBox.information(self, "Готово", f"Карта замен уточнена.\nСкор: {score:.3f}")

    def show_substitution_dialog(self):
        if not self.encrypted_input.toPlainText():
            QMessageBox.warning(self, "Внимание", "Введите зашифрованный текст.")
            return
        encrypted_freq = self._calculate_frequencies(self.encrypted_input.toPlainText())
        encrypted_freq_dict = {char: prob for char, prob in encrypted_freq}
        dialog = SubstitutionMapDialog(
            self,
            self.substitution_map,
            encrypted_freq_dict,
            self.reference_frequencies
        )
        if dialog.exec():
            self.substitution_map = dialog.get_substitution_map()
            self.apply_substitution()

    def apply_substitution(self):
        encrypted_text = self.encrypted_input.toPlainText()
        decrypted_text = apply_map(encrypted_text, self.substitution_map)
        self.decrypted_output.setText(decrypted_text)

    def load_standard_reference_table(self):
        self.reference_frequencies = STANDARD_RUSSIAN_FREQ.copy()
        self.status_bar.showMessage("Стандартная таблица частот загружена.")

    def show_encrypted_frequency(self):
        text = self.encrypted_input.toPlainText()
        if not text:
            QMessageBox.warning(self, "Внимание", "Текст пуст.")
            return
        frequency_data = self._calculate_frequencies(text)
        FrequencyTableDialog(self, frequency_data).exec()

    def show_reference_frequency(self):
        if not self.reference_frequencies:
            QMessageBox.warning(self, "Внимание", "Таблица частот не загружена.")
            return
        frequency_data = sorted(self.reference_frequencies.items(), key=lambda x: x[1], reverse=True)
        FrequencyTableDialog(self, frequency_data).exec()

    def _calculate_frequencies(self, text: str):
        if not text:
            return []
        counts = {}
        for ch in text:
            counts[ch] = counts.get(ch, 0) + 1
        total = len(text)
        return sorted([(c, counts[c]/total) for c in counts], key=lambda x: x[1], reverse=True)

# ================== ДИАЛОГ РЕДАКТИРОВАНИЯ КАРТЫ ==================
class SubstitutionMapDialog(QDialog):
    def __init__(self, parent=None, substitution_map=None, encrypted_freq=None, reference_freq=None):
        super().__init__(parent)
        self.substitution_map = substitution_map.copy() if substitution_map else {}
        self.encrypted_freq = encrypted_freq or {}
        self.reference_freq = reference_freq or {}
        self.setWindowTitle("Редактирование карты замен")
        self.setGeometry(100, 100, 1200, 700)
        self.setup_ui()

    def setup_ui(self):
        main_layout = QHBoxLayout(self)

        left_panel = QVBoxLayout()
        left_label = QLabel("Частоты шифротекста")
        left_label.setAlignment(Qt.AlignCenter)
        left_panel.addWidget(left_label)

        self.encrypted_table = QTableWidget()
        self.encrypted_table.setColumnCount(2)
        self.encrypted_table.setHorizontalHeaderLabels(["Символ", "Вероятность"])
        self.encrypted_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.encrypted_table.setEditTriggers(QTableWidget.NoEditTriggers)

        encrypted_sorted = sorted(self.encrypted_freq.items(), key=lambda x: x[1], reverse=True)
        self.encrypted_table.setRowCount(len(encrypted_sorted))
        for row, (char, prob) in enumerate(encrypted_sorted):
            ci = QTableWidgetItem(f"'{char}'")
            ci.setTextAlignment(Qt.AlignCenter)
            pi = QTableWidgetItem(f"{prob:.4f}")
            pi.setTextAlignment(Qt.AlignCenter)
            self.encrypted_table.setItem(row, 0, ci)
            self.encrypted_table.setItem(row, 1, pi)

        left_panel.addWidget(self.encrypted_table)
        main_layout.addLayout(left_panel, 1)

        center_panel = QVBoxLayout()
        center_label = QLabel("Карта замен")
        center_label.setAlignment(Qt.AlignCenter)
        center_panel.addWidget(center_label)

        scroll = QScrollArea()
        scroll_widget = QWidget()
        self.map_layout = QGridLayout(scroll_widget)
        self.map_layout.addWidget(QLabel("Зашифр."), 0, 0)
        self.map_layout.addWidget(QLabel("→"), 0, 1)
        self.map_layout.addWidget(QLabel("Расшифр."), 0, 2)

        self.line_edits = {}
        row = 1
        for enc_char, enc_prob in encrypted_sorted:
            enc_label = QLabel(f"'{enc_char}'\n({enc_prob:.4f})")
            enc_label.setAlignment(Qt.AlignCenter)
            self.map_layout.addWidget(enc_label, row, 0)
            self.map_layout.addWidget(QLabel("→"), row, 1)

            dec_char = self.substitution_map.get(enc_char, "")
            dec_prob = self.reference_freq.get(dec_char, 0.0) if dec_char else 0.0

            line_edit = QLineEdit(dec_char)
            line_edit.setMaxLength(1)
            line_edit.setAlignment(Qt.AlignCenter)
            prob_label = QLabel(f"({dec_prob:.4f})")
            prob_label.setAlignment(Qt.AlignCenter)

            line_edit.textChanged.connect(
                lambda text, lbl=prob_label: lbl.setText(f"({self.reference_freq.get(text, 0.0):.4f})")
            )

            container = QWidget()
            v = QVBoxLayout(container)
            v.setContentsMargins(0,0,0,0)
            v.setSpacing(2)
            v.addWidget(line_edit)
            v.addWidget(prob_label)

            self.line_edits[enc_char] = line_edit
            self.map_layout.addWidget(container, row, 2)
            row += 1

        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        center_panel.addWidget(scroll)

        btns = QHBoxLayout()
        apply_btn = QPushButton("Применить")
        apply_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Отмена")
        cancel_btn.clicked.connect(self.reject)
        btns.addWidget(apply_btn)
        btns.addWidget(cancel_btn)
        center_panel.addLayout(btns)

        main_layout.addLayout(center_panel, 1)

        right_panel = QVBoxLayout()
        right_label = QLabel("Эталон")
        right_label.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(right_label)

        self.reference_table = QTableWidget()
        self.reference_table.setColumnCount(2)
        self.reference_table.setHorizontalHeaderLabels(["Символ", "Вероятность"])
        self.reference_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.reference_table.setEditTriggers(QTableWidget.NoEditTriggers)

        reference_sorted = sorted(self.reference_freq.items(), key=lambda x: x[1], reverse=True)
        self.reference_table.setRowCount(len(reference_sorted))
        for row, (char, prob) in enumerate(reference_sorted):
            ci = QTableWidgetItem(f"'{char}'")
            ci.setTextAlignment(Qt.AlignCenter)
            pi = QTableWidgetItem(f"{prob:.4f}")
            pi.setTextAlignment(Qt.AlignCenter)
            self.reference_table.setItem(row, 0, ci)
            self.reference_table.setItem(row, 1, pi)

        right_panel.addWidget(self.reference_table)
        main_layout.addLayout(right_panel, 1)

    def get_substitution_map(self):
        result = {}
        for enc_char, le in self.line_edits.items():
            dec_char = le.text()
            if dec_char != "":
                result[enc_char] = dec_char
        return result

# ================== ГЛАВНОЕ ОКНО ==================
class CipherApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Шифратор и Криптоанализ")
        self.setGeometry(100, 100, 1400, 900)
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.encryption_tab = EncryptionTab()
        self.cryptanalysis_tab = CryptanalysisTab()
        self.tabs.addTab(self.encryption_tab, "Шифрование")
        self.tabs.addTab(self.cryptanalysis_tab, "Криптоанализ")
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Готов к работе.")

# ================== SELF-TEST (опционально) ==================
def run_self_tests():
    sample = "Привет,\nМИР!!!   Это   тест — строка... ‘кавычки’ и   ТАБ\tещё."
    prep = preprocess_text(sample)
    print("Исходное: ", sample)
    print("Предобработка:", prep)
    # Проверим что нет табов/переводов строки
    assert "\n" not in prep and "\t" not in prep
    # Проверим что только символы из алфавита
    assert all(ch in ALPHABET for ch in prep)
    # Простой тест Цезаря
    c_enc = caesar_cipher(prep, 5, True)
    c_dec = caesar_cipher(c_enc, 5, False)
    assert c_dec == prep
    print("Тесты пройдены.")

if __name__ == "__main__":
    # При необходимости раскомментируйте для проверки:
    # run_self_tests()
    app = QApplication(sys.argv)
    app.setFont(QFont(app.font().family(), 10))
    window = CipherApp()
    window.show()
    sys.exit(app.exec())
