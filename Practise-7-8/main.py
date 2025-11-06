import sys
import string
import math
import csv
from functools import reduce

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QComboBox, QLineEdit, QLabel, QFileDialog,
    QMessageBox, QGridLayout, QStatusBar, QTableWidget, QTableWidgetItem,
    QDialog, QTabWidget, QScrollArea, QSplitter, QHeaderView
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

# --- Логика шифрования ---
LOWERCASE_CYRILLIC = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
SYMBOLS = string.punctuation + ' '
ALPHABET = LOWERCASE_CYRILLIC + SYMBOLS
MOD = len(ALPHABET)

# Эталонная частотность русского языка
STANDARD_RUSSIAN_FREQ = {
    ' ': 0.175, 'о': 0.090, 'е': 0.072, 'а': 0.062, 'и': 0.062, 'н': 0.053, 'т': 0.053,
    'с': 0.045, 'р': 0.040, 'в': 0.038, 'л': 0.035, 'к': 0.028, 'м': 0.026, 'д': 0.025,
    'п': 0.023, 'у': 0.021, 'я': 0.018, 'ы': 0.016, 'з': 0.016, 'ь': 0.014, 'ъ': 0.014,
    'б': 0.014, 'г': 0.013, 'ч': 0.012, 'й': 0.010, 'х': 0.009, 'ж': 0.007, 'ю': 0.006,
    'ш': 0.006, 'ц': 0.004, 'щ': 0.003, 'э': 0.003, 'ф': 0.002, 'ё': 0.002
}


def preprocess_text(text: str) -> str:
    text = text.lower()
    return "".join([char for char in text if char in ALPHABET])


def caesar_cipher(text: str, key: int, encrypt=True) -> str:
    result = []
    shift = key if encrypt else -key
    for char in text:
        original_index = ALPHABET.find(char)
        if original_index != -1:
            new_index = (original_index + shift) % MOD
            result.append(ALPHABET[new_index])
        else:
            result.append(char)
    return "".join(result)


def extended_gcd(a: int, b: int) -> tuple:
    if a == 0:
        return b, 0, 1
    else:
        g, y, x = extended_gcd(b % a, a)
        return g, x - (b // a) * y, y


def modinv(a: int, m: int) -> int:
    g, x, _ = extended_gcd(a, m)
    if g != 1:
        raise ValueError(f"Обратного элемента для {a} по модулю {m} не существует.")
    else:
        return x % m


def affine_cipher(text: str, key_a: int, key_b: int, encrypt=True) -> str:
    if math.gcd(key_a, MOD) != 1:
        raise ValueError(f"Ключ 'a' ({key_a}) должен быть взаимно прост с {MOD}.")
    result = []
    for char in text:
        original_index = ALPHABET.find(char)
        if original_index != -1:
            x = original_index
            if encrypt:
                y = (key_a * x + key_b) % MOD
            else:
                a_inv = modinv(key_a, MOD)
                y = (a_inv * (x - key_b)) % MOD
            result.append(ALPHABET[y])
        else:
            result.append(char)
    return "".join(result)


def vigenere_cipher(text: str, key: str, encrypt=True) -> str:
    if not key:
        raise ValueError("Ключ для шифра Виженера не может быть пустым.")
    key = preprocess_text(key)
    if not key:
        raise ValueError("Ключ не содержит кириллических букв.")
    result = []
    key_index = 0
    for char in text:
        original_index = ALPHABET.find(char)
        if original_index != -1:
            shift = ALPHABET.find(key[key_index % len(key)])
            current_shift = shift if encrypt else -shift
            new_index = (original_index + current_shift) % MOD
            result.append(ALPHABET[new_index])
            key_index += 1
        else:
            result.append(char)
    return "".join(result)


def gronsfeld_cipher(text: str, key: str, encrypt=True) -> str:
    if not key.isdigit():
        raise ValueError("Ключ для шифра Гронсфельда должен состоять только из цифр.")
    result = []
    key_index = 0
    for char in text:
        original_index = ALPHABET.find(char)
        if original_index != -1:
            shift = int(key[key_index % len(key)])
            current_shift = shift if encrypt else -shift
            new_index = (original_index + current_shift) % MOD
            result.append(ALPHABET[new_index])
            key_index += 1
        else:
            result.append(char)
    return "".join(result)


# --- Класс для окна с таблицей частот ---
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


# --- Вкладка Шифрования ---
class EncryptionTab(QWidget):
    def __init__(self):
        super().__init__()
        self.current_state = 'initial'
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)

        # Верхняя панель
        top_layout = QHBoxLayout()
        self.open_button = QPushButton("Открыть .txt файл")
        self.open_button.clicked.connect(self.open_file)
        self.save_button = QPushButton("Сохранить результат")
        self.save_button.clicked.connect(self.save_file)
        top_layout.addWidget(self.open_button)
        top_layout.addWidget(self.save_button)
        top_layout.addStretch(1)
        main_layout.addLayout(top_layout)

        # Текстовые области
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

        # Панель управления
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

        # Кнопки частот
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

        # Статус бар
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
                self.status_bar.showMessage("Ошибка при чтении файла.")

    def save_file(self):
        text_to_save = self.result_text_edit.toPlainText()
        if not text_to_save:
            QMessageBox.warning(self, "Внимание", "Нет текста для сохранения.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить результат", "", "Текстовые файлы (*.txt)"
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text_to_save)
                self.status_bar.showMessage(f"Результат сохранен.")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл: {e}")

    def get_key(self):
        key_str = self.key_input.text().strip()
        if not key_str:
            raise ValueError("Ключ не может быть пустым.")
        return key_str

    def encrypt_text(self):
        try:
            if self.current_state not in ['initial', 'preprocessed', 'decrypted']:
                QMessageBox.warning(self, "Неверная операция", "Текст уже зашифрован.")
                return

            key_str = self.get_key()
            raw_text = self.original_text_edit.toPlainText()
            if not raw_text:
                QMessageBox.warning(self, "Внимание", "Нет исходного текста.")
                return

            text_to_process = preprocess_text(raw_text)
            cipher_name = self.cipher_combo.currentText()
            encrypted_text = ""

            if cipher_name == "Цезарь":
                key = int(key_str)
                encrypted_text = caesar_cipher(text_to_process, key, encrypt=True)
            elif cipher_name == "Аффинный":
                try:
                    key_a, key_b = map(int, key_str.split(','))
                except ValueError:
                    raise ValueError("Для аффинного шифра введите два ключа через запятую (например, 5,2).")
                encrypted_text = affine_cipher(text_to_process, key_a, key_b, encrypt=True)
            elif cipher_name == "Виженер":
                encrypted_text = vigenere_cipher(text_to_process, key_str, encrypt=True)
            elif cipher_name == "Гронсфельд":
                encrypted_text = gronsfeld_cipher(text_to_process, key_str, encrypt=True)

            self.result_text_edit.setText(encrypted_text)
            self.current_state = 'encrypted'
            self.status_bar.showMessage(f"Текст успешно зашифрован шифром {cipher_name}.")
        except ValueError as e:
            QMessageBox.critical(self, "Ошибка", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Ошибка шифрования", f"Произошла непредвиденная ошибка: {e}")

    def decrypt_text(self):
        try:
            if self.current_state != 'encrypted':
                QMessageBox.warning(self, "Неверная операция", "Расшифровка возможна только для зашифрованного текста.")
                return

            key_str = self.get_key()
            text_to_decrypt = self.result_text_edit.toPlainText()
            if not text_to_decrypt:
                QMessageBox.warning(self, "Внимание", "Нет текста для расшифровки.")
                return

            cipher_name = self.cipher_combo.currentText()
            decrypted_text = ""

            if cipher_name == "Цезарь":
                key = int(key_str)
                decrypted_text = caesar_cipher(text_to_decrypt, key, encrypt=False)
            elif cipher_name == "Аффинный":
                try:
                    key_a, key_b = map(int, key_str.split(','))
                except ValueError:
                    raise ValueError("Для аффинного шифра введите два ключа через запятую (например, 5,2).")
                decrypted_text = affine_cipher(text_to_decrypt, key_a, key_b, encrypt=False)
            elif cipher_name == "Виженер":
                decrypted_text = vigenere_cipher(text_to_decrypt, key_str, encrypt=False)
            elif cipher_name == "Гронсфельд":
                decrypted_text = gronsfeld_cipher(text_to_decrypt, key_str, encrypt=False)

            self.result_text_edit.setText(decrypted_text)
            self.current_state = 'decrypted'
            self.status_bar.showMessage(f"Текст успешно расшифрован.")
        except ValueError as e:
            QMessageBox.critical(self, "Ошибка", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Ошибка расшифровки", f"Произошла непредвиденная ошибка: {e}")

    def _calculate_frequencies(self, text: str):
        if not text:
            return []
        counts = {}
        for char in text:
            counts[char] = counts.get(char, 0) + 1
        total_chars = len(text)
        return sorted(
            [(char, count / total_chars) for char, count in counts.items()],
            key=lambda item: item[1],
            reverse=True
        )

    def show_frequency_table_original(self):
        text_to_analyze = self.original_text_edit.toPlainText()
        if not text_to_analyze:
            QMessageBox.warning(self, "Внимание", "Исходный текст пуст.")
            return
        frequency_data = self._calculate_frequencies(text_to_analyze)
        dialog = FrequencyTableDialog(self, frequency_data)
        dialog.exec()

    def show_frequency_table_preprocessed(self):
        raw_text = self.original_text_edit.toPlainText()
        if not raw_text:
            QMessageBox.warning(self, "Внимание", "Исходный текст пуст.")
            return
        text_to_analyze = preprocess_text(raw_text)
        if not text_to_analyze:
            QMessageBox.warning(self, "Внимание", "После предобработки текст пуст.")
            return
        frequency_data = self._calculate_frequencies(text_to_analyze)
        dialog = FrequencyTableDialog(self, frequency_data)
        dialog.exec()

    def show_frequency_table_result(self):
        text_to_analyze = self.result_text_edit.toPlainText()
        if not text_to_analyze:
            QMessageBox.warning(self, "Внимание", "Нет текста для анализа.")
            return
        frequency_data = self._calculate_frequencies(text_to_analyze)
        dialog = FrequencyTableDialog(self, frequency_data)
        dialog.exec()


# --- УЛУЧШЕННАЯ ВКЛАДКА КРИПТОАНАЛИЗА ---
class CryptanalysisTab(QWidget):
    def __init__(self):
        super().__init__()
        self.substitution_map = {}
        self.reference_frequencies = {}
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)

        # Верхняя панель с кнопками (убран лейбл "Файл не выбран")
        top_layout = QHBoxLayout()
        self.open_button = QPushButton("Открыть .txt файл")
        self.open_button.clicked.connect(self.load_encrypted_file)
        self.save_button = QPushButton("Сохранить результат")
        self.save_button.clicked.connect(self.save_decrypted_result)

        self.import_freq_button = QPushButton("Импорт эталонной таблицы")
        self.import_freq_button.clicked.connect(self.import_reference_frequency)
        self.import_freq_button.setStyleSheet("""
            QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 5px; }
            QPushButton:hover { background-color: #0b7dda; }
        """)

        top_layout.addWidget(self.open_button)
        top_layout.addWidget(self.save_button)
        top_layout.addWidget(self.import_freq_button)
        top_layout.addStretch(1)
        main_layout.addLayout(top_layout)

        # Текстовые области
        text_layout = QHBoxLayout()

        encrypted_group = QVBoxLayout()
        encrypted_group.addWidget(QLabel("Зашифрованный текст:"))
        self.encrypted_input = QTextEdit()
        self.encrypted_input.setPlaceholderText("Введите или вставьте зашифрованный текст...")
        self.encrypted_input.textChanged.connect(self.on_encrypted_text_changed)
        encrypted_group.addWidget(self.encrypted_input)
        text_layout.addLayout(encrypted_group)

        result_group = QVBoxLayout()
        result_group.addWidget(QLabel("Результат расшифровки:"))
        self.decrypted_output = QTextEdit()
        self.decrypted_output.setReadOnly(True)
        self.decrypted_output.setPlaceholderText("Результат появится здесь...")
        result_group.addWidget(self.decrypted_output)
        text_layout.addLayout(result_group)

        main_layout.addLayout(text_layout)

        # Панель управления
        control_layout = QGridLayout()

        self.auto_button = QPushButton("1. Автоматический анализ")
        self.auto_button.clicked.connect(self.auto_substitute)
        self.auto_button.setStyleSheet("""
            QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }
            QPushButton:hover { background-color: #45a049; }
        """)

        self.show_map_button = QPushButton("2. Редактировать замены")
        self.show_map_button.clicked.connect(self.show_substitution_dialog)

        freq_button_layout = QGridLayout()
        self.freq_encrypted_button = QPushButton("Частота зашифрованного")
        self.freq_encrypted_button.clicked.connect(self.show_encrypted_frequency)
        self.freq_reference_button = QPushButton("Частота эталонная")
        self.freq_reference_button.clicked.connect(self.show_reference_frequency)
        self.save_map_button = QPushButton("Сохранить карту замен")
        self.save_map_button.clicked.connect(self.save_substitution_map)

        freq_button_layout.addWidget(self.freq_encrypted_button, 0, 0)
        freq_button_layout.addWidget(self.freq_reference_button, 0, 1)
        freq_button_layout.addWidget(self.save_map_button, 1, 0, 1, 2)

        control_layout.addWidget(self.auto_button, 0, 0)
        control_layout.addWidget(self.show_map_button, 0, 1)
        control_layout.addLayout(freq_button_layout, 1, 0, 1, 2)

        main_layout.addLayout(control_layout)

        self.status_bar = QStatusBar()
        main_layout.addWidget(self.status_bar)
        self.status_bar.showMessage("Готов к анализу. Введите зашифрованный текст или откройте файл.")

        self.load_standard_reference_table()

    def import_reference_frequency(self):
        """Импорт эталонной таблицы частот из CSV файла (поддержка , ; таб, запятая как десятичный разделитель и символ-пробел)."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Импорт эталонной таблицы частот", "", "CSV files (*.csv);;All files (*.*)"
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
                    # По умолчанию запятая
                    dialect = csv.excel
                reader = csv.reader(f, dialect)
                # Пытаемся пропустить заголовок
                first = next(reader, None)
                def try_float(s: str):
                    try:
                        return float(s.strip().replace(',', '.'))
                    except Exception:
                        return None
                new_freq = {}

                # Если первая строка похожа на заголовок, не используем её
                if first and (try_float(first[1]) is None):
                    pass  # заголовок
                else:
                    if first and len(first) >= 2:
                        raw_char = first[0]
                        prob = try_float(first[1])
                        if prob is not None:
                            char = raw_char
                            # убираем только внешние кавычки, но не пробелы
                            if (len(char) >= 2) and ((char[0] == "'" and char[-1] == "'") or (char[0] == '"' and char[-1] == '"')):
                                char = char[1:-1]
                            new_freq[char] = prob

                for row in reader:
                    if not row or len(row) < 2:
                        continue
                    raw_char = row[0]
                    prob = try_float(row[1])
                    if prob is None:
                        continue
                    char = raw_char
                    # убираем только внешние кавычки, но не пробелы
                    if (len(char) >= 2) and ((char[0] == "'" and char[-1] == "'") or (char[0] == '"' and char[-1] == '"')):
                        char = char[1:-1]
                    # допускаем любой одиночный символ, включая пробел
                    if char == "":
                        continue
                    new_freq[char] = prob

            if new_freq:
                self.reference_frequencies = new_freq
                QMessageBox.information(
                    self,
                    "Успех",
                    f"Эталонная таблица импортирована. Загружено {len(new_freq)} символов."
                )
                self.status_bar.showMessage(f"Импортирована эталонная таблица: {len(new_freq)} символов")
            else:
                QMessageBox.warning(self, "Внимание", "Не удалось прочитать данные из файла.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось импортировать файл: {e}")

    def load_encrypted_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Открыть зашифрованный текст", "", "Текстовые файлы (*.txt)"
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.encrypted_input.setText(content)
                self.status_bar.showMessage("Файл успешно открыт.")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось прочитать файл: {e}")
                self.status_bar.showMessage("Ошибка при чтении файла.")

    def save_decrypted_result(self):
        result = self.decrypted_output.toPlainText()
        if not result:
            QMessageBox.warning(self, "Внимание", "Нет результата для сохранения.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить результат", "", "Текстовые файлы (*.txt)"
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(result)
                self.status_bar.showMessage("Результат сохранен.")
                QMessageBox.information(self, "Успех", f"Результат сохранен в {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл: {e}")

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
                    writer = csv.writer(f)
                    writer.writerow(["Зашифрованный символ", "Расшифрованный символ"])
                    for enc_char, dec_char in sorted(self.substitution_map.items()):
                        writer.writerow([enc_char, dec_char])
                QMessageBox.information(self, "Успех", f"Карта замен сохранена в {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл: {e}")

    def on_encrypted_text_changed(self):
        text = self.encrypted_input.toPlainText()
        if not text:
            self.decrypted_output.clear()
            self.substitution_map = {}
            self.status_bar.showMessage("Ожидание ввода текста...")
            return

        freq_data = self._calculate_frequencies(text)
        self.status_bar.showMessage(f"Найдено {len(freq_data)} уникальных символов. Нажмите 'Автоматический анализ'.")

    def auto_substitute(self):
        """Автоматическая замена на основе частотного анализа"""
        text = self.encrypted_input.toPlainText()
        if not text:
            QMessageBox.warning(self, "Внимание", "Сначала введите зашифрованный текст.")
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
        self.status_bar.showMessage(f"Выполнен автоматический анализ. Создано {len(self.substitution_map)} замен.")

        QMessageBox.information(
            self,
            "Автоанализ выполнен",
            f"Создано {len(self.substitution_map)} автоматических замен.\n\n"
            "Вы можете отредактировать замены через кнопку 'Редактировать замены'."
        )

    def show_substitution_dialog(self):
        """Показать диалог редактирования замен"""
        if not self.encrypted_input.toPlainText():
            QMessageBox.warning(self, "Внимание", "Сначала введите зашифрованный текст.")
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
        decrypted_text = ""
        for char in encrypted_text:
            decrypted_text += self.substitution_map.get(char, char)
        self.decrypted_output.setText(decrypted_text)

    def load_standard_reference_table(self):
        self.reference_frequencies = STANDARD_RUSSIAN_FREQ.copy()
        self.status_bar.showMessage("Загружена стандартная таблица частот русского языка.")

    def show_encrypted_frequency(self):
        text = self.encrypted_input.toPlainText()
        if not text:
            QMessageBox.warning(self, "Внимание", "Зашифрованный текст пуст.")
            return
        frequency_data = self._calculate_frequencies(text)
        dialog = FrequencyTableDialog(self, frequency_data)
        dialog.exec()

    def show_reference_frequency(self):
        if not self.reference_frequencies:
            QMessageBox.warning(self, "Внимание", "Эталонная таблица не загружена.")
            return
        frequency_data = sorted(
            self.reference_frequencies.items(),
            key=lambda item: item[1],
            reverse=True
        )
        dialog = FrequencyTableDialog(self, frequency_data)
        dialog.exec()

    def _calculate_frequencies(self, text: str):
        if not text:
            return []
        counts = {}
        for char in text:
            counts[char] = counts.get(char, 0) + 1
        total_chars = len(text)
        return sorted(
            [(char, count / total_chars) for char, count in counts.items()],
            key=lambda item: item[1],
            reverse=True
        )


# --- Диалог для редактирования карты замен с таблицами частот ---
class SubstitutionMapDialog(QDialog):
    def __init__(self, parent=None, substitution_map=None, encrypted_freq=None, reference_freq=None):
        super().__init__(parent)
        self.substitution_map = substitution_map.copy() if substitution_map else {}
        self.encrypted_freq = encrypted_freq or {}
        self.reference_freq = reference_freq or {}
        self.setWindowTitle("Редактирование карты замен с таблицами частот")
        self.setGeometry(100, 100, 1200, 700)
        self.setup_ui()

    def setup_ui(self):
        main_layout = QHBoxLayout(self)

        # Левая панель - Частоты шифротекста
        left_panel = QVBoxLayout()
        left_label = QLabel("Частоты зашифрованного текста")
        left_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        left_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_panel.addWidget(left_label)

        self.encrypted_table = QTableWidget()
        self.encrypted_table.setColumnCount(2)
        self.encrypted_table.setHorizontalHeaderLabels(["Символ", "Вероятность"])
        self.encrypted_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.encrypted_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.encrypted_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        encrypted_sorted = sorted(self.encrypted_freq.items(), key=lambda x: x[1], reverse=True)
        self.encrypted_table.setRowCount(len(encrypted_sorted))
        for row, (char, prob) in enumerate(encrypted_sorted):
            char_item = QTableWidgetItem(f"'{char}'")
            char_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            prob_item = QTableWidgetItem(f"{prob:.4f}")
            prob_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.encrypted_table.setItem(row, 0, char_item)
            self.encrypted_table.setItem(row, 1, prob_item)

        left_panel.addWidget(self.encrypted_table)
        main_layout.addLayout(left_panel, 1)

        # Центральная панель - Карта замен (по всем символам шифротекста)
        center_panel = QVBoxLayout()
        center_label = QLabel("Карта замен")
        center_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        center_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center_panel.addWidget(center_label)

        scroll = QScrollArea()
        scroll_widget = QWidget()
        self.map_layout = QGridLayout(scroll_widget)
        self.map_layout.setSpacing(8)

        self.map_layout.addWidget(QLabel("Зашифр."), 0, 0)
        self.map_layout.addWidget(QLabel("→"), 0, 1)
        self.map_layout.addWidget(QLabel("Расшифр."), 0, 2)

        self.line_edits = {}
        row = 1

        for enc_char, enc_prob in encrypted_sorted:
            enc_text = f"'{enc_char}'\n({enc_prob:.4f})"
            enc_label = QLabel(enc_text)
            enc_label.setStyleSheet("font-size: 12px; font-weight: bold;")
            enc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.map_layout.addWidget(enc_label, row, 0)

            arrow_label = QLabel("→")
            arrow_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.map_layout.addWidget(arrow_label, row, 1)

            dec_char = self.substitution_map.get(enc_char, "")
            dec_prob = self.reference_freq.get(dec_char, 0.0) if dec_char else 0.0

            line_edit = QLineEdit(dec_char)
            line_edit.setMaxLength(1)
            line_edit.setFixedWidth(60)
            line_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
            line_edit.setStyleSheet("""
                QLineEdit { font-size: 14px; font-weight: bold; border: 2px solid #4CAF50; border-radius: 3px; padding: 5px; }
            """)

            prob_label = QLabel(f"({dec_prob:.4f})")
            prob_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            prob_label.setStyleSheet("font-size: 10px; color: #666;")

            line_edit.textChanged.connect(
                lambda text, lbl=prob_label: lbl.setText(f"({self.reference_freq.get(text, 0.0):.4f})")
            )

            container = QWidget()
            v = QVBoxLayout(container)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(2)
            v.addWidget(line_edit)
            v.addWidget(prob_label)

            self.line_edits[enc_char] = line_edit
            self.map_layout.addWidget(container, row, 2)

            row += 1

        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        center_panel.addWidget(scroll)

        button_layout = QHBoxLayout()
        self.apply_button = QPushButton("Применить")
        self.apply_button.clicked.connect(self.accept)
        self.apply_button.setStyleSheet("""
            QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; font-size: 13px; }
            QPushButton:hover { background-color: #45a049; }
        """)
        self.cancel_button = QPushButton("Отмена")
        self.cancel_button.clicked.connect(self.reject)
        self.cancel_button.setStyleSheet("QPushButton { padding: 10px; font-size: 13px; }")

        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.cancel_button)
        center_panel.addLayout(button_layout)

        main_layout.addLayout(center_panel, 1)

        # Правая панель - Эталонные частоты
        right_panel = QVBoxLayout()
        right_label = QLabel("Эталонные частоты")
        right_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        right_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_panel.addWidget(right_label)

        self.reference_table = QTableWidget()
        self.reference_table.setColumnCount(2)
        self.reference_table.setHorizontalHeaderLabels(["Символ", "Вероятность"])
        self.reference_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.reference_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.reference_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        reference_sorted = sorted(self.reference_freq.items(), key=lambda x: x[1], reverse=True)
        self.reference_table.setRowCount(len(reference_sorted))
        for row, (char, prob) in enumerate(reference_sorted):
            char_item = QTableWidgetItem(f"'{char}'")
            char_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            prob_item = QTableWidgetItem(f"{prob:.4f}")
            prob_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.reference_table.setItem(row, 0, char_item)
            self.reference_table.setItem(row, 1, prob_item)

        right_panel.addWidget(self.reference_table)
        main_layout.addLayout(right_panel, 1)

    def get_substitution_map(self):
        result = {}
        for enc_char, line_edit in self.line_edits.items():
            dec_char = line_edit.text()  # не .strip(), чтобы пробелы не терялись
            if dec_char != "":
                result[enc_char] = dec_char
        return result


# --- Главное окно приложения ---
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CipherApp()
    window.show()
    sys.exit(app.exec())
