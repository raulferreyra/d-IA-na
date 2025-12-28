from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QComboBox,
    QCheckBox,
    QSlider,
    QPushButton,
    QLabel,
)


class SettingsDialog(QDialog):
    def __init__(self, parent=None, current=None):
        super().__init__(parent)

        self.setWindowTitle("Configuración")
        self.setMinimumWidth(460)

        form = QFormLayout(self)

        self.provider = QComboBox()
        self.provider.addItems(["openai", "anthropic", "deepseek", "local"])
        self.provider.setCurrentText((current or {}).get("provider", "openai"))

        self.model = QLineEdit((current or {}).get("model", "gpt-4o-mini"))

        self.api_key = QLineEdit((current or {}).get("api_key", ""))
        self.api_key.setEchoMode(QLineEdit.Password)

        self.show_key = QCheckBox("Mostrar API key")
        self.show_key.setChecked(False)
        self.show_key.stateChanged.connect(self._toggle_key_visibility)

        self.tts_enabled = QCheckBox("Voz habilitada")
        self.tts_enabled.setChecked(
            bool((current or {}).get("tts_enabled", False)))

        self.tts_volume = QSlider(Qt.Horizontal)
        self.tts_volume.setRange(0, 100)
        self.tts_volume.setValue(int((current or {}).get("tts_volume", 35)))

        self.language = QComboBox()
        self.language.addItems(["es", "en"])
        self.language.setCurrentText((current or {}).get("language", "es"))

        # Persona: base fixed + extra tags only
        self.persona_base = QLabel("técnico, directo")
        self.persona_extra = QLineEdit(
            (current or {}).get("persona_extra", ""))
        self.persona_extra.setPlaceholderText(
            "Ej: bromista, serio, sarcástico")

        form.addRow("Proveedor IA", self.provider)
        form.addRow("Modelo IA", self.model)
        form.addRow("API Key", self.api_key)
        form.addRow("", self.show_key)
        form.addRow("Voz", self.tts_enabled)
        form.addRow("Volumen", self.tts_volume)
        form.addRow("Idioma", self.language)
        form.addRow("Base (fijo)", self.persona_base)
        form.addRow("Estilo extra (tags)", self.persona_extra)

        btns = QHBoxLayout()
        btns.addStretch(1)

        self.btn_cancel = QPushButton("Cancelar")
        self.btn_save = QPushButton("Guardar")

        self.btn_cancel.clicked.connect(self.reject)
        self.btn_save.clicked.connect(self.accept)

        btns.addWidget(self.btn_cancel)
        btns.addWidget(self.btn_save)
        form.addRow(btns)

    def _toggle_key_visibility(self):
        self.api_key.setEchoMode(
            QLineEdit.Normal if self.show_key.isChecked() else QLineEdit.Password)

    def get_values(self) -> dict:
        return {
            "provider": self.provider.currentText().strip(),
            "model": self.model.text().strip(),
            "api_key": self.api_key.text().strip(),
            "tts_enabled": self.tts_enabled.isChecked(),
            "tts_volume": int(self.tts_volume.value()),
            "language": self.language.currentText().strip(),
            "persona_extra": self.persona_extra.text().strip(),
        }
