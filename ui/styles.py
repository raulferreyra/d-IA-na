def overlay_stylesheet() -> str:
    return """
        QWidget {
            background: rgba(0, 0, 0, 140);
            border-radius: 12px;
        }
        QLabel {
            color: #E9F2FF;
            font-size: 15px;
            font-weight: 600;
        }
        QTextEdit {
            background: rgba(10, 20, 40, 180);
            color: #E9F2FF;
            border: 1px solid rgba(233, 242, 255, 120);
            border-radius: 10px;
            padding: 8px;
            font-family: Consolas, "Cascadia Mono", monospace;
            font-size: 12px;
        }
    """


def mic_idle_style(btn_size: int) -> str:
    return f"""
        QPushButton {{
            background-color: rgba(190, 70, 70, 210);
            border: 1px solid rgba(255, 255, 255, 120);
            border-radius: {btn_size // 2}px;
            padding: 8px;
        }}
        QPushButton:hover {{
            background-color: rgba(210, 85, 85, 220);
        }}
    """


def mic_recording_style(btn_size: int) -> str:
    return f"""
        QPushButton {{
            background-color: rgba(80, 170, 110, 230);
            border: 1px solid rgba(255, 255, 255, 120);
            border-radius: {btn_size // 2}px;
            padding: 8px;
        }}
        QPushButton:hover {{
            background-color: rgba(95, 190, 125, 240);
        }}
    """
