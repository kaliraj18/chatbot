import sys
from PySide6.QtWidgets import QApplication, QVBoxLayout, QPushButton, QWidget, QTextEdit, QLineEdit, QMessageBox
from PySide6.QtCore import Slot
import requests

class Chatbot(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("Chatbot")
        layout = QVBoxLayout()

        # Widgets
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Type your message here...")
        self.send_button = QPushButton("Send")

        self.send_button.clicked.connect(self.send_message)

        # Add widgets to layout
        layout.addWidget(self.chat_display)
        layout.addWidget(self.user_input)
        layout.addWidget(self.send_button)

        self.setLayout(layout)

    @Slot()
    def send_message(self):
        # Get user input
        user_message = self.user_input.text()

        # Generate bot's response
        try:
            self.chat_display.append("You: " + user_message)
            response = self.generate_response(user_message)
            self.chat_display.append("Bot: " + response+"\n")
        except Exception as e:
            error_dialog = QMessageBox()
            error_dialog.setWindowTitle("Error: Something Went Wrong")
            error_dialog.setText(str(e))
            error_dialog.exec()

        # Clear the user input field
        self.user_input.clear()



    def generate_response(self, message):
        res = requests.post("http://localhost:8081/api/query", data={"query":message})
        return res.json()["answer"]


def main():
    # Qt Application
    app = QApplication(sys.argv)

    # Start chatbot
    chatbot = Chatbot()
    chatbot.show()

    # Execute application
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
