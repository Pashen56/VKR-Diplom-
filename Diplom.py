import pyqtgraph as pg
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QGridLayout, QComboBox
import math

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Установка заголовка окна, размеров и цвета фона
        self.setWindowTitle("График функции")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: white; color: black;")

        # Создание центрального виджета
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Создание главного вертикального лейаута
        layout = QVBoxLayout(central_widget)

        # Создание лейаута для выбора функции
        function_layout = QHBoxLayout()
        function_label = QLabel("Функция:")
        self.function_edit = QLineEdit()
        self.function_edit.setPlaceholderText("Введите функцию")
        function_layout.addWidget(function_label)
        function_layout.addWidget(self.function_edit)
        layout.addLayout(function_layout)

        # Меню для ввода параметров
        menu_layout = QGridLayout()
        self.param_values = {}
        menu_labels = ['a', 'b', 'c', 't', 'i', 'l', 'v0', 'c0', 'm', 'D']
        row = 0
        col = 0
        for label in menu_labels:
            label_widget = QLabel(label + ":")
            menu_layout.addWidget(label_widget, row, col)
            value_edit = QLineEdit()
            value_edit.setPlaceholderText("Введите значение")
            menu_layout.addWidget(value_edit, row, col + 1)
            self.param_values[label] = value_edit
            col += 2
            if col > 8:
                row += 1
                col = 0
        layout.addLayout(menu_layout)

        # Создание кнопки построения графика
        plot_button = QPushButton("Построить график")
        plot_button.setStyleSheet("background-color: #008cff; color: black;")
        plot_button.clicked.connect(self.plot_function)
        layout.addWidget(plot_button)

        # Создание лейаута для выбора предварительно заданной функции
        predefined_function_layout = QHBoxLayout()
        predefined_function_label = QLabel("Предварительно заданная функция:")
        self.predefined_function_combo = QComboBox()
        self.predefined_function_combo.addItems(['x^2', 'x^3', 'x^4', 'sin', 'cos', 'tan'])
        self.predefined_function_combo.setStyleSheet("background-color: white; color: black;")
        predefined_function_layout.addWidget(predefined_function_label)
        predefined_function_layout.addWidget(self.predefined_function_combo)
        layout.addLayout(predefined_function_layout)

        # Создание лейаута для выбора функции из списка
        function_select_layout = QHBoxLayout()
        function_select_label = QLabel("Выберите функцию:")
        self.function_select_combo = QComboBox()
        self.function_select_combo.addItems(['sin(x)', 'cos(x)', 'tan(x)', 'x*2', 'x*3', 'd*x**2', 'AnswerMurad'])
        self.function_select_combo.setStyleSheet("background-color: white; color: black;")
        self.function_select_combo.currentIndexChanged.connect(self.select_function)
        function_select_layout.addWidget(function_select_label)
        function_select_layout.addWidget(self.function_select_combo)
        layout.addLayout(function_select_layout)

        # Создание виджета для вывода графика
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#ffffff')
        layout.addWidget(self.plot_widget)
        self.plot_widget.showGrid(x=True, y=True, alpha=1)

    def plot_function(self):
        """
        Строит график в зависимости от выбранной функции
        """
        function_text = self.function_edit.text()
        predefined_function = self.predefined_function_combo.currentText()

        # Получение значений параметров из текстовых полей
        params = {}
        for key, value_edit in self.param_values.items():
            value_text = value_edit.text()
            if value_text:
                try:
                    params[key] = float(value_text)
                except ValueError:
                    pass

        def SuperSum(t, i, l, v0, c0, m, D):
            i = int(i) # количество слагаемых в сумме
            l = int(l) # длина закрепления
            pi = math.pi
            sum = 0
            s = t # момент времен
            v0 = int(v0) # постоянная скорость движения полотна
            c = int(c0) # скорость распространения колебаний в покоящемся полотне
            m = int(m) # удельная масса полотна на единицу площади
            D = int(D) # величина жесткости на изгиб

            for n in range(i):
                n = n + 1

                pi_n_l = (pi * n) / l
                teta = (1 + (((D) / (m * (c ** 2))) * ((pi_n_l ** 2)))) ** (1 / 2)

                chasti_argument_cos_i_sin_1 = pi_n_l * t * ((c * teta) - v0)
                chasti_argument_cos_i_sin_2 = pi_n_l * t * ((c * teta) + v0)

                mnogitel_cos_1 = 1 + (v0 / (c * teta))
                mnogitel_cos_2 = 1 - (v0 / (c * teta))
                mnogitel_sinusov = 1 / (pi_n_l * c * teta)

                x = np.arange(0 - 0.25, l + 0.25, 0.0001)

                U_x_t = 480 * ((l ** 8) / (pi ** 8)) * (
                        (((((-1) ** n) + 1) / (n ** 8)) * (((pi ** 2) * (n ** 2)) - 42))
                        * ((mnogitel_cos_1 * np.cos(chasti_argument_cos_i_sin_1 + pi_n_l * x)
                            + mnogitel_cos_2 * np.cos(chasti_argument_cos_i_sin_2 - pi_n_l * x))
                           +
                           ((mnogitel_sinusov) * (np.sin(chasti_argument_cos_i_sin_1 + pi_n_l * x)
                                                  + np.sin(chasti_argument_cos_i_sin_2 - pi_n_l * x))))
                )
                sum = sum + U_x_t

            sum = sum + l ** 8 / 315
            return sum

        if function_text == 'AnswerMurad':
            l = int(params['l'])
            x_values = np.arange(0-0.25, l+0.25, 0.0001)
            y_values = SuperSum(params['t'], params['i'], params['l'], params['v0'], params['c0'], params['m'], params['D'])
            self.plot_widget.plot(x_values, y_values, clear=True).setPen(pg.mkPen(color='#000000'))

        elif function_text:
            x_values = np.arange(-10, 10, 0.001)
            y_values = [eval(function_text.replace('x', str(x)), {'math': math, 'sin': math.sin, 'cos': math.cos, 'tan': math.tan, 'e': math.exp, 'log': math.log10, 'sqrt': math.sqrt}, params) for x in x_values]
            self.plot_widget.plot(x_values, y_values, clear=True).setPen(pg.mkPen(color='#000000'))
            self.plot_widget.plot()

        elif predefined_function:

            if predefined_function == 'sin':

                x = np.linspace(-10, 10, 1000)

                y = np.sin(x)

                self.plot_widget.plot(x, y).setPen(pg.mkPen(color='#000000'))

            elif predefined_function == 'cos':

                x = np.linspace(-10, 10, 1000)

                y = np.cos(x)

                self.plot_widget.plot(x, y).setPen(pg.mkPen(color='#000000'))

            elif predefined_function == 'tan':

                x = np.linspace(-10, 10, 1000)

                y = np.tan(x)

                self.plot_widget.plot(x, y).setPen(pg.mkPen(color='#000000'))

            elif predefined_function == 'x^2':

                x = np.linspace(-10, 10, 1000)

                y = x*x

                self.plot_widget.plot(x, y).setPen(pg.mkPen(color='#000000'))

            elif predefined_function == 'x^3':

                x = np.linspace(-10, 10, 1000)

                y = x*x*x

                self.plot_widget.plot(x, y).setPen(pg.mkPen(color='#000000'))

            elif predefined_function == 'x^4':

                x = np.linspace(-10, 10, 1000)

                y = x*x*x*x

                self.plot_widget.plot(x, y).setPen(pg.mkPen(color='#000000'))

    def select_function(self, index):
        """
        Выбирает функцию из списка
        """
        function = self.function_select_combo.currentText()
        self.function_edit.setText(function)

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()