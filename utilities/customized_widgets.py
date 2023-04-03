from PyQt5 import QtWidgets


def group_widget(name=''):
    group = QtWidgets.QGroupBox(name)
    group.setStyleSheet("font: bold Arial 10px")
    return group


def label_widget(name=''):
    label = QtWidgets.QLabel(name)
    label.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
    return label


def lcdnumber_widget():
    lcd = QtWidgets.QLCDNumber()
    lcd.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
    return lcd


def spinbox_widget(range_min, range_max, step, value):
    spinbox = QtWidgets.QSpinBox()
    spinbox.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
    spinbox.setRange(range_min, range_max)
    spinbox.self.QSpinBox_offset_xcenter.setSingleStep(step)
    spinbox.self.QSpinBox_offset_xcenter.setValue(value)
    return spinbox


def doublespinbox_widget(range_min, range_max, step, decimals, value):
    doublespinbox = QtWidgets.QDoubleSpinBox()
    doublespinbox.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
    doublespinbox.setRange(-range_min, range_max)
    doublespinbox.setSingleStep(step)
    doublespinbox.setDecimals(decimals)
    doublespinbox.setValue(value)
    return doublespinbox


def pushbutton_widget(name='', checkable=False):
    button = QtWidgets.QPushButton(name)
    button.setStyleSheet("background-color: grey; color: white; font: bold Arial 10px")
    button_size = button.fontMetrics().boundingRect(button.text()).size()
    button.setFixedSize(button_size.width() + 16, button_size.height() + 12)
    if checkable:
        button.setCheckable(True)
        button.setStyleSheet('QPushButton:checked { background-color: #66CDAA; }')
    return button


def radiobutton_widget(name=''):
    button = QtWidgets.QRadioButton(name)
    button.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
    return button


def combobox_widget():
    combobox = QtWidgets.QComboBox()
    combobox.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
    return combobox


def lineedit_widget():
    lineedit = QtWidgets.QLineEdit()
    lineedit.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
    return lineedit


def checkbox_widget(name=''):
    checkbox = QtWidgets.QCheckBox(name)
    checkbox.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
    return checkbox


def toolbar_widget():
    toolbar = QtWidgets.QToolBar()
    toolbar.setStyleSheet("background-color: dark; color: white; font: bold Arial 12px; width: 36px; height: 18px")
    return toolbar
