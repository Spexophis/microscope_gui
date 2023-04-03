from PyQt5 import QtWidgets, QtGui


def group_widget(name=''):
    group = QtWidgets.QGroupBox(name)
    group.setStyleSheet('''
        QGroupBox {
            background-color: #444444;
            border: 2px solid #444444;
            border-radius: 4px;
            margin-top: 1ex;
            font-weight: bold;
            font-size: 12px;
            color: #CCCCCC;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 3px;
        }
    ''')
    return group


def label_widget(name=''):
    label = QtWidgets.QLabel(name)
    label.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
    # label.setStyleSheet("background-color: dark; color: #f0f0f0; padding: 2px")
    label.setStyleSheet('background-color: #444444; color: #ECECEC; padding: 5px; border-radius: 5px;')
    return label


def lcdnumber_widget(num=None, n=None):
    lcd = QtWidgets.QLCDNumber()
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(255, 255, 255))  # set text color
    palette.setColor(QtGui.QPalette.Background, QtGui.QColor(37, 37, 38))  # set background color
    palette.setColor(QtGui.QPalette.Light, QtGui.QColor(64, 64, 64))  # set light color
    palette.setColor(QtGui.QPalette.Dark, QtGui.QColor(26, 26, 27))  # set dark color
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(0, 85, 255))  # set highlight color
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(255, 255, 255))  # set highlighted text color
    lcd.setPalette(palette)
    lcd.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
    lcd.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
    # lcd.setStyleSheet('''
    #     QLCDNumber {
    #     background-color: black;
    #     color: white;
    #     border: 2px solid grey;
    #     }
    # ''')
    lcd.setDecMode()
    if num is not None:
        lcd.display(num)
    if n is not None:
        lcd.setDigitCount(n)
    lcd.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
    lcd.setMinimumWidth(lcd.sizeHint().width())
    return lcd


def spinbox_widget(range_min, range_max, step, value):
    spinbox = QtWidgets.QSpinBox()
    spinbox.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
    spinbox.setStyleSheet('''
        QSpinBox { 
        background-color: white; 
        color: black; 
        border: 1px solid grey; 
        } 
    ''')
    spinbox.setRange(range_min, range_max)
    spinbox.setSingleStep(step)
    spinbox.setValue(value)
    spinbox.setMaximum(range_max)
    spinbox.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
    spinbox.setMinimumWidth(spinbox.sizeHint().width())
    spinbox.setMinimumHeight(spinbox.sizeHint().height())
    return spinbox


def doublespinbox_widget(range_min, range_max, step, decimals, value):
    doublespinbox = QtWidgets.QDoubleSpinBox()
    doublespinbox.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
    # doublespinbox.setStyleSheet('''
    #     QDoubleSpinBox {
    #     background-color: white;
    #     color: black;
    #     border: 1px solid grey;
    #     } 
    # ''')
    # Set the dark palette
    dark_palette = QtGui.QPalette()
    dark_palette.setColor(QtGui.QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    spin_box.setPalette(dark_palette)
    doublespinbox.setRange(-range_min, range_max)
    doublespinbox.setSingleStep(step)
    doublespinbox.setDecimals(decimals)
    doublespinbox.setValue(value)
    doublespinbox.setMaximum(range_max)
    doublespinbox.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
    doublespinbox.setMinimumWidth(doublespinbox.sizeHint().width())
    doublespinbox.setMinimumHeight(doublespinbox.sizeHint().height())
    return doublespinbox


def pushbutton_widget(name='', checkable=False):
    button = QtWidgets.QPushButton(name)
    button.setCheckable(checkable)
    button.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
    button.setStyleSheet('''
        QPushButton {
            background-color: #282828;
            border-style: outset;
            border-radius: 4px;
            color: #FFFFFF;
            padding: 4px;
        }
        
        QPushButton:hover {
            background-color: #3e8e41;
        }
        
        QPushButton:pressed {
            background-color: #2e6d3b;
            border-style: inset;
        }
        
        QPushButton:checked {
            background-color: #2e6d3b;
            border-style: inset;
        }
    ''')
    button_size = button.fontMetrics().boundingRect(button.text()).size()
    button.setFixedSize(button_size.width() + 24, button_size.height() + 16)
    return button


def checkbox_widget(name=''):
    checkbox = QtWidgets.QCheckBox(name)
    checkbox.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
    checkbox.setStyleSheet('''
        QCheckBox {
            background-color: qradialgradient(cx: 0.5, cy: 0.5, radius: 0.5, fx: 0.5, fy: 0.5, 
                stop: 0.8 #fff, stop: 0.9 #ddd);
            color: #000;
            font: 14px;
            padding: 5px;
            border: 2px solid #aaa;
            border-radius: 20px;
        }
        QCheckBox::indicator {
            width: 25px;
            height: 25px;
            border-radius: 20px;
            border: 2px solid #aaa;
            background-color: qradialgradient(cx: 0.5, cy: 0.5, radius: 0.5, fx: 0.5, fy: 0.5, 
                stop: 0.8 #fff, stop: 0.9 #ddd);
        }
        QCheckBox::indicator:checked {
            background-color: qradialgradient(cx: 0.5, cy: 0.5, radius: 0.5, fx: 0.5, fy: 0.5, 
                stop: 0.8 #9af, stop: 0.9 #8bf);
            border: 2px solid #8bf;
        }
    ''')
    checkbox.setChecked(False)
    checkbox.setCheckable(True)
    return checkbox

def radiobutton_widget(name=''):
    radiobutton = QtWidgets.QRadioButton(name)
    radiobutton.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
    # set the palette to use the Fusion style
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(200, 200, 200))  # set text color
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))  # set background color
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))  # set button color
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(142, 45, 197))  # set highlight color
    radiobutton.setPalette(palette)

    # set the style sheet
    radiobutton.setStyleSheet('''
    QRadioButton {
        background-color: #444444;
        color: white;
    }
    
    QRadioButton::indicator {
        width: 18px;
        height: 18px;
    }

    QRadioButton::indicator::unchecked {
        border: 2px solid rgb(200, 200, 200);
        border-radius: 8px;
    }

    QRadioButton::indicator::checked {
        background-color: rgb(192, 255, 62);
        border: 2px solid rgb(192, 255, 62);
        border-radius: 8px;
    }''')
    return radiobutton


def combobox_widget(list_items):
    combobox = QtWidgets.QComboBox()
    for item in list_items:
        combobox.addItem(item)
    combobox.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
    combobox.setStyle(QtWidgets.QStyleFactory.create('Fusion'))
    combobox.setStyleSheet('QComboBox {background-color: #454545; color: #f0f0f0;}'
                            'QComboBox::drop-down {background-color: #454545;}'
                            'QComboBox::down-arrow {image: url(down_arrow.png);}')
    return combobox


def lineedit_widget():
    lineedit = QtWidgets.QLineEdit()
    lineedit.setStyleSheet("background-color: dark; color: white; font: bold Arial 12px")
    return lineedit


def toolbar_widget():
    toolbar = QtWidgets.QToolBar()
    toolbar.setStyleSheet("background-color: dark; color: white; font: bold Arial 12px; width: 36px; height: 18px")
    return toolbar
