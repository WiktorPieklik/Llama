ui: mainwindow

mainwindow:
	pyuic5 -x ./designer/mainwindow.ui -o ./src/gui/mainwindow_ui.py

clean_ui:
	rm -f src/gui/mainwindow_ui.py

clean: clean_ui
