package ru.vo.analyzelit;

import javafx.fxml.FXML;
import javafx.scene.control.TextField;

public class View {
    @FXML
    private TextField textLiteral;

    public void display(String value) {
        textLiteral.setText(value);
    }
}
