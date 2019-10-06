package ru.vo.analyzelit;

import javafx.fxml.FXML;
import javafx.geometry.Rectangle2D;
import javafx.scene.ImageCursor;
import javafx.scene.SnapshotParameters;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.TextInputDialog;
import javafx.scene.image.*;
import javafx.scene.input.MouseButton;
import javafx.scene.input.MouseEvent;
import javafx.scene.paint.Color;
import javafx.stage.Screen;
import javafx.stage.Stage;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PipedInputStream;
import java.util.Optional;

public class Controller extends View {
    @FXML
    private Canvas canvas, canvasCell;
    private GraphicsContext gc;

    private final int CELL_COUNT = 30;
    private final double BRUSH_THICKNESS = 10;

    private ModeEnum mode = ModeEnum.WRITING;
    private ImageCursor pen = new ImageCursor(new Image("/pens.png"));
    private ImageCursor eraser = new ImageCursor(new Image("/eraser.png"));

    private int[] data;

    @FXML
    public void initialize() {
        Stage stage = App.getPrimaryStage();
        stage.widthProperty().addListener((observable, oldValue, newValue) -> {
            resizeCanvas(stage.getWidth() - 37, stage.getHeight() - 200);
            paint(stage);
        });
        stage.heightProperty().addListener((observable, oldValue, newValue) -> {
            resizeCanvas(stage.getWidth() - 37, stage.getHeight() - 200);
            paint(stage);
        });
        data = new int[CELL_COUNT * CELL_COUNT];
        gc = canvas.getGraphicsContext2D();
        canvas.setCursor(pen);
    }

    private void resizeCanvas(double width, double height) {
        canvasCell.setWidth(width);
        canvas.setWidth(width);
        canvasCell.setHeight(height);
        canvas.setHeight(height);
        data = new int[CELL_COUNT * CELL_COUNT];
    }

    private void paint(Stage stage) {
        double width = Screen.getPrimary().getVisualBounds().getWidth() * 10;
        double height = Screen.getPrimary().getVisualBounds().getHeight() * 10;
        GraphicsContext gc = canvasCell.getGraphicsContext2D();
        gc.setLineWidth(0.5);
        gc.clearRect(0, 0, canvasCell.getWidth(), canvasCell.getHeight());

        int cellWidth = (int) (canvas.getWidth() / CELL_COUNT);
        int cellHeight = (int) (canvas.getHeight() / CELL_COUNT);

        for (int i = cellWidth; i < width; i += cellWidth) {
            gc.strokeLine(i, 0, i, height);
        }
        for (int i = cellHeight == 0 ? cellWidth : cellHeight; i < height; i += cellHeight == 0 ? cellWidth : cellHeight) {
            gc.strokeLine(0, i, width, i);
        }
    }

    @FXML
    public void onMousePressed(MouseEvent event) {
        if (mode == ModeEnum.WRITING) {
            gc.setStroke(Color.BLACK);
            gc.setLineWidth(BRUSH_THICKNESS);
            double x = event.getX();
            double y = event.getY();
            gc.beginPath();
            gc.moveTo(x, y);
            gc.stroke();
        } else if (mode == ModeEnum.REMOVAL) {
            double x = event.getX();
            double y = event.getY();
            gc.clearRect(x, y, 10, 10);
        }
    }

    @FXML
    public void onMouseDragged(MouseEvent event) {
        if (mode == ModeEnum.WRITING) {
            double x = event.getX();
            double y = event.getY();
            gc.lineTo(x, y);
            gc.stroke();
            gc.closePath();
            gc.beginPath();
            gc.moveTo(x, y);
        } else if (mode == ModeEnum.REMOVAL) {
            double x = event.getX();
            double y = event.getY();
            gc.clearRect(x, y, 10, 10);
        }
    }

    @FXML
    public void onMouseReleased(MouseEvent event) {
        if (mode == ModeEnum.WRITING) {
            double x = event.getX();
            double y = event.getY();
            gc.lineTo(x, y);
            gc.stroke();
            gc.closePath();
        } else if (mode == ModeEnum.REMOVAL) {
            double x = event.getX();
            double y = event.getY();
            gc.clearRect(x + 10, y + 10, 20, 20);
        }
    }

    @FXML
    public void recognizedOnMouseClicked() {
        try {
            INDArray output = MultiLayerNetwork.load(new File("nn"), false).output(NeuralNetwork.getInput(data));
            char ch = (char) (output.getInt(0) + 195);
            System.out.println(output.getInt(0));
            display(ch + "");
        } catch (IOException exc) {

        }
    }

    @FXML
    public void writingOnMouseClicked(MouseEvent event) {
        mode = ModeEnum.WRITING;
        canvas.setCursor(pen);
    }

    @FXML
    public void removalOnMouseClicked(MouseEvent event) {
        mode = ModeEnum.REMOVAL;
        canvas.setCursor(eraser);
    }

    @FXML
    public void learningOnMouseClicked(MouseEvent event) {
        double width = canvas.getWidth();
        double height = canvas.getHeight();

       // PixelReader pxReader = canvas.snapshot(null, null).getPixelReader();

       // gc.clearRect(0, 0, canvas.getWidth(), canvas.getHeight());

        int cellWidth = (int) (width / CELL_COUNT);
        int cellHeight = (int) (height / CELL_COUNT);

        for (int i = 0; i < data.length; i++) {
            data[i] = 0;
        }

        for (int i = 0; i < width; i += cellWidth) {
            for (int j = 0; j < height; j += cellHeight) {
                SnapshotParameters spp = new SnapshotParameters();
                spp.setViewport(new Rectangle2D(i, j, cellWidth, cellHeight));

                PixelReader pxReader = canvas.snapshot(spp, null).getPixelReader();

                for (int k = 0; k < cellWidth; k++) {
                    for (int p = 0; p < cellHeight; p++) {
                        if (!pxReader.getColor(k, p).toString().equals("0xffffffff")) {
                            gc.setFill(Color.BLUE);
                            gc.fillRect(i, j, cellWidth, cellHeight);
                            data[i + j] = 1;
                        }
                    }
                }
            }
        }

        TextInputDialog dialog = new TextInputDialog();

        dialog.setTitle("Learning");
        dialog.setHeaderText("Input Literal");
        dialog.setContentText("Literal:");

        Optional<String> result = dialog.showAndWait();

        result.ifPresent(name -> {
            MultiLayerNetwork nn = NeuralNetwork.createNN(CELL_COUNT * CELL_COUNT);
            try {
                nn.load(new File("nn"), true);
            } catch (IOException exc) {
                exc.printStackTrace();
            }
            nn.fit(NeuralNetwork.generateDataSet(data, name.charAt(0) - 192));
            try {
                nn.save(new File("nn"));
            } catch (IOException exc) {
                exc.printStackTrace();
            }
        });


    }

    @FXML
    public void clearOnMouseClicked(MouseEvent event) {
        gc.clearRect(0, 0, canvas.getWidth(), canvas.getHeight());
    }
}
