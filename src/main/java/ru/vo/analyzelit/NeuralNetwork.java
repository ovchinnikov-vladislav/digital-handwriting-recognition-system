package ru.vo.analyzelit;

import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;

public class NeuralNetwork {

    public static MultiLayerNetwork createNN(int numInputs) {
        int seed = 2345;
        double learningRate = 0.05;

        int numHiddenNodes = 1500;
        int numOutputs = 33;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(learningRate, 0.9))
                .list()
                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.L2)
                        .activation(Activation.IDENTITY)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        return net;
    }

    public static DataSet generateDataSet(int[] data, int result) {
        INDArray input = Nd4j.zeros(data.length, data.length);
        INDArray output = Nd4j.zeros(data.length, 33);

        for (int index = 0; index < data.length; index++) {
            for (int i = 0; i < data.length; i++) {
                input.put(index, i, data[i]);
            }
            for (int i = 0; i < 33; i++) {
                if (i == result)
                    output.put(index, i, result);
                else
                    output.put(index, i, 0);
            }

        }
        return new DataSet(input, output);
    }

    public static INDArray getInput(int[] data) {
        INDArray input = Nd4j.zeros(data.length, data.length);
        for (int index = 0; index < data.length; index++) {
            for (int i = 0; i < data.length; i++) {
                input.put(index, i, data[i]);
            }
        }
        return input;
    }

    private static double scaleXY(int i, int maxI){
        return (double) i / (double) (maxI - 1) -0.5;
    }
}
