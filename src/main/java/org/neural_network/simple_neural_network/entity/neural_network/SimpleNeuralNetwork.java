package org.neural_network.simple_neural_network.entity.neural_network;

import lombok.NonNull;
import org.neural_network.simple_neural_network.entity.NeuronLayer;
import org.neural_network.simple_neural_network.tools.loss.LossType;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class SimpleNeuralNetwork extends NeuralNetwork {

    public SimpleNeuralNetwork(Double LEARNING_RATE, @NonNull List<NeuronLayer> layers, LossType lossType) {
        super(LEARNING_RATE, layers, lossType);
    }

    @Override
    public List<Double> doPrediction(List<Double> featuresVector) {
        List<Double> previousLabels = new ArrayList<>();
        for (int i = 0; i < layers.size() - 1; i++) {
            List<Double> inputData = i == 0 ? featuresVector : previousLabels;
            previousLabels = layers.get(i).doPredication(inputData);
        }
        return layers.getLast().doPredication(previousLabels);
    }

    protected void calculateOutputLayerLossDerivative(List<Double> labels,
                                                      List<Double> expectedLabels) {
        layers.getLast().setLastLossDerivativeByOutputData(labels.parallelStream()
                .map(label ->
                        lossType.getLossFunctionDerivative().calcLoss(expectedLabels.get(labels.indexOf(label)), label) *
                                layers.getLast()
                                        .getNeurons()
                                        .getFirst()
                                        .getActivateFunc() //Раньше здесь была getActivateFunc из слоя
                                        .getActivateFunctionDerivative()
                                        .activate(layers
                                                .getLast()
                                                .getNeurons()
                                                .get(labels.indexOf(label))
                                                .getLastLinearRegression()
                                        )
                )
                .collect(Collectors.toList())
        );
    }
}
