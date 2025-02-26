package org.neural_network.simple_neural_network.entity.neural_network;

import lombok.NonNull;
import org.neural_network.simple_neural_network.entity.Neuron;
import org.neural_network.simple_neural_network.entity.NeuronLayer;
import org.neural_network.simple_neural_network.tools.MathFunc;
import org.neural_network.simple_neural_network.tools.loss.LossType;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class SoftmaxNeuralNetwork extends NeuralNetwork {

    private Double maxZ;

    public SoftmaxNeuralNetwork(Double LEARNING_RATE, @NonNull List<NeuronLayer> layers, LossType lossType) {
        super(LEARNING_RATE, layers, lossType);
    }

    @Override
    public List<Double> doPrediction(List<Double> featuresVector) {
        List<Double> previousLabels = new ArrayList<>();
        for (int i = 0; i < layers.size() - 1; i++) {
            List<Double> inputData = i == 0 ? featuresVector : previousLabels;
            previousLabels = layers.get(i).doPredication(inputData);
        }
        return softmaxPredication(layers.getLast(), previousLabels);
    }

    private List<Double> softmaxPredication(NeuronLayer layer, List<Double> featuresVector) {
        List<Double> otherNeuronZ = layers.getLast()
                .getNeurons()
                .stream()
                .map(neuron -> neuron.linearRegression(featuresVector))
                .collect(Collectors.toList());
        layer.setLastInputData(featuresVector);
        this.maxZ = otherNeuronZ.stream().max(Double::compare).orElseThrow();
        return layer.getNeurons().parallelStream()
                .map(neuron -> neuron.getSoftmaxPredication(featuresVector, otherNeuronZ, maxZ))
                .collect(Collectors.toList());
    }

    @Override
    protected void calculateOutputLayerLossDerivative(List<Double> labels, List<Double> expectedLabels) {
        layers.getLast().setLastLossDerivativeByOutputData(labels.parallelStream()
                .map(label ->
                        lossType.equals(LossType.LOG_LOSS) ?
                                MathFunc.softmaxLogLossDerivative(label, expectedLabels.get(labels.indexOf(label))) :
                                (lossType.getLossFunctionDerivative().calcLoss(expectedLabels.get(labels.indexOf(label)), label) *
                                MathFunc.softmaxDerivative(
                                        layers.getLast()
                                                .getNeurons()
                                                .get(labels.indexOf(label))
                                                .getLastLinearRegression(),
                                        layers.getLast().getNeurons().stream().map(Neuron::getLastLinearRegression).toList(),
                                        maxZ
                                )
                                )
                )
                .collect(Collectors.toList()));
    }
}
